# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,  # Transformer 的通道维度。
            transformer: nn.Module,  # 用于预测蒙版的 Transformer 模块。
            num_multimask_outputs: int = 3,  # 在消除蒙版歧义时要预测的蒙版数量，默认为 3。
            activation: Type[nn.Module] = nn.GELU,  # 在放大蒙版时使用的激活函数类型
            iou_head_depth: int = 3,  # 用于预测蒙版质量的多层感知器（MLP）的深度。
            iou_head_hidden_dim: int = 256,  # 用于预测蒙版质量的 MLP 的隐藏维度。
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(  # 用于将输出向量转换为与输入图像相同大小的蒙版。
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )  # 一个包含多个多层感知器（MLP）的模块列表，用于生成蒙版的超网络。每个蒙版标记都有一个对应的 MLP，用于生成与之关联的蒙版。

        self.iou_prediction_head = MLP(  # 用于预测蒙版质量的 MLP。
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,  # 与 image_embeddings 相同形状的位置编码。
            sparse_prompt_embeddings: torch.Tensor,  # 点和框的嵌入。稀疏
            dense_prompt_embeddings: torch.Tensor,  # 蒙版输入的嵌入。密集
            multimask_output: bool,  # 指示是否返回多个蒙版或单个蒙版。
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 第一个张量是批量预测的蒙版。第二个张量是蒙版质量的预测值。
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(  # 接收一些输入张量，包括图像嵌入、位置编码以及稀疏和密集提示的嵌入，并返回预测的蒙版以及蒙版质量的预测值。
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0],
                                      dim=0)  # 将图像嵌入的张量在批量维度上进行重复，以匹配 tokens 的形状中的批量大小。
        src = src + dense_prompt_embeddings  # 将扩展后的图像嵌入与密集提示嵌入相加，以将密集提示信息添加到图像嵌入中。
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)  # 将位置编码张量在批量维度上进行重复，以匹配 tokens 的形状中的批量大小。
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)  # 将扩展后的图像嵌入、位置编码和标记作为输入传递给 Transformer 模型，并运行它以获取输出。
        # hs 是 Transformer 的所有隐藏状态，src 是 Transformer 的最终输出。
        iou_token_out = hs[:, 0, :]  # 从 Transformer 的隐藏状态中选择 IOU 标记的输出。
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]  # 从 Transformer 的隐藏状态中选择蒙版标记的输出。

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)  # 对 Transformer 输出的张量进行转置操作，以便将通道维度移动到正确的位置。
        upscaled_embedding = self.output_upscaling(src)  # 对转置后的张量进行上采样，以将其转换为与输入图像相同大小的蒙版。
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            # 使用与该蒙版标记对应的超网络 MLP 来处理蒙版嵌入。这将产生一个张量列表，其中每个元素都是与对应的蒙版标记相关联的预测蒙版的表示。
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)  # 将处理后的蒙版嵌入张量列表堆叠在一起，以形成最终的超网络输入张量。
        b, c, h, w = upscaled_embedding.shape
        # 将上采样后的蒙版嵌入与超网络输入进行矩阵乘法，以生成预测的蒙版。
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        # 使用 IOU 标记的输出作为输入，通过蒙版质量预测头部的 MLP 来预测蒙版质量。
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
