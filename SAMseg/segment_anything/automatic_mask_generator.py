# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from .modeling import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamAutomaticMaskGenerator:
    def __init__(
            self,
            model: Sam,
            points_per_side: Optional[int] = 64,  # 沿图像一侧要采样的点的数量。
            points_per_batch: int = 128,  # 设置模型同时检测的点数。
            pred_iou_thresh: float = 0.86,  # [0,1]中的滤波阈值，使用模型的预测掩码质量。
            stability_score_thresh: float = 0.92,  # [0,1]中的滤波阈值，使用掩码在截断值变化下的稳定性，对模型的掩码预测进行二值化
            stability_score_offset: float = 1.0,  # 计算稳定性分数时，偏移截止值的量。
            box_nms_thresh: float = 0.7,  # 非最大抑制用于过滤重复掩码的框IoU截止。
            crop_n_layers: int = 1,  # 如果>0，将对图像的裁剪再次运行掩膜预测。
            crop_nms_thresh: float = 0.7,  # 非最大抑制用于过滤不同物体之间的重复掩码的框IoU截止。
            crop_overlap_ratio: float = 512 / 1500,  # 设置物体重叠的程度。
            crop_n_points_downscale_factor: int = 2,  # 在层n中采样的每侧的点数按比例缩小
            point_grids: Optional[List[np.ndarray]] = None,  # 用于采样的点的显式网格上的列表，归一化为[0,1]。
            min_mask_region_area: int = 25,  # 如果>0，将应用后处理来移除面积小于min_mask_region_area的掩膜来中断开连接的区域和孔。
            output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
                point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """
        # # ---------------Generate feat------------------
        # self.predictor.set_image(image)
        # feat = self.predictor.features
        # # ----------------------------------------------

        # Generate masks
        mask_data = self._generate_masks(image)  # 该方法使用输入的图像生成掩码数据。

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:  # 对掩码进行后处理，以过滤小的断开区域和孔。
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # # ------------------------------------------------------------------------------------
        # # Write mask records
        # curr_anns = []  # 用于将生成的掩码记录写入列表中，并返回该列表。
        # for idx in range(len(mask_data["segmentations"])):  # 遍历了生成的每个掩码，为每个掩码创建一个字典记录。
        #     ann = {
        #         "segmentation": mask_data["segmentations"][idx],  # 掩码的编码（可以是二进制掩码或 RLE 编码）。
        #         "area": area_from_rle(mask_data["rles"][idx]),  # 掩码的面积。
        #         "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),  # 掩码周围的边界框，以 XYWH 格式表示。
        #         "predicted_iou": mask_data["iou_preds"][idx].item(),  # 模型对掩码质量的预测 IoU 值。
        #         "point_coords": [mask_data["points"][idx].tolist()],  # 用于生成掩码的点坐标。
        #         "stability_score": mask_data["stability_score"][idx].item(),  # 掩码的稳定性分数。
        #         "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),  # 生成掩码时使用的图像裁剪框。
        #     }
        #     curr_anns.append(ann)
        # # ------------------------------------------------------------------------------------

        # # ----------------------获取按 iou_preds 升序排列的索引---------------------------------
        # iou_preds_tensor = torch.tensor(mask_data["iou_preds"])
        # # 获取按 iou_preds 升序排列的索引
        # sorted_indices = torch.argsort(iou_preds_tensor)
        # # 使用排序的索引重排 mask_data
        # mask_data_sorted = {
        #     "segmentations": [mask_data["segmentations"][idx] for idx in sorted_indices],
        #     "rles": [mask_data["rles"][idx] for idx in sorted_indices],
        #     "boxes": [mask_data["boxes"][idx] for idx in sorted_indices],
        #     "iou_preds": [mask_data["iou_preds"][idx] for idx in sorted_indices],
        #     "points": [mask_data["points"][idx] for idx in sorted_indices],
        #     "stability_score": [mask_data["stability_score"][idx] for idx in sorted_indices],
        #     "crop_boxes": [mask_data["crop_boxes"][idx] for idx in sorted_indices],
        # }
        #
        # # Write mask records
        # curr_anns = []  # 用于将生成的掩码记录写入列表中，并返回该列表。
        # for idx in range(len(mask_data_sorted["segmentations"])):  # 遍历了生成的每个掩码，为每个掩码创建一个字典记录。
        #     ann = {
        #         "segmentation": mask_data_sorted["segmentations"][idx],
        #         "area": area_from_rle(mask_data_sorted["rles"][idx]),
        #         "bbox": box_xyxy_to_xywh(mask_data_sorted["boxes"][idx]).tolist(),
        #         "predicted_iou": mask_data_sorted["iou_preds"][idx].item(),
        #         "point_coords": [mask_data_sorted["points"][idx].tolist()],
        #         "stability_score": mask_data_sorted["stability_score"][idx].item(),
        #         "crop_box": box_xyxy_to_xywh(mask_data_sorted["crop_boxes"][idx]).tolist(),
        #     }
        #     curr_anns.append(ann)
        # # ------------------------------------------------------------------------------------

        # ----------------------------------------计算每个掩码的面积-------------------------------------
        area_tensor = torch.tensor([area_from_rle(rle) for rle in mask_data["rles"]])
        # 获取按面积降序排列的索引
        sorted_indices = torch.argsort(area_tensor, descending=True)
        # 使用排序的索引重排 mask_data
        mask_data_sorted = {
            "segmentations": [mask_data["segmentations"][idx] for idx in sorted_indices],
            "rles": [mask_data["rles"][idx] for idx in sorted_indices],
            "boxes": [mask_data["boxes"][idx] for idx in sorted_indices],
            "iou_preds": [mask_data["iou_preds"][idx] for idx in sorted_indices],
            "points": [mask_data["points"][idx] for idx in sorted_indices],
            "stability_score": [mask_data["stability_score"][idx] for idx in sorted_indices],
            "crop_boxes": [mask_data["crop_boxes"][idx] for idx in sorted_indices],
        }

        # Write mask records
        curr_anns = []  # 用于将生成的掩码记录写入列表中，并返回该列表。
        for idx in range(len(mask_data_sorted["segmentations"])):  # 遍历了生成的每个掩码，为每个掩码创建一个字典记录。
            ann = {
                "segmentation": mask_data_sorted["segmentations"][idx],
                "area": area_tensor[sorted_indices[idx]].item(),  # 使用预计算的面积
                "bbox": box_xyxy_to_xywh(mask_data_sorted["boxes"][idx]).tolist(),
                "predicted_iou": mask_data_sorted["iou_preds"][idx].item(),
                "point_coords": [mask_data_sorted["points"][idx].tolist()],
                "stability_score": mask_data_sorted["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data_sorted["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)
        # ------------------------------------------------------------------------------------

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(  # NMS 算法会根据一定的规则选择优先级较高的掩码框，并将其他重叠的掩码框过滤掉。
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data

    def _process_crop(
            self,
            image: np.ndarray,
            crop_box: List[int],  # 裁剪框的边界坐标，格式为 [x0, y0, x1, y1]
            crop_layer_idx: int,  # 裁剪框所在的裁剪层级索引。
            orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
            self,
            points: np.ndarray,
            im_size: Tuple[int, ...],
            crop_box: List[int],
            orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU  从 data 中筛选出预测的 IoU 大于该阈值的掩码。
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score,调用 calculate_stability_score 函数计算每个掩码的稳定性分数。
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
            mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data
