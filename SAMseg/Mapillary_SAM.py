import os
import json
import numpy as np
import torch
import cv2
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm
from SAM_superpixels import sam_superpixels  # 请确保这个模块已正确实现


def setup_distributed():
    """初始化分布式训练环境"""
    dist.init_process_group(backend='nccl')  # 使用NCCL后端
    torch.cuda.set_device(dist.get_rank())  # 设置当前进程使用的GPU


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


class MapillaryDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        """
        Mapillary数据集加载器
        :param root: 数据集根目录
        :param mode: 数据集模式（train/test/val）
        :param transform: 数据转换
        """
        self.root = root
        self.mode = mode
        self.transform = transform
        self.images = []

        # 根据Mapillary目录结构设置路径
        if mode == 'train':
            img_dir = os.path.join(root, 'training', 'images')
        elif mode == 'test':
            img_dir = os.path.join(root, 'testing', 'images')
        elif mode == 'val':
            img_dir = os.path.join(root, 'validation', 'images')
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'test', or 'val'.")

        # 遍历目录，收集图像文件
        for filename in os.listdir(img_dir):
            if filename.endswith('.jpg'):
                self.images.append(os.path.join(img_dir, filename))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, img_path


def visualize_mask(combined_mask, random_color=True):
    """生成可视化掩码图像"""
    unique_labels = np.unique(combined_mask)

    # 创建颜色映射
    if random_color:
        colors = np.random.rand(len(unique_labels), 4)
        colors[:, 3] = 0.6
    else:
        colors = np.zeros((len(unique_labels), 4))
        colors[:, :3] = [30 / 255, 144 / 255, 255 / 255]
        colors[:, 3] = 0.6

    h, w = combined_mask.shape
    mask_img = np.zeros((h, w, 4))
    for i, label in enumerate(unique_labels):
        if label > 0:
            mask_img[combined_mask == label] = colors[i]

    return (mask_img[:, :, :3] * 255).astype(np.uint8)


def save_outputs(combined_mask, img_path, mask_dir, vis_dir):
    """保存结果文件"""
    # 生成文件名
    base_name = os.path.basename(img_path).replace('.jpg', '')

    # 保存JSON掩码
    mask_path = os.path.join(mask_dir, f"{base_name}.json")
    with open(mask_path, 'w') as f:
        json.dump(combined_mask.tolist(), f)

    # 保存可视化图像
    vis_img = visualize_mask(combined_mask)
    vis_path = os.path.join(vis_dir, f"{base_name}.png")
    cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))


def main():
    # 初始化分布式环境
    setup_distributed()

    # 路径配置
    data_root = "/data/Mapillary/"
    save_root = "/data/Mapillary_SAM/"

    # 创建输出目录
    mask_save_dir = os.path.join(save_root, "Mapillary_sam_train/Mapillary_SAM_mask/")
    vis_save_dir = os.path.join(save_root, "Mapillary_sam_train/Mapillary_SAM/")
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    # 加载SAM模型
    sam_checkpoint = "/checkpoints/sam_vit_h.pth"
    model_type = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)  # 注册模型
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # 创建数据加载器
    dataset = MapillaryDataset(data_root, mode='train')  # 修改为'test'或'val'以处理其他数据集
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            sampler=sampler,
                            num_workers=4,
                            pin_memory=True)

    # 处理数据
    for batch in tqdm(dataloader, desc="Processing Images"):
        images, img_paths = batch
        for img, path in zip(images, img_paths):
            # 转换图像格式
            img_np = img.numpy().astype(np.uint8)

            # 生成掩码
            masks = mask_generator.generate(img_np)
            combined_mask = sam_superpixels(img_np, masks)

            # 保存结果
            save_outputs(combined_mask, path, mask_save_dir, vis_save_dir)

            # 释放显存
            del masks, combined_mask
            torch.cuda.empty_cache()
    # 清理环境
    cleanup_distributed()


if __name__ == "__main__":
    # 运行命令（单卡示例）：
    # CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 Mapillary_SAM.py
    main()


# import os
# import json
# import numpy as np
# import torch
# import cv2
# import torch.distributed as dist
# from torch.utils.data import Dataset, DataLoader, DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# from tqdm import tqdm
# from SAM_superpixels import sam_superpixels  # 请确保这个模块已正确实现
#
#
# def setup_distributed(backend='nccl'):
#     """初始化分布式训练环境"""
#     dist.init_process_group(backend=backend)  # 使用NCCL后端
#     torch.cuda.set_device(dist.get_rank())  # 设置当前进程使用的GPU
#
#
# def cleanup_distributed():
#     """清理分布式环境"""
#     dist.destroy_process_group()
#
#
# class MapillaryDataset(Dataset):
#     def __init__(self, root, mode='train', transform=None):
#         """
#         Mapillary数据集加载器
#         :param root: 数据集根目录
#         :param mode: 数据集模式（train/test/val）
#         :param transform: 数据转换
#         """
#         self.root = root
#         self.mode = mode
#         self.transform = transform
#         self.images = []
#
#         # 根据Mapillary目录结构设置路径
#         if mode == 'train':
#             img_dir = os.path.join(root, 'training', 'images')
#         elif mode == 'test':
#             img_dir = os.path.join(root, 'testing', 'images')
#         elif mode == 'val':
#             img_dir = os.path.join(root, 'validation', 'images')
#         else:
#             raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'test', or 'val'.")
#
#         # 遍历目录，收集图像文件
#         for filename in os.listdir(img_dir):
#             if filename.endswith('.jpg'):
#                 self.images.append(os.path.join(img_dir, filename))
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         img_path = self.images[idx]
#         image = cv2.imread(img_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         if self.transform:
#             image = self.transform(image)
#         return image, img_path
#
#
# def visualize_mask(combined_mask, random_color=True):
#     """生成可视化掩码图像"""
#     unique_labels = np.unique(combined_mask)
#
#     # 创建颜色映射
#     if random_color:
#         colors = np.random.rand(len(unique_labels), 4)
#         colors[:, 3] = 0.6
#     else:
#         colors = np.zeros((len(unique_labels), 4))
#         colors[:, :3] = [30 / 255, 144 / 255, 255 / 255]
#         colors[:, 3] = 0.6
#
#     h, w = combined_mask.shape
#     mask_img = np.zeros((h, w, 4))
#     for i, label in enumerate(unique_labels):
#         if label > 0:
#             mask_img[combined_mask == label] = colors[i]
#
#     return (mask_img[:, :, :3] * 255).astype(np.uint8)
#
#
# def save_outputs(combined_mask, img_path, mask_dir, vis_dir):
#     """保存结果文件"""
#     # 生成文件名
#     base_name = os.path.basename(img_path).replace('.jpg', '')
#
#     # 保存JSON掩码
#     mask_path = os.path.join(mask_dir, f"{base_name}.json")
#     with open(mask_path, 'w') as f:
#         json.dump(combined_mask.tolist(), f)
#
#     # 保存可视化图像
#     vis_img = visualize_mask(combined_mask)
#     vis_path = os.path.join(vis_dir, f"{base_name}.png")
#     cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
#
#
# def main():
#     # 初始化分布式环境
#     setup_distributed()
#
#     # 获取当前进程的rank
#     rank = dist.get_rank()
#     print(f"Rank {rank} is using GPU {torch.cuda.current_device()}")
#
#     # 路径配置
#     data_root = "/data3/qinghua/qinghua_before/UDA/data/Mapillary/"
#     save_root = "/data2/qinghua/ZDJ/datasets/Mapillary_SAM/"
#
#     # 创建输出目录（仅在主进程中创建）
#     if rank == 0:
#         mask_save_dir = os.path.join(save_root, "Mapillary_sam_train/Mapillary_SAM_mask/")
#         vis_save_dir = os.path.join(save_root, "Mapillary_sam_train/Mapillary_SAM/")
#         os.makedirs(mask_save_dir, exist_ok=True)
#         os.makedirs(vis_save_dir, exist_ok=True)
#     dist.barrier()  # 同步所有进程
#
#     # 加载SAM模型
#     sam_checkpoint = "/data3/qinghua/qinghua_before/ZDJ/models/SAM/sam_vit_h.pth"
#     model_type = "vit_h"
#     device = torch.device(f"cuda:{rank}")  # 每个进程使用对应的GPU
#     sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
#     sam = DDP(sam, device_ids=[rank])  # 使用DDP包装模型
#
#     # 使用 sam.module 访问原始模型
#     mask_generator = SamAutomaticMaskGenerator(sam.module)
#
#     # 创建数据加载器
#     dataset = MapillaryDataset(data_root, mode='train')  # 修改为'test'或'val'以处理其他数据集
#     sampler = DistributedSampler(dataset)  # 分布式采样器
#     dataloader = DataLoader(dataset,
#                             batch_size=1,
#                             sampler=sampler,
#                             num_workers=4,
#                             pin_memory=True)
#
#     # 处理数据
#     for batch in tqdm(dataloader, desc=f"Rank {rank} Processing Images"):
#         images, img_paths = batch
#         for img, path in zip(images, img_paths):
#             # 转换图像格式
#             img_np = img.numpy().astype(np.uint8)
#
#             # 生成掩码
#             masks = mask_generator.generate(img_np)
#             combined_mask = sam_superpixels(img_np, masks)
#
#             # 保存结果（仅在主进程中保存）
#             if rank == 0:
#                 save_outputs(combined_mask, path, mask_save_dir, vis_save_dir)
#
#             # 释放显存
#             del masks, combined_mask
#             torch.cuda.empty_cache()
#
#     # 清理环境
#     cleanup_distributed()
#
#
# if __name__ == "__main__":
#     main()
#
# # CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12245 Mapillary_SAM.py

