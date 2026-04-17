import torch.nn.functional as F
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
from SAM_superpixels import sam_superpixels
import time
from segment_anything.utils.amg import rle_to_mask

# 设置环境变量，指定使用 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 检查是否检测到了 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")


def apply_mask_to_superpixels(superpixel_segment, mask):
    masked_superpixel_segment = np.zeros_like(superpixel_segment)
    # 只保留掩码区域的超像素标签
    masked_superpixel_segment[mask > 0] = superpixel_segment[mask > 0]
    return masked_superpixel_segment


def generate_superpixels(image, n_segments, slic_scale_factor=1.0):
    """Generate superpixels.
    """
    image = (image * 255).astype(np.uint8)
    if slic_scale_factor != 1:
        dsize = (int(image.shape[0] * slic_scale_factor),
                 int(image.shape[1] * slic_scale_factor))
        image = cv2.resize(
            image, dsize=dsize, interpolation=cv2.INTER_LINEAR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        image.shape[1], image.shape[0], 3, num_superpixels=n_segments, num_levels=1, prior=2,
        histogram_bins=5, double_step=False)
    seeds.iterate(image, num_iterations=15)
    segment = seeds.getLabels()

    return segment


def visualize_non_masked_area_sam(image, sam_masks):
    # 创建一个全白的背景
    result_image = np.ones_like(image) * 255  # 初始设置为全白（背景）
    # 将所有掩码部分设为黑色
    for mask in sam_masks:
        mask = mask['segmentation']  # 获取掩码的二进制数据
        for i in range(3):  # 对每个通道 (R, G, B)
            result_image[:, :, i] = np.where(mask == 1, 0, result_image[:, :, i])  # 掩码部分变黑色
    # 显示结果图像
    plt.imshow(result_image)
    plt.axis('off')  # 隐藏坐标轴
    plt.show()


def extract_non_masked_area(masks, image):
    image_shape = image.shape
    # 初始化一个全0的总掩码
    total_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    # 合并所有的SAM掩码
    for mask in masks:
        total_mask = np.logical_or(total_mask, mask['segmentation']).astype(np.uint8)
    # 反转掩码，得到不在SAM掩码中的部分
    non_masked_mask = np.logical_not(total_mask).astype(np.uint8)
    return non_masked_mask


def combine_masks(image, masks, superpixel_segment):
    label_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    # 将每个掩码区域分配一个独特的标签
    for i, mask in enumerate(masks):
        label_map[mask['segmentation'] > 0] = i + 1  # 标签从 1 开始

    # Find the next available label for superpixel segments
    next_label = len(masks) + 1

    # Assign superpixel segments to the label map with labels starting from next_label
    superpixel_nonzero = superpixel_segment > 0
    label_map[superpixel_nonzero] = superpixel_segment[superpixel_nonzero] + next_label - 1

    return label_map


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_anns(masks):
    for i, mask in enumerate(masks):
        show_mask(mask['segmentation'], plt.gca(), True)


def show_combined_mask(combined_mask, random_color=False):
    """Display the combined mask with color visualization."""
    unique_labels = np.unique(combined_mask)

    # Create a color map
    if random_color:
        # Generate random colors for each unique label
        colors = np.random.rand(len(unique_labels), 4)  # RGBA
        colors[:, 3] = 0.6  # Set alpha to 0.6
    else:
        # Fixed color for visualization
        colors = np.zeros((len(unique_labels), 4))
        colors[:, :3] = [30 / 255, 144 / 255, 255 / 255]  # RGB
        colors[:, 3] = 0.6  # Alpha

    # Create an RGBA image to show the mask
    h, w = combined_mask.shape
    mask_image = np.zeros((h, w, 4))  # RGBA image
    for i, label in enumerate(unique_labels):
        if label > 0:  # Skip background or zero label
            mask_image[combined_mask == label] = colors[i]

    plt.imshow(mask_image)
    plt.axis('off')
    plt.show()


image = cv2.imread(
    "/data/dongjun/datasets/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"
)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "/home/dongjun/zdj/model/SAM/sam_vit_h.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)  # 注册模型
sam.to(device=device)

# predictor = SamPredictor(sam)
# predictor.set_image(image)
# feat = predictor.features
mask_generator = SamAutomaticMaskGenerator(sam)  # 生成sam预测对象
masks = mask_generator.generate(image)
print(len(masks))

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('on')
# plt.show()


label_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
for i, mask in enumerate(masks):
    label_map[mask['segmentation'] > 0] = i + 1  # 标签从 1 开始
show_combined_mask(label_map, random_color=True)
print()
