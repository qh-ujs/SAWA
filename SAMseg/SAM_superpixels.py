# import pickle
# import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
import torch


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


def visualize_non_masked_area_sam(image, sam_masks):  # 显示SAM未分割的部分
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


def apply_mask_to_superpixels(superpixel_segment, mask):
    masked_superpixel_segment = np.zeros_like(superpixel_segment)
    # 只保留掩码区域的超像素标签
    masked_superpixel_segment[mask > 0] = superpixel_segment[mask > 0]

    # 找到掩码区域内的唯一超像素标签
    unique_labels = np.unique(masked_superpixel_segment[masked_superpixel_segment > 0])

    # 为每个唯一标签分配新的逐1增加的数值
    label_mapping = {label: i + 1 for i, label in enumerate(unique_labels)}

    # 使用新的数值更新 masked_superpixel_segment
    for old_label, new_label in label_mapping.items():
        masked_superpixel_segment[masked_superpixel_segment == old_label] = new_label
    return masked_superpixel_segment


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


def sam_superpixels(image, masks):
    non_masked_mask = extract_non_masked_area(masks, image)  # 提取SAM未分割出来的部分
    n_segments = int(image.shape[0] * image.shape[1] / (16 ** 2)) # 希望生成的超像素数量
    # 生成超像素分割
    superpixel_segment = generate_superpixels(image, n_segments)  # 图片进行超像素

    masked_superpixel_segment = apply_mask_to_superpixels(superpixel_segment, non_masked_mask)  # 将SAM没有分割出来的部分进行超像素

    combined_mask = combine_masks(image, masks, masked_superpixel_segment)  # 结合超像素和 masks
    return combined_mask


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

    plt.figure(figsize=(20, 20))
    plt.imshow(mask_image)
    plt.axis('off')
    plt.show()


def load_mask_and_feat(filename, mask_dir, feat_dir):
    # 直接使用文件名，而不移除扩展名
    mask_file_name = filename + '.json'
    feat_file_name = filename + '.json'

    # 构造完整的文件路径
    mask_file_path = os.path.join(mask_dir, mask_file_name)
    feat_file_path = os.path.join(feat_dir, feat_file_name)

    # 加载 combined_mask
    with open(mask_file_path, 'r') as f:
        combined_mask = np.array(json.load(f))

    # 加载 feat 并将其转换为 Tensor
    with open(feat_file_path, 'r') as f:
        feat_data = json.load(f)
        # 假设 feat 数据存储为嵌套的列表结构（可以是数字或嵌套列表）
        # 使用 torch.tensor 将其转换回原始 Tensor 格式
        feat = torch.tensor(feat_data)

    return combined_mask, feat


def load_mask(filename, mask_dir):
    # 直接使用文件名，而不移除扩展名
    mask_file_name = filename + '.json'
    # 构造完整的文件路径
    mask_file_path = os.path.join(mask_dir, mask_file_name)

    # 加载 combined_mask
    with open(mask_file_path, 'r') as f:
        combined_mask = np.array(json.load(f))
    return combined_mask




# def load_feat_list(filename, feat_dir):
#     # 构造 JSON 文件路径
#     feat_file_name = filename + '.json'
#     feat_file_path = os.path.join(feat_dir, feat_file_name)
#
#     # 加载 JSON 文件
#     with open(feat_file_path, 'r') as f:
#         feat_list = json.load(f)
#
#     # 将 features 字段从列表转换为 Tensor
#     for item in feat_list:
#         if 'features' in item:
#             item['features'] = torch.tensor(item['features'], dtype=torch.float32)
#
#     return feat_list

def load_feat_list(filename, feat_dir):
    # 构造 JSON 文件路径
    feat_file_name = filename + '.json'
    feat_file_path = os.path.join(feat_dir, feat_file_name)

    # 加载 JSON 文件
    with open(feat_file_path, 'r') as f:
        feat_list = json.load(f)
        feat = torch.tensor(feat_list)
    return feat


def remap_mask_values(label):
    # 获取唯一值并排序
    unique_values = np.unique(label)

    # 创建一个映射字典，将每个唯一值映射为从 0 开始递增的值
    value_mapping = {v: i for i, v in enumerate(unique_values)}

    # 使用映射表将原始值替换为连续值
    remapped_label = np.vectorize(value_mapping.get)(label)

    return remapped_label

# Cityscapes TrainIds 的颜色映射表
CITYSCAPES_COLORS = np.array([
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (70, 70, 70),  # Building
    (102, 102, 156),  # Wall
    (190, 153, 153),  # Fence
    (153, 153, 153),  # Pole
    (250, 170, 30),  # Traffic Light
    (220, 220, 0),  # Traffic Sign
    (107, 142, 35),  # Vegetation
    (152, 251, 152),  # Terrain
    (70, 130, 180),  # Sky
    (220, 20, 60),  # Person
    (255, 0, 0),  # Rider
    (0, 0, 142),  # Car
    (0, 0, 70),  # Truck
    (0, 60, 100),  # Bus
    (0, 80, 100),  # Train
    (0, 0, 230),  # Motorcycle
    (119, 11, 32),  # Bicycle
    (0, 0, 0)  # Void (usually mapped to black)
], dtype=np.uint8)


def visualize_cityscapes_label(seg_gt, colors):
    h, w = seg_gt.shape
    # 创建一个空的彩色图像
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    # 遍历颜色映射并填充彩色图像
    for label_id, color in enumerate(colors):
        color_image[seg_gt == label_id] = color
    # 显示图像
    plt.figure(figsize=(20, 20))
    plt.imshow(color_image)
    plt.axis('off')
    plt.title("Cityscapes Label Visualization")
    plt.show()