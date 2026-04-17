import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

from SAMseg.SAM_superpixels import show_combined_mask, CITYSCAPES_COLORS


def load_cityscapes_label(label_path):
    label = np.array(Image.open(label_path))  # 使用PIL加载
    return label


def process_masks(masks, label):
    """主处理函数"""
    result_mask = np.full_like(label, 255, dtype=np.uint8)
    total_small = 0

    for block_id in np.unique(masks):
        block_region = (masks == block_id)

        # 提取当前块内的标签
        labels_in_block = label[block_region]

        # 统计标签分布
        unique_labels, counts = np.unique(labels_in_block, return_counts=True)
        if len(unique_labels) == 0:
            continue

        # 确定主标签
        main_label = unique_labels[np.argmax(counts)]

        # 标记主区域
        main_area = (label == main_label) & block_region
        result_mask[main_area] = main_label

        # 统计次要像素
        total_small += np.sum(block_region) - np.sum(main_area)

    return result_mask, total_small


def visualize_comparison(seg_gt, result_mask, colors):
    """
    垂直并排显示原始标签和处理结果
    参数：
        seg_gt: 原始标签图 (H, W)
        result_mask: 处理后的掩码图 (H, W)
        colors: 包含255类颜色的完整映射表 (20个颜色)
    """
    h, w = seg_gt.shape

    # 创建画布和子图
    plt.figure(figsize=(10, 15))

    # 绘制原始标签 -------------------------------------------------
    plt.subplot(2, 1, 1)  # 2行1列，第1个位置
    gt_color = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in enumerate(colors):
        gt_color[seg_gt == label_id] = color
    plt.imshow(gt_color)
    plt.axis('off')
    plt.title("Ground Truth Label", fontsize=20)

    # 绘制处理结果 -------------------------------------------------
    plt.subplot(2, 1, 2)  # 2行1列，第2个位置
    res_color = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id, color in enumerate(colors):
        res_color[result_mask == label_id] = color
    plt.imshow(res_color)
    plt.axis('off')
    plt.title("Processed Result Mask", fontsize=20)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 输入路径
    gt_path = "/data2/qinghua/ZDJ/datasets_111/cityscapes/gtFine/val/frankfurt/frankfurt_000001_080391_gtFine_labelTrainIds.png"
    sam_json_path = "/data2/qinghua/ZDJ/datasets/Cityscapes_SAM/city_sam_val/cityscapes_SAM_mask/frankfurt_000001_080391_leftImg8bit.json"

    # 加载数据
    gt = load_cityscapes_label(gt_path)
    with open(sam_json_path, 'r') as f:
        masks = np.array(json.load(f))  # 读取掩码
    h, w = masks.shape
    # 处理掩码
    result_mask, small_pixels = process_masks(masks, gt)
    ErrorRate = small_pixels / (h * w)
    # 输出结果
    print(f"次要像素总数: {small_pixels}")
    print(f"错误率: {ErrorRate}")
    print(f"结果掩码形状: {result_mask.shape}")
    print(f"唯一标签值: {np.unique(result_mask)}")
    # 可视化对比
    show_combined_mask(masks, random_color=True)
    visualize_comparison(
        seg_gt=gt,
        result_mask=result_mask,
        colors=CITYSCAPES_COLORS
    )
