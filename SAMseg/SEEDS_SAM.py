import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# --------------------------
# 1) 读取图像 & 初始化 SAM
# --------------------------
img_path = "frankfurt_000001_016029_leftImg8bit.png"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
H, W = image.shape[:2]

sam_checkpoint = "sam_vit_h.pth"
model_type = "vit_h"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# --------------------------
# 2) SAM 分割
# --------------------------
masks = mask_generator.generate(image)
print(f"SAM masks: {len(masks)}")

# --------------------------
# 3) SEEDS 超像素
# --------------------------
n_segments = 800
thickness = 2

lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
seeds = cv2.ximgproc.createSuperpixelSEEDS(W, H, 3, num_superpixels=n_segments,
                                           num_levels=1, prior=2,
                                           histogram_bins=5, double_step=False)
seeds.iterate(lab, num_iterations=15)
seeds_labels = seeds.getLabels()

# --------------------------
# 4) 构建 combined mask
# --------------------------
combined_mask = np.zeros((H, W), dtype=np.int32)
# 先放 SAM 各个块
for i, m in enumerate(masks):
    combined_mask[m['segmentation'] > 0] = i + 1  # SAM 块标签从 1 开始

# SAM 没覆盖的区域用 SEEDS 填充
next_label = len(masks) + 1
mask_not_covered = combined_mask == 0
combined_mask[mask_not_covered] = seeds_labels[mask_not_covered] + next_label

# --------------------------
# 5) 忽略小块（面积太小直接置黑）
# --------------------------
min_area = 200  # 面积阈值，可以调整
unique_labels = np.unique(combined_mask)
for lbl in unique_labels:
    if lbl == 0:
        continue
    mask_area = np.sum(combined_mask == lbl)
    if mask_area < min_area:
        combined_mask[combined_mask == lbl] = 0  # 置黑

# --------------------------
# 6) 转成可视化的 mask 列表
# --------------------------
fake_masks = []
unique_labels = np.unique(combined_mask)
for lbl in unique_labels:
    if lbl == 0:
        continue
    mask_bin = (combined_mask == lbl).astype(np.uint8)
    fake_masks.append({'segmentation': mask_bin})

# --------------------------
# 7) 可视化
# --------------------------
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_anns(masks):
    for mask in masks:
        show_mask(mask['segmentation'], plt.gca(), True)

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
show_anns(fake_masks)
plt.show()
