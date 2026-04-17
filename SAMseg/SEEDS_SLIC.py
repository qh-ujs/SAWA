import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io, segmentation

img_path = "frankfurt_000001_016029_leftImg8bit.png"
img = io.imread(img_path)
H, W = img.shape[:2]
n_segments = 600

slic_segments = segmentation.slic(
    img,
    n_segments=n_segments,
    compactness=0.1,   # 这个最重要！
    start_label=0
)

# ---------- 1) SLIC 可视化 ----------
slic_bound = segmentation.find_boundaries(slic_segments, mode='outer')
thickness = 3
if thickness > 1:
    kernel = np.ones((thickness, thickness), dtype=np.uint8)
    slic_bound = cv2.dilate(slic_bound.astype(np.uint8), kernel, iterations=1).astype(bool)

overlay_slic = img.copy()
overlay_slic[slic_bound] = 255
# 显示
plt.figure(figsize=(12, 6))
plt.title(f"SLIC")
plt.imshow(overlay_slic)
plt.axis('off')
plt.tight_layout()
plt.show()


# ---------- 2) SEEDS 可视化 ----------
# 准备 RGB->LAB（SEEDS 通常用 LAB）
rgb = img.copy()
lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

# SEEDS 参数（照你给的写法）
seeds = cv2.ximgproc.createSuperpixelSEEDS(
    W, H, 3, num_superpixels=n_segments, num_levels=1, prior=2,
    histogram_bins=5, double_step=False
)
# iterate 生成超像素
seeds.iterate(lab, num_iterations=15)
labels = seeds.getLabels()  # HxW, 0 开始
# 用 skimage 的 find_boundaries 直接找边界（更简洁）
seeds_bound = segmentation.find_boundaries(labels, mode='thick')
# 放粗
if thickness > 1:
    kernel = np.ones((thickness, thickness), dtype=np.uint8)
    seeds_bound = cv2.dilate(seeds_bound.astype(np.uint8), kernel, iterations=1).astype(bool)

# 在原图上标白
overlay_seeds = rgb.copy()
overlay_seeds[seeds_bound] = 255

# 显示 SEEDS 结果
plt.figure(figsize=(12, 6))
plt.title("SEEDS superpixels (boundaries)")
plt.imshow(overlay_seeds)
plt.axis('off')
plt.tight_layout()
plt.show()
