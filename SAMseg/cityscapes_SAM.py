import os
import json
import numpy as np
import torch
import cv2
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm
from SAM_superpixels import sam_superpixels


def setup_distributed():
    """Set up distributed training."""
    dist.init_process_group(backend='nccl')  # Use 'nccl' for GPUs
    torch.cuda.set_device(dist.get_rank())  # Set GPU device for the process


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


class CityscapesDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        self.root = root
        self.transform = transform
        self.images = []
        img_dir = os.path.join(root, 'leftImg8bit', mode)
        for city in os.listdir(img_dir):
            city_dir = os.path.join(img_dir, city)
            for file_name in os.listdir(city_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    self.images.append(os.path.join(city_dir, file_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, img_path


def show_combined_mask(combined_mask, random_color=False):
    """Save the combined mask with color visualization."""
    unique_labels = np.unique(combined_mask)

    # Create a color map
    if random_color:
        colors = np.random.rand(len(unique_labels), 4)  # RGBA
        colors[:, 3] = 0.6  # Set alpha to 0.6
    else:
        colors = np.zeros((len(unique_labels), 4))
        colors[:, :3] = [30 / 255, 144 / 255, 255 / 255]  # RGB
        colors[:, 3] = 0.6  # Alpha

    h, w = combined_mask.shape
    mask_image = np.zeros((h, w, 4))  # RGBA image
    for i, label in enumerate(unique_labels):
        if label > 0:
            mask_image[combined_mask == label] = colors[i]

    return mask_image


def save_results(combined_mask, img_path, mask_save_dir, label_save_dir):
    """Save combined_mask and feat to JSON files and the mask image."""
    combined_mask_file_name = os.path.basename(img_path).replace('_leftImg8bit.png', '_leftImg8bit.json')
    combined_mask_path = os.path.join(mask_save_dir, combined_mask_file_name)
    with open(combined_mask_path, 'w') as f:
        json.dump(combined_mask.tolist(), f)

    label_image = show_combined_mask(combined_mask, random_color=True)
    label_file_name = os.path.basename(img_path).replace('_leftImg8bit.png', '_leftImg8bit.png')
    label_image_path = os.path.join(label_save_dir, label_file_name)
    cv2.imwrite(label_image_path, (label_image[:, :, :3] * 255).astype(np.uint8))


def main():
    setup_distributed()

    # 定义路径
    data_dir = "/data/cityscapes/"
    mask_save_dir = os.path.join(data_dir, "SAM1_best/city_sam_test/cityscapes_SAM_mask/")
    label_save_dir = os.path.join(data_dir, "SAM1_best/city_sam_test/cityscapes_SAM/")

    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    # 加载SAM模型
    sam_checkpoint = "/checkpoints/sam_vit_h.pth"
    model_type = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)  # 注册模型
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Initialize dataset and dataloader for validation set
    dataset = CityscapesDataset(data_dir, mode='test')  # Change to 'val' mode
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=4)

    # Process images in validation set
    for images, img_paths in tqdm(dataloader):
        for image, img_path in zip(images, img_paths):
            image = image.numpy()  # Convert to numpy array

            masks= mask_generator.generate(image)
            combined_mask = sam_superpixels(image, masks)

            # Save results
            save_results(combined_mask, img_path, mask_save_dir, label_save_dir)

    cleanup_distributed()


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 cityscapes_SAM.py
