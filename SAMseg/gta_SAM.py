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

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12347'
# os.environ['WORLD_SIZE'] = '1'
# os.environ['RANK'] = '0'


def setup_distributed():
    os.environ['MASTER_ADDR'] = 'localhost'  # 或使用具体的 IP 地址
    os.environ['MASTER_PORT'] = '12388'  # 更改为未被使用的端口
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())


def cleanup_distributed():
    dist.destroy_process_group()


def show_combined_mask(combined_mask, random_color=False):
    """Save the combined mask with color visualization."""
    unique_labels = np.unique(combined_mask)
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


class GTADataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        img_dir = os.path.join(root, 'images')  # Updated to point to images directory
        for file_name in os.listdir(img_dir):
            if file_name.endswith('.png'):  # Assuming images are .png
                self.images.append(os.path.join(img_dir, file_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, img_path


def save_results(combined_mask, img_path, mask_save_dir, label_save_dir):
    combined_mask_file_name = os.path.basename(img_path).replace('.png', '.json')
    combined_mask_path = os.path.join(mask_save_dir, combined_mask_file_name)
    with open(combined_mask_path, 'w') as f:
        json.dump(combined_mask.tolist(), f)
    label_image = show_combined_mask(combined_mask, random_color=True)
    label_file_name = os.path.basename(img_path).replace('.png', '.png')
    label_image_path = os.path.join(label_save_dir, label_file_name)
    cv2.imwrite(label_image_path, (label_image[:, :, :3] * 255).astype(np.uint8))


def main():
    setup_distributed()
    data_dir = "/data/GTA5/"
    mask_save_dir = "/data/GTA_SAM/GTA_SAM_mask/"
    label_save_dir = "/data/GTA_SAM/GTA_SAM/"
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)
    sam_checkpoint = "/checkpoints/sam_vit_h.pth"
    model_type = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    dataset = GTADataset(data_dir)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=4)
    for images, img_paths in tqdm(dataloader):
        for image, img_path in zip(images, img_paths):
            image = image.numpy()
            masks = mask_generator.generate(image)
            combined_mask = sam_superpixels(image, masks)
            save_results(combined_mask, img_path, mask_save_dir, label_save_dir)
    cleanup_distributed()


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 gta_SAM.py
