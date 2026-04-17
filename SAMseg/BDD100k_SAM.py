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
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(dist.get_rank())


def cleanup_distributed():
    dist.destroy_process_group()


class BDDDataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.images = []

        img_dir = os.path.join(root, "seg", "images", mode)
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
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
    unique_labels = np.unique(combined_mask)
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
    base_name = os.path.basename(img_path).split('.')[0]

    mask_path = os.path.join(mask_dir, f"{base_name}.json")
    with open(mask_path, 'w') as f:
        json.dump(combined_mask.tolist(), f)

    vis_img = visualize_mask(combined_mask)
    vis_path = os.path.join(vis_dir, f"{base_name}.png")
    cv2.imwrite(vis_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))


def main():
    setup_distributed()

    # 严格按用户提供的路径设置
    data_root = "/data/BDD100k/bdd100k_seg/bdd100k/"
    save_root = "/data/bdd100k_SAM/"

    mask_save_dir = os.path.join(save_root, "BDD_sam_val/BDD_SAM_mask/")
    vis_save_dir = os.path.join(save_root, "BDD_sam_val/BDD_SAM/")
    os.makedirs(mask_save_dir, exist_ok=True)
    os.makedirs(vis_save_dir, exist_ok=True)

    sam_checkpoint = "/checkpoints/sam_vit_h.pth"
    model_type = "vit_h"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    dataset = BDDDataset(data_root, mode='val')
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            sampler=sampler,
                            num_workers=4,
                            pin_memory=True)

    for batch in tqdm(dataloader, desc="Processing Images"):
        images, img_paths = batch
        for img, path in zip(images, img_paths):
            img_np = img.numpy().astype(np.uint8)
            masks = mask_generator.generate(img_np)
            combined_mask = sam_superpixels(img_np, masks)

            save_outputs(combined_mask, path, mask_save_dir, vis_save_dir)

            del masks, combined_mask
            torch.cuda.empty_cache()

    cleanup_distributed()


if __name__ == "__main__":
    main()
# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 BDD100k_SAM.py
