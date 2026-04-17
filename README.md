# Structure-aware Prior Guidance for Domain Generalized Semantic Segmentation
[Qinghua Ren](https://)<sup>1</sup>, [Dongjun Zhang](https://)<sup>2</sup>, et al. <br />
Jiangsu University

Paper: https://arxiv.org/pdf/SAWA.pdf

SAWA is a lightweight framework for **Domain Generalized Semantic Segmentation (DGSS)** that injects structure-aware semantic priors into **Vision Foundation Models (VFMs)**. SAWA achieves state-of-the-art performance on multiple DGSS benchmarks, reaching **68.70% mIoU for GTAV → Cityscapes+Mapillary+BDD100K** and **71.90% mIoU for Cityscapes → Mapillary+BDD100K** generalization. Using only the Cityscapes training data, SAWA further achieves **83.53% mIoU on the Cityscapes validation set**.

![SAWA Framework](docs/frame.png)

## Try and Test
**Experience the demo:** Users can open [demo.ipynb](demo.ipynb) in any Jupyter-supported editor to explore our demonstration.
![Demo Preview](docs/demo.png)

For testing on the cityscapes dataset, refer to the 'Install' and 'Setup' sections below.
  
## Environment Setup
To set up your environment, execute the following commands:
```bash
conda create -n SAWA -y
conda activate SAWA
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install "mmsegmentation>=1.0.0"
pip install "mmdet>=3.0.0"
pip install xformers=='0.0.20' # optional for DINOv2
pip install -r requirements.txt
pip install future tensorboard
```

## Dataset Preparation
The Preparation is similar as [DDB](https://github.com/xiaoachen98/DDB).

**Cityscapes:** Download `leftImg8bit_trainvaltest.zip` and `gt_trainvaltest.zip` from [Cityscapes Dataset](https://www.cityscapes-dataset.com/downloads/) and extract them to `data/cityscapes`.

**Mapillary:** Download MAPILLARY v1.2 from [Mapillary Research](https://research.mapillary.com/) and extract it to `data/mapillary`.

**GTA:** Download all image and label packages from [TU Darmstadt](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract them to `data/gta`.

**ACDC**: Download all image and label packages from [ACDC](https://acdc.vision.ee.ethz.ch/) and extract them to `data/acdc`.

**BDD100K:** Download the BDD100K dataset from [BDD100K](https://bdd-data.berkeley.edu/) and extract it to `data/bdd100k`.

**SYNTHIA:** Download the SYNTHIA-RAND-CITYSCAPES dataset from [SYNTHIA Dataset](https://synthia-dataset.net/) and extract it to `data/synthia`.

**Prepare datasets with these commands:**
```shell
cd SAWA
mkdir data
# Convert data for validation if preparing for the first time
python tools/convert_datasets/gta.py data/gta # Source domain
python tools/convert_datasets/cityscapes.py data/cityscapes
# Convert Mapillary to Cityscapes format and resize for validation
python tools/convert_datasets/mapillary2cityscape.py data/mapillary data/mapillary/cityscapes_trainIdLabel --train_id
python tools/convert_datasets/mapillary_resize.py data/mapillary/validation/images data/mapillary/cityscapes_trainIdLabel/val/label data/mapillary/half/val_img data/mapillary/half/val_label
```

**Generate SAM Masks with these commands:**
```shell
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h.pth
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 SAMseg/gta_SAM.py --data_dir data/gta --mask_save_dir data/gta_sam/GTA_SAM_mask --label_save_dir data/gta_sam/GTA_SAM --sam_checkpoint checkpoints/sam_vit_h.pth
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 SAMseg/Mapillary_SAM.py --data_dir data/mapillary --mask_save_dir data/mapillary_sam/mapillary_SAM_mask --label_save_dir data/mapillary_sam/mapillary_SAM --sam_checkpoint checkpoints/sam_vit_h.pth
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 SAMseg/BDD100K_SAM.py --data_dir data/BDD100K --mask_save_dir data/BDD100K_sam/BDD100K_SAM_mask --label_save_dir data/BDD100K_sam/BDD100K_SAM --sam_checkpoint checkpoints/sam_vit_h.pth
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch --nproc_per_node=1 SAMseg/cityscapes_SAM.py --data_dir data/cityscapes --mask_save_dir data/cityscapes_sam/cityscapes_SAM_mask --label_save_dir data/cityscapes_sam/cityscapes_SAM --sam_checkpoint checkpoints/sam_vit_h.pth
Please configure the SAM mask path in: mmseg/datasets/transforms/transforms.py
```

**The final folder structure should look like this:**

```
SAWA
├── ...
├── checkpoints
│   ├── dinov2_vitl14_pretrain.pth
│   ├── dinov2_converted.pth
│   ├── eva02_L_pt_m38m_p14to16.pt
│   ├── eva02_L_converted.pth
│   ├── sam_vit_h.pth
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── cityscapes_sam
│   │   ├── cityscapes_SAM_mask
│   │   ├── cityscapes_SAM
│   ├── bdd100k
│   │   ├── images
│   │   |   ├── 10k
│   │   │   |    ├── train
│   │   │   |    ├── val
│   │   ├── labels
│   │   |   ├── sem_seg
│   │   |   |    ├── masks
│   │   │   |    |    ├── train
│   │   │   |    |    ├── val
│   ├── BDD100K_sam
│   │   ├── BDD100K_SAM_mask
│   │   ├── BDD100K_SAM
│   ├── mapillary
│   │   ├── training
│   │   ├── cityscapes_trainIdLabel
│   │   ├── half
│   │   │   ├── val_img
│   │   │   ├── val_label
│   ├── mapillary_sam
│   │   ├── mapillary_SAM_mask
│   │   ├── mapillary_SAM
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── gta_sam
│   │   ├── GTA_SAM_mask
│   │   ├── GTA_SAM
├── ...
```
## Pretraining Weights
* **Download:** Download pre-trained weights from [facebookresearch](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) for testing. Place them in the project directory without changing the file name.
* **Convert:** Convert pre-trained weights for training or evaluation.
  ```bash
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/dinov2_converted.pth
  python tools/convert_models/convert_eva2_512x512.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/eva02_L_converted.pth
  ```

## Evaluation
  Run the evaluation:
  ```
  python tools/test.py configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py checkpoints/SAWA_g2cbm_pretrained.pth --backbone dinov2_converted.pth
  ```

## Training
Start training in single GPU:
```
python tools/train.py configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py
```
Start training in multiple GPU:
```
PORT=12345 CUDA_VISIBLE_DEVICES=1,2,3,4 bash tools/dist_train.sh configs/dinov2/rein_dinov2_mask2former_1024x1024_bs4x2.py NUM_GPUS
```

## Generate full weights
Because we only fine-tune and save the SAWA and head weights, if you need a complete set of segmentor weights, you need to use this script:
```
python generate_full_weights.py --segmentor_save_path SEGMENTOR_SAVE_PATH --backbone CONVERTED_BACKBONE --SAWA_head SAWA_HEAD
```

### checkpoints

you can download checkpoints from https://drive.google.com/drive/folders/1CLGTYCXS_aIRg9h3VP8kM9sgXGYXV8aX

## Acknowledgment
Our implementation is mainly based on following repositories. Thanks for their authors.
* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [DDB](https://github.com/xiaoachen98/DDB)
* [DTP](https://github.com/w1oves/DTP)
* [Rein](https://github.com/w1oves/Rein)

