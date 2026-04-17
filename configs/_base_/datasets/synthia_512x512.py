Syn_type = "CityscapesDataset"
Syn_root = "/data/Synthia/"
Syn_crop_size = (512, 512)
Syn_train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="LoadSamMaskFromFile"),
    dict(type="Resize", scale=(1280, 720)),
    dict(type="RandomCrop", crop_size=Syn_crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]
Syn_test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadSamMaskFromFile"),
    dict(type="Resize", scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]
train_Syn = dict(
    type=Syn_type,
    data_root=Syn_root,
    data_prefix=dict(
        img_path="train/RGB",
        seg_map_path="train/GT/LABELS",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=Syn_train_pipeline,
)
val_Syn = dict(
    type=Syn_type,
    data_root=Syn_root,
    data_prefix=dict(
        img_path="val/RGB",
        seg_map_path="val/GT/LABELS",
    ),
    img_suffix=".png",
    seg_map_suffix="_labelTrainIds.png",
    pipeline=Syn_test_pipeline,
)