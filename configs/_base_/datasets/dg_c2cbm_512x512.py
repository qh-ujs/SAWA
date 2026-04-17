_base_ = [
    "./cityscapes_512x512.py",
    "./bdd100k_512x512.py",
    "./mapillary_512x512.py",
    "./gta_512x512.py",
    "./synthia_512x512.py",
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_cityscapes}},
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            {{_base_.val_cityscapes}},
            {{_base_.val_bdd}},
            {{_base_.val_mapillary}},
            {{_base_.val_gta}},
            {{_base_.val_Syn}},
        ],
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    # type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["citys", "map", "bdd"]
    # type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["citys"]
    # type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["GTA", "Syn"]
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["citys", "map", "bdd", "GTA", "Syn"]
)
test_evaluator=val_evaluator
