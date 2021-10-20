import yaml
import pytorch_lightning as pl
from scripts.light_yolox.arch import LightYOLOX
from scripts.light_yolox.task import TrainTask
from datasets.transform.augmentations import *
from datasets.coco import COCODataSet
from torch.utils.data.dataloader import DataLoader


def main(cfg_path):
    with open(cfg_path, "r") as rf:
        cfg = yaml.safe_load(rf)
    net = LightYOLOX(**cfg['model'])

    nano_transform = Sequence(
        transforms=[
            RandBCS(
                brightness=0.2,
                contrast=(0.6, 1.4),
                saturation=(0.5, 1.2)
            ),
            NanoPerspective(
                keep_ratio=True,
                dst_shape=(cfg['data']['size'], cfg['data']['size']),
                translate=0.2,
                flip=0.5,
                scale=(0.5, 1.4)
            ),
            RandScaleToMax(max_threshes=[cfg['data']['size'], ], center_padding=False),
            PixelNormalize(),
            ChannelSwitch(channel_order=(2, 1, 0)),  # bgr -> rgb
            CoordTransform(c_type="xywh")
        ]
    )

    basic_transform = Sequence(
        transforms=[
            RandScaleToMax(max_threshes=[cfg['data']['size'], ], center_padding=False),
            PixelNormalize(),
            ChannelSwitch(channel_order=(2, 1, 0))  # bgr -> rgb
        ]
    )

    train_dataset = COCODataSet(
        img_dir=cfg['data']['train_img_dir'],
        json_path=cfg['data']['train_json_path'],
        transform=nano_transform
    )
    val_dataset = COCODataSet(
        img_dir=cfg['data']['val_img_dir'],
        json_path=cfg['data']['val_json_path'],
        transform=basic_transform
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg['device']['batchsize_per_gpu'],
        shuffle=True,
        num_workers=cfg['device']['workers_per_gpu'],
        collate_fn=train_dataset.collect_fn,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg['device']['batchsize_per_gpu'],
        shuffle=False,
        num_workers=cfg['device']['workers_per_gpu'],
        pin_memory=True,
        collate_fn=val_dataset.collect_fn,
        drop_last=False,
    )
    cfg['iter_per_epoch'] = len(train_dataloader)
    task = TrainTask(cfg, net)
    trainer = pl.Trainer(
        max_epochs=cfg['optim']['epochs'],
        gpus=cfg['device']['gpus'],
        check_val_every_n_epoch=1,
        accelerator="ddp",
        benchmark=True,
        sync_batchnorm=True,
        gradient_clip_val=10
    )
    trainer.fit(task, train_dataloader, val_dataloader)


if __name__ == '__main__':
    cpath = "./conifgs/shuffle_pan_yolox_n.yaml"
    main(cfg_path=cpath)
