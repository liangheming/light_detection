import yaml
import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBar

sys.path.append("../../")

from scripts.light_focal.arch import LightGFOCAL
from scripts.light_focal.task import TrainTask
from datasets.transform.augmentations import *
from datasets.coco import COCODataSet
from torch.utils.data.dataloader import DataLoader


def main(cfg_path):
    with open(cfg_path, "r") as rf:
        cfg = yaml.safe_load(rf)
    net = LightGFOCAL(**cfg['model'])

    nano_transform = Sequence(
        transforms=[
            RandBCS(
                brightness=0.2,
                contrast=(0.6, 1.4),
                saturation=(0.5, 1.2)
            ),
            NanoPerspective(
                keep_ratio=True,
                dst_shape=(cfg['data']['t_size'], cfg['data']['t_size']),
                translate=0.2,
                flip=0.5,
                scale=(0.5, 1.2)
            ),
            RandScaleToMax(max_threshes=[cfg['data']['t_size'], ], center_padding=False),
            PixelNormalize(),
            ChannelSwitch(channel_order=(2, 1, 0)),  # bgr -> rgb
        ]
    )

    basic_transform = Sequence(
        transforms=[
            RandScaleToMax(max_threshes=[cfg['data']['v_size'], ], center_padding=False),
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
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg['device']['batchsize_per_gpu'],
        shuffle=False,
        num_workers=cfg['device']['workers_per_gpu'],
        collate_fn=val_dataset.collect_fn,
        drop_last=False,
        pin_memory=True
    )
    cfg['iter_per_epoch'] = len(train_dataloader)
    task = TrainTask(cfg, net)
    trainer = pl.Trainer(
        default_root_dir=cfg['save_dir'],
        max_epochs=cfg['optim']['epochs'],
        gpus=cfg['device']['gpus'],
        accelerator="ddp",
        benchmark=True,
        sync_batchnorm=True,
        val_check_interval=1.0,
        callbacks=[ProgressBar(refresh_rate=0)],
        # precision=16,
        # amp_backend="native"
    )
    trainer.fit(task, train_dataloader, val_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="configs/shuffle_pan_gfocal_s.yaml",
                        help="train config file path")
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    args = parser.parse_args()
    pl.seed_everything(args.seed)
    main(cfg_path=args.config)
