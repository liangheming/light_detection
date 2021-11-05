import yaml
import torch
from scripts.light_focal.arch import LightGFOCAL
from datasets.transform.augmentations import *
from datasets.coco import COCODataSet
from torch.utils.data.dataloader import DataLoader


def main():
    basic_transform = Sequence(
        transforms=[
            RandScaleToMax(max_threshes=[320, ], center_padding=False),
            PixelNormalize(),
            ChannelSwitch(channel_order=(2, 1, 0))  # bgr -> rgb
        ]
    )
    val_dataset = COCODataSet(
        img_dir="/home/lion/data/coco/val2017",
        json_path="/home/lion/data/coco/annotations/instances_val2017.json",
        transform=basic_transform
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=1,
        collate_fn=val_dataset.collect_fn,
        drop_last=False,
        pin_memory=True
    )
    with open("configs/shuffle_pan_gfocal_s.yaml", "r") as rf:
        cfg = yaml.safe_load(rf)
    net = LightGFOCAL(**cfg["model"])
    weight = torch.load("workspace/shuffle_pan_gfocal_s/last.pth", map_location="cpu")
    net.load_state_dict(weight)
    net.eval()
    for img, label, meta_info in val_dataloader:
        target = {"label": label, "batch_len": meta_info['batch_len']}
        out = net(img, target)
        print(out)
        break


if __name__ == '__main__':
    main()
