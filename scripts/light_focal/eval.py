import os
import yaml
import sys
import argparse
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append("../../")

from scripts.light_focal.arch import LightGFOCAL
from datasets.transform.augmentations import *
from datasets.coco import COCODataSet, coco_ids
from torch.utils.data.dataloader import DataLoader


@torch.no_grad()
def main(cfg_path):
    with open(cfg_path, "r") as rf:
        cfg = yaml.safe_load(rf)
    cuda = True
    cfg['data']['val_img_dir'] = "/home/lion/data/coco/val2017"
    cfg['data']['val_json_path'] = "/home/lion/data/coco/annotations/instances_val2017.json"
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    net = LightGFOCAL(**cfg['model'])
    weight = torch.load(os.path.join(cfg['save_dir'], "ema_last.pth"), map_location="cpu")
    net.load_state_dict(weight)
    basic_transform = Sequence(
        transforms=[
            RandScaleToMax(max_threshes=[cfg['data']['v_size'], ], center_padding=False),
            PixelNormalize(),
            ChannelSwitch(channel_order=(2, 1, 0))  # bgr -> rgb
        ]
    )
    val_dataset = COCODataSet(
        img_dir=cfg['data']['val_img_dir'],
        json_path=cfg['data']['val_json_path'],
        transform=basic_transform,
        empty_allowed=True
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
    coco_gt = COCO(cfg['data']['val_json_path'])
    net.eval()
    net.to(device)
    predicts_list = list()
    # for info in tqdm(val_dataset):
    #     img_inp = torch.from_numpy(info.img).unsqueeze(0).permute(0, 3, 1, 2).float()
    #     img_id, r = info.ext_prop['id'], info.ext_prop['r']
    #     prediction = net.predict(img_inp)[0]
    #     if prediction is None:
    #         continue
    #     prediction = prediction.cpu().numpy()
    #     prediction[:, :4] = prediction[:, :4] / r
    #     for box in prediction:
    #         x1, y1, x2, y2, conf, cls_id = box
    #         w, h, coco_cls_id = x2 - x1, y2 - y1, coco_ids[int(cls_id)]
    #         pred_item = {
    #             "image_id": img_id,
    #             "score": conf,
    #             "category_id": coco_cls_id,
    #             "bbox": [x1, y1, w, h],
    #         }
    #         predicts_list.append(pred_item)
    for img, label, meta_info in tqdm(val_dataloader):
        img = img.to(device)
        ext_info = meta_info['ext_props']
        out = net(img)
        for i in range(len(out)):
            prediction = out[i]
            if prediction is None:
                continue
            prediction = prediction.cpu().numpy()
            r = ext_info[i]['r']
            img_id = ext_info[i]['id']
            prediction[:, :4] = prediction[:, :4] / r
            for box in prediction:
                x1, y1, x2, y2, conf, cls_id = box
                w, h, coco_cls_id = x2 - x1, y2 - y1, coco_ids[int(cls_id)]
                pred_item = {
                    "image_id": img_id,
                    "score": conf,
                    "category_id": coco_cls_id,
                    "bbox": [x1, y1, w, h],
                }
                predicts_list.append(pred_item)

    coco_dt = coco_gt.loadRes(predicts_list)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="configs/shuffle_pan_gfocal_s.yaml",
                        help="train config file path")
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    args = parser.parse_args()
    main(cfg_path=args.config)
