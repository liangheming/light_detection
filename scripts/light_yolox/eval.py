import os
import yaml
import sys
import argparse
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.append("../../")

from scripts.light_yolox.arch import LightYOLOX
from datasets.transform.augmentations import *
from datasets.coco import COCODataSet, coco_ids


def main(cfg_path):
    with open(cfg_path, "r") as rf:
        cfg = yaml.safe_load(rf)
    cfg['model']['head']['stacks'] = 1
    cfg['model']['head']['conf_thresh'] = 0.01
    cfg['model']['head']['nms_thresh'] = 0.6
    cfg['data']['size'] = 416
    print(cfg)
    net = LightYOLOX(**cfg['model'])
    # weight = torch.load(os.path.join(cfg['save_dir'], "ema_last.pth"), map_location="cpu")
    weight = torch.load(os.path.join("workspace/shuffle_pan_yolox_n", "ema_last.pth"), map_location="cpu")
    net.load_state_dict(weight)
    # coco_gt = COCO(cfg['data']['val_json_path'])
    coco_gt = COCO("/home/lion/data/coco/annotations/instances_val2017.json")
    basic_transform = Sequence(
        transforms=[
            RandScaleToMax(max_threshes=[cfg['data']['size'], ], center_padding=False),
            PixelNormalize(),
            ChannelSwitch(channel_order=(2, 1, 0))  # bgr -> rgb
        ]
    )
    # val_dataset = COCODataSet(
    #     img_dir=cfg['data']['val_img_dir'],
    #     json_path=cfg['data']['val_json_path'],
    #     transform=basic_transform,
    #     empty_allowed=True
    # )
    val_dataset = COCODataSet(
        img_dir="/home/lion/data/coco/val2017",
        json_path="/home/lion/data/coco/annotations/instances_val2017.json",
        transform=basic_transform,
        empty_allowed=True
    )

    net.eval()
    predicts_list = list()
    for info in tqdm(val_dataset):
        img_inp = torch.from_numpy(info.img).unsqueeze(0).permute(0, 3, 1, 2).float()
        img_id, r = info.ext_prop['id'], info.ext_prop['r']
        prediction = net.predict(img_inp)[0]
        if prediction is None:
            continue
        prediction = prediction.cpu().numpy()
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
                        default="conifgs/shuffle_pan_yolox_s.yaml",
                        help="train config file path")
    parser.add_argument("--seed", type=int, default=1024, help="random seed")
    args = parser.parse_args()
    main(cfg_path=args.config)
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.130
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.249
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.121
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.027
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.109
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.237
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.152
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.230
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.242
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.046
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.230
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.427
