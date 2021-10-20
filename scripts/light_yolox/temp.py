from datasets.coco import COCODataSet, nano_transform
from model.backbone.shufflenet import build_backbone
from torch.utils.data.dataloader import DataLoader
from scripts.light_yolox.arch import LightYOLOX
from model.loss.yolox_loss import YOLOXLoss


def main():
    data_set = COCODataSet(img_dir="/home/lion/data/coco/val2017",
                           json_path="/home/lion/data/coco/annotations/instances_val2017.json",
                           transform=nano_transform,
                           visualize=False
                           )
    dataloader = DataLoader(dataset=data_set, batch_size=32, shuffle=True, num_workers=2,
                            collate_fn=data_set.collect_fn, drop_last=True)

    net = LightYOLOX()
    loss = YOLOXLoss(model=net)
    i = 0
    for img, target, meta_info in dataloader:
        target = {"label": target, "batch_len": meta_info['batch_len']}
        out = net(img)
        loss_out = loss(out, target)
        print(loss_out)
        pre = net.predict(img)
        print(pre)
        i += 1
        if i > 5:
            break


if __name__ == '__main__':
    main()
