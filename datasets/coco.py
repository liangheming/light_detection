import os
from typing import List

import torch
from datasets.base import BaseDetectionDataset
from datasets.transform.augmentations import *
from pycocotools.coco import COCO

coco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
colors = [(67, 68, 113), (130, 45, 169), (2, 202, 130), (127, 111, 90), (92, 136, 113),
          (33, 250, 7), (238, 92, 104), (0, 151, 197), (134, 9, 145), (253, 181, 88),
          (246, 11, 137), (55, 72, 220), (136, 8, 253), (56, 73, 180), (85, 241, 53),
          (153, 207, 15), (187, 183, 180), (149, 32, 71), (92, 113, 184), (131, 7, 201),
          (56, 20, 219), (243, 201, 77), (13, 74, 96), (79, 14, 44), (195, 150, 66),
          (2, 249, 42), (195, 135, 43), (105, 70, 66), (120, 107, 116), (122, 241, 22),
          (17, 19, 179), (162, 185, 124), (31, 65, 117), (88, 200, 80), (232, 49, 154),
          (72, 1, 46), (59, 144, 187), (200, 193, 118), (123, 165, 219), (194, 84, 34),
          (91, 184, 108), (252, 64, 153), (251, 121, 27), (105, 93, 210), (89, 85, 81),
          (58, 12, 154), (81, 3, 50), (200, 40, 236), (155, 147, 180), (73, 29, 176),
          (193, 19, 175), (157, 225, 121), (128, 195, 235), (146, 251, 108), (13, 146, 186),
          (231, 118, 145), (253, 15, 105), (187, 149, 62), (121, 247, 158), (34, 8, 142),
          (83, 61, 48), (119, 218, 69), (197, 94, 130), (222, 176, 142), (21, 20, 77),
          (6, 42, 17), (136, 33, 156), (39, 252, 211), (52, 50, 40), (183, 115, 34),
          (107, 80, 164), (195, 215, 74), (7, 154, 135), (136, 35, 24), (131, 241, 125),
          (208, 99, 208), (5, 4, 129), (137, 156, 175), (29, 141, 67), (44, 20, 99)]

hsv_color = OneOf(
    transforms=[
        Identity(),
        RandHSV()
    ]
)
augment_transform = Sequence(
    transforms=[
        hsv_color,
        RandCrop(min_thresh=0.6, max_thresh=1.0).reset(p=0.2),
        RandScaleToMax(max_threshes=[640]),
        RandPerspective(degree=(-5, 5), scale=(0.6, 1.4), translate=0.2)
    ]
)
nano_transform = Sequence(
    transforms=[
        RandBCS(
            brightness=0.2,
            contrast=(0.6, 1.4),
            saturation=(0.5, 1.2)
        ),
        NanoPerspective(
            keep_ratio=True,
            dst_shape=(416, 416),
            translate=0.2,
            flip=0.5,
            scale=(0.5, 1.4)
        ),
        RandScaleToMax(max_threshes=[416, ], center_padding=False),
        PixelNormalize(),
        ChannelSwitch(channel_order=(2, 1, 0)),  # bgr -> rgb
        CoordTransform(c_type="xywh")
    ]
)
basic_transform = Sequence(
    transforms=[
        RandScaleToMax(max_threshes=[416], center_padding=True),
        PixelNormalize(),
        ChannelSwitch(channel_order=(2, 1, 0))  # bgr -> rgb
    ]
)


class COCODataSet(BaseDetectionDataset):
    def __init__(self,
                 img_dir,
                 json_path,
                 transform=None,
                 use_crowed=False,
                 empty_allowed=False,
                 visualize=False):
        self.img_dir = img_dir
        self.json_path = json_path
        if transform is None:
            transform = basic_transform
        self.transform = transform
        self.use_crowed = use_crowed
        self.visualize = visualize
        self.empty_allowed = empty_allowed
        self.coco = COCO(annotation_file=json_path)
        self.__get_datalist__()

    def __get_datalist__(self):
        box_info_list = list()
        for img_id in self.coco.imgs.keys():
            instance = self.coco.imgs[img_id]
            filename = instance['file_name']
            width, height = instance['width'], instance['height']
            img_path = os.path.join(self.img_dir, filename)
            assert os.path.exists(img_path), "{:s} is not exists".format(img_path)
            assert width > 1 and height > 1, "invalid width or heights"
            annotations = self.coco.imgToAnns[img_id]
            label_list = list()
            for ann in annotations:
                category_id, box, iscrowd = ann['category_id'], ann['bbox'], ann['iscrowd']
                if not self.use_crowed and iscrowd == 1:
                    continue
                cls_id = coco_ids.index(category_id)
                assert 0 <= cls_id < len(coco_ids), \
                    "invalid cls id: coco id is {:d}, cls_id is {:d}".format(
                        category_id, cls_id)
                x1, y1 = box[:2]
                x2, y2 = x1 + box[2], y1 + box[3]
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
                    print("warning box ", box)
                label_list.append((cls_id, x1, y1, x2, y2))
            valid_box_len = len(label_list)
            if valid_box_len == 0 and not self.empty_allowed:
                continue
            label_info = np.array(label_list)
            box_info = BoxInfoCV(img_path=img_path, boxes=label_info[:, 1:], labels=label_info[:, 0])
            box_info.id = img_id
            box_info.ext_prop.update({"id": img_id})
            box_info.ext_prop.update({"shape": (width, height)})
            box_info_list.append(box_info)
        self.data_list = box_info_list

    def __getitem__(self, index):
        assert self.transform is not None, "instance property transform is None, please set transform first"
        box_info = self.data_list[index].clone().load_img()
        box_info = self.transform(box_info)
        if self.visualize:
            import uuid
            draw_img = box_info.draw_img(colors=colors, names=coco_names)
            img_name = str(uuid.uuid4()).replace("-", "")[:16]
            cv.imwrite("{:s}.jpg".format(img_name), draw_img)
            # cv.imshow(__name__, draw_img)
            # cv.waitKey(0)
        return box_info

    def __len__(self):
        return len(self.data_list)

    def collect_fn(self, batch: List[BoxInfoCV]):
        batch_img = list()
        batch_target = list()
        batch_length = list()

        ext_props = list()
        for item in batch:
            target = np.concatenate([item.labels[:, None], item.boxes], axis=-1)
            batch_img.append(item.img)
            ext_props.append(item.ext_prop)
            batch_target.append(target)
            batch_length.append(len(target))
        batch_img = torch.from_numpy(np.stack(batch_img, axis=0)).permute(0, 3, 1, 2).contiguous().float()
        batch_target = torch.from_numpy(np.concatenate(batch_target, axis=0)).float()
        return batch_img, batch_target, {"batch_len": batch_length, "ext_props": ext_props}


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    data_set = COCODataSet(img_dir="/home/lion/data/coco/val2017",
                           json_path="/home/lion/data/coco/annotations/instances_val2017.json",
                           transform=nano_transform,
                           visualize=False
                           )
    dataloader = DataLoader(dataset=data_set, batch_size=16, shuffle=True, num_workers=2,
                            collate_fn=data_set.collect_fn, drop_last=True)
    for inp, label, meta_info in dataloader:
        print(inp.shape)
        print(label.shape)
        print(meta_info)
