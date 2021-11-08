import math
import random
from typing import Dict, Optional, Tuple

import cv2 as cv
import numpy as np
from copy import deepcopy

cv.setNumThreads(0)
cv.ocl.setUseOpenCL(False)


class BoxInfoCV(object):
    def __init__(self, img_path, boxes=None, labels=None, padding=(0, 0, 0)):
        """
        :param img_path:
        :param boxes: [n,4] (x_min,y_min,x_max,y_max)
        :param labels: [ids]
        :param padding: bgr padding
        """
        super(BoxInfoCV, self).__init__()
        self.img_path = img_path
        self.img = None
        self.boxes = boxes
        self.labels = labels
        self.__init__padding__(padding)

        self.xyxy = True
        self.normalized_box = False
        self.ext_prop = dict()

    def __init__padding__(self, padding):
        if isinstance(padding, str):
            if padding.strip() == "mean_std":
                self.padding_val = (103, 116, 123)
            elif padding.strip() == "constant":
                self.padding_val = (114, 114, 114)
        elif isinstance(padding, tuple) or isinstance(padding, list):
            self.padding_val = tuple(padding)
        else:
            raise NotImplementedError()

    def revise_label(self):
        if self.boxes is None:
            self.boxes = np.zeros(shape=(0, 4))
        if self.labels is None:
            self.labels = np.zeros(shape=(0,))

    def clone(self):
        return deepcopy(self)

    def load_img(self):
        self.img = cv.imread(self.img_path)
        return self

    def draw_img(self, colors, names):
        assert self.img is not None, "self.img is None"
        ret_img = self.img.copy()
        if self.boxes is None or len(self.boxes) == 0:
            return ret_img
        assert self.xyxy and not self.normalized_box, "box form should be xyxy and not normalized coord"
        for label_idx, (x1, y1, x2, y2) in zip(self.labels, self.boxes):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(ret_img, (x1, y1), (x2, y2), color=colors[int(label_idx)], thickness=2)
            cv.putText(ret_img, "{:s}".format(names[int(label_idx)]),
                       (x1, y1 + 5),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       colors[int(label_idx)], 1)
        return ret_img


class BasicTransform(object):
    def __init__(self, p=0.5):
        self.p = p

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        pass

    def __call__(self, box_info: BoxInfoCV) -> BoxInfoCV:
        assert box_info.img is not None, "please load in img first"
        aug_p = np.random.uniform()
        if aug_p <= self.p:
            box_info = self.aug(box_info)
        return box_info

    def reset(self, **settings):
        p = settings.get('p', None)
        if p is not None:
            self.p = p
        return self


class Identity(BasicTransform):
    def __init__(self, **kwargs):
        kwargs['p'] = 1.0
        super(Identity, self).__init__(**kwargs)

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        return box_info


class RandNoise(BasicTransform):
    def __init__(self, **kwargs):
        kwargs['p'] = 1.0
        super(RandNoise, self).__init__(**kwargs)

    @staticmethod
    def img_aug(img):
        mu = 0
        pre_type = img.dtype
        sigma = np.random.uniform(1, 15)
        ret_img = img + np.random.normal(mu, sigma, img.shape)
        ret_img = ret_img.clip(0., 255.).astype(pre_type)
        return ret_img

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        box_info.img = self.img_aug(box_info.img)
        return box_info


class RandBlur(BasicTransform):
    """
    随机进行模糊
    """

    def __init__(self, **kwargs):
        kwargs['p'] = 1.0
        super(RandBlur, self).__init__(**kwargs)

    @staticmethod
    def gaussian_blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return img

    @staticmethod
    def median_blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.medianBlur(img, kernel_size, 0)
        return img

    @staticmethod
    def blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.blur(img, (kernel_size, kernel_size))
        return img

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        aug_blur = np.random.choice([self.gaussian_blur, self.median_blur, self.blur])
        img = aug_blur(img)
        return img

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        box_info.img = self.img_aug(box_info.img)
        return box_info


class RandHSV(BasicTransform):
    """
    color jitter
    """

    def __init__(self, hgain=0.014, sgain=0.68, vgain=0.36, **kwargs):
        kwargs['p'] = 1.0
        super(RandHSV, self).__init__(**kwargs)
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
        dtype = img.dtype
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val))).astype(dtype)
        ret_img = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
        return ret_img

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        box_info.img = self.img_aug(box_info.img)
        return box_info


class RandBCS(BasicTransform):
    def __init__(self, brightness=0.2, contrast=(0.6, 1.4), saturation=(0.5, 1.2), **kwargs):
        kwargs['p'] = 1.0
        super(RandBCS, self).__init__(**kwargs)

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def img_aug(self, img):
        dtype = img.dtype
        img = img.astype(np.float32) / 255
        if self.brightness is not None and random.randint(0, 1):
            img += random.uniform(-self.brightness, self.brightness)
        if self.contrast is not None and random.randint(0, 1):
            img *= random.uniform(self.contrast[0], self.contrast[1])
        if self.saturation is not None and random.randint(0, 1):
            hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            hsv_img[..., 1] *= random.uniform(self.saturation[0], self.saturation[1])
            img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
        img = np.clip(img * 255, a_min=0.0, a_max=255.0).astype(dtype)
        return img

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        box_info.img = self.img_aug(box_info.img)
        return box_info


class RandScaleToMax(BasicTransform):
    def __init__(self,
                 max_threshes,
                 center_padding=True,
                 pad_to_square=True,
                 minimum_rectangle=False,
                 scale_up=True,
                 division=64,
                 **kwargs):
        kwargs['p'] = 1.0
        super(RandScaleToMax, self).__init__(**kwargs)
        assert isinstance(max_threshes, list)
        self.max_threshes = max_threshes
        self.pad_to_square = pad_to_square
        self.minimum_rectangle = minimum_rectangle
        self.scale_up = scale_up
        self.division = division
        self.center_padding = center_padding

    def make_border(self, img: np.ndarray, max_thresh, border_val):
        h, w = img.shape[:2]
        r = min(max_thresh / h, max_thresh / w)
        if not self.scale_up:
            r = min(r, 1.0)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        if r != 1.0:
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        if not self.pad_to_square:
            return img, r, (0, 0)
        dw, dh = int(max_thresh - new_w), int(max_thresh - new_h)
        if self.minimum_rectangle:
            dw, dh = np.mod(dw, self.division), np.mod(dh, self.division)
        if self.center_padding:
            dw /= 2
            dh /= 2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        else:
            left, top = 0, 0
            right, bottom = dw, dh
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=border_val)
        return img, r, (left, top)

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        max_thresh = np.random.choice(self.max_threshes)
        img, r, (left, top) = self.make_border(box_info.img, max_thresh, box_info.padding_val)
        box_info.img = img
        box_info.ext_prop.update(
            {"r": r, "left": left, "top": top}
        )
        if box_info.boxes is not None and len(box_info.boxes):
            assert box_info.xyxy and not box_info.normalized_box, "box form should be xyxy and not normalized coord"
            box_info.boxes = box_info.boxes * r
            box_info.boxes[:, [0, 2]] = box_info.boxes[:, [0, 2]] + left
            box_info.boxes[:, [1, 3]] = box_info.boxes[:, [1, 3]] + top
        return box_info

    def reset(self, **settings):
        super(RandScaleToMax, self).reset(**settings)
        max_threshes = settings.get('max_threshes', None)
        if max_threshes is not None:
            self.max_threshes = max_threshes
        return self


class RandScaleMinMax(BasicTransform):
    def __init__(self, min_threshes, max_thresh=1024, **kwargs):
        kwargs['p'] = 1.0
        super(RandScaleMinMax, self).__init__(**kwargs)
        assert isinstance(min_threshes, list)
        self.min_threshes = min_threshes
        self.max_thresh = max_thresh

    def scale_img(self, img: np.ndarray, min_thresh):
        h, w = img.shape[:2]
        min_side, max_side = min(h, w), max(h, w)
        r = min(min_thresh / min_side, self.max_thresh / max_side)
        if r != 1:
            img = cv.resize(img, (int(round(w * r)), int(round(h * r))), interpolation=cv.INTER_LINEAR)
        return img, r

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        min_thresh = np.random.choice(self.min_threshes)
        img, ratio = self.scale_img(box_info.img, min_thresh)
        box_info.img = img
        if box_info.boxes is not None and len(box_info.boxes):
            assert not box_info.normalized_box, " shouldn't be normalized coord"
            box_info.boxes = box_info.boxes * ratio
        return box_info


class LRFlip(BasicTransform):
    """
    左右翻转
    """

    def __init__(self, **kwargs):
        super(LRFlip, self).__init__(**kwargs)

    @staticmethod
    def img_aug(img: np.ndarray) -> np.ndarray:
        img = np.fliplr(img)
        return img

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        _, w = box_info.img.shape[:2]
        box_info.img = self.img_aug(box_info.img)
        if box_info.boxes is not None and len(box_info.boxes):
            assert box_info.xyxy, "box form should be xyxy "
            box_info.boxes[:, [2, 0]] = w - box_info.boxes[:, [0, 2]]
        return box_info


class UDFlip(BasicTransform):
    """
    上下翻转
    """

    def __init__(self, **kwargs):
        super(UDFlip, self).__init__(**kwargs)

    @staticmethod
    def img_aug(img: np.ndarray) -> np.ndarray:
        img = np.flipud(img)
        return img

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        h, _ = box_info.img.shape[:2]
        box_info.img = self.img_aug(box_info.img)
        if box_info.boxes is not None and len(box_info.boxes):
            assert box_info.xyxy, "box form should be xyxy "
            box_info.boxes[:, [3, 1]] = h - box_info.boxes[:, [1, 3]]
        return box_info


class RandPerspective(BasicTransform):
    def __init__(self,
                 target_size=None,
                 degree=(0, 0),
                 translate=0.0,
                 scale=(1.0, 1.0),
                 shear=0,
                 perspective=0.0,
                 **kwargs):
        kwargs['p'] = 1.0
        super(RandPerspective, self).__init__(**kwargs)
        assert isinstance(target_size, tuple) or target_size is None
        assert isinstance(degree, tuple)
        assert isinstance(scale, tuple)
        self.target_size = target_size
        self.degree = degree
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective

    def reset(self, **settings):
        super(RandPerspective, self).reset(**settings)
        target_size = settings.get('target_size', None)
        degree = settings.get('degree', None)
        translate = settings.get('translate', None)
        scale = settings.get('scale', None)
        shear = settings.get('shear', None)
        perspective = settings.get('perspective', None)
        if target_size is not None:
            assert isinstance(target_size, tuple)
            self.target_size = target_size
        if degree is not None:
            assert isinstance(degree, tuple)
            self.degree = degree
        if translate is not None:
            self.translate = translate
        if scale is not None:
            assert isinstance(scale, tuple)
            self.scale = scale
        if shear is not None:
            self.shear = shear
        if perspective is not None:
            self.perspective = perspective
        return self

    def get_transform_matrix(self, img):
        if self.target_size is not None:
            width, height = self.target_size
        else:
            height, width = img.shape[:2]

        matrix_c = np.eye(3)
        matrix_c[0, 2] = -img.shape[1] / 2
        matrix_c[1, 2] = -img.shape[0] / 2

        matrix_p = np.eye(3)
        matrix_p[2, 0] = random.uniform(-self.perspective, self.perspective)
        matrix_p[2, 1] = random.uniform(-self.perspective, self.perspective)

        matrix_r = np.eye(3)
        angle = np.random.uniform(self.degree[0], self.degree[1])
        scale = np.random.uniform(self.scale[0], self.scale[1])
        matrix_r[:2] = cv.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

        matrix_t = np.eye(3)
        matrix_t[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        matrix_t[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * height

        matrix_s = np.eye(3)
        matrix_s[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        matrix_s[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        return matrix_t @ matrix_s @ matrix_r @ matrix_p @ matrix_c, width, height, scale

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        transform_matrix, width, height, scale = self.get_transform_matrix(box_info.img)
        if self.perspective:
            box_info.img = cv.warpPerspective(box_info.img,
                                              transform_matrix,
                                              dsize=(width, height),
                                              borderValue=box_info.padding_val)
        else:  # affine
            box_info.img = cv.warpAffine(box_info.img,
                                         transform_matrix[:2],
                                         dsize=(width, height),
                                         borderValue=box_info.padding_val)
        if box_info.boxes is None or len(box_info.boxes) == 0:
            return box_info
        n = len(box_info.boxes)
        if n:
            assert box_info.xyxy and not box_info.normalized_box, "box form should be xyxy and not normalized coord"
            xy = np.ones((n * 4, 3))
            # x1,y1,x2,y2,x1,y2,x2,y1
            xy[:, :2] = box_info.boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = (xy @ transform_matrix.T)
            if self.perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (box_info.boxes[:, 2] - box_info.boxes[:, 0]) * (box_info.boxes[:, 3] - box_info.boxes[:, 1])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 2) & (h > 2) & (area / (area0 * scale + 1e-16) > 0.2) & (ar < 20)
            box_info.boxes = xy[i]
            if box_info.labels is not None and len(box_info.labels) > 0:
                box_info.labels = box_info.labels[i]
            return box_info


class Mosaic(BasicTransform):
    def __init__(self,
                 candidate_box_info,
                 color_gitter=None,
                 target_size=640,
                 rand_center=True,
                 translate=0.1,
                 scale=(0.5, 1.5),
                 degree=(0, 0),
                 **kwargs):
        kwargs['p'] = 1.0
        super(Mosaic, self).__init__(**kwargs)
        assert isinstance(candidate_box_info, list)
        self.candidate_box_info = candidate_box_info
        if color_gitter is None:
            color_gitter = Identity()
        self.color_gitter = color_gitter
        self.target_size = target_size
        self.rand_center = rand_center
        self.affine = RandPerspective(target_size=(target_size, target_size),
                                      translate=translate,
                                      scale=scale,
                                      degree=degree)
        self.scale_max = RandScaleToMax(max_threshes=[target_size], pad_to_square=False)

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        mosaic_border = (-self.target_size // 2, -self.target_size // 2)
        if self.rand_center:
            yc, xc = [int(random.uniform(-x, 2 * self.target_size + x)) for x in mosaic_border]
        else:
            yc, xc = [self.target_size, self.target_size]
        indices = [random.randint(0, len(self.candidate_box_info) - 1) for _ in range(3)]
        img4 = np.tile(np.array(box_info.padding_val, dtype=np.uint8)[None, None, :],
                       (self.target_size * 2, self.target_size * 2, 1))
        # img4 = np.ones(shape=(self.target_size * 2, self.target_size * 2, 3)) * box_info.PADDING_VAL
        box_info4 = list()
        for i, index in enumerate([1] + indices):
            if i == 0:
                box_info_i = box_info
            else:
                box_info_i = self.candidate_box_info[index].clone().load_img()
            box_info_i = self.color_gitter(box_info_i)
            box_info_i = self.scale_max(box_info_i)
            h, w = box_info_i.img.shape[:2]
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.target_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.target_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.target_size * 2), min(self.target_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = box_info_i.img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            if box_info_i.boxes is not None and len(box_info_i.boxes):
                box_info_i.boxes[:, [0, 2]] = box_info_i.boxes[:, [0, 2]] + padw
                box_info_i.boxes[:, [1, 3]] = box_info_i.boxes[:, [1, 3]] + padh
                box_info4.append(box_info_i)
        box_info.img = img4
        if len(box_info4):
            box_4 = np.concatenate([item.boxes for item in box_info4], axis=0)
            np.clip(box_4, 0, 2 * self.target_size, out=box_4)
            label_4 = [item.labels for item in box_info4 if item.labels is not None]
            label_4 = np.concatenate(label_4, axis=0) if len(label_4) > 0 else None
            box_info.boxes = box_4
            box_info.labels = label_4
        else:
            return self.affine(box_info)
        valid_index = np.bitwise_and((box_info.boxes[:, 2] - box_info.boxes[:, 0]) > 2,
                                     (box_info.boxes[:, 3] - box_info.boxes[:, 1]) > 2)
        box_info.boxes = box_info.boxes[valid_index, :]
        if box_info.labels is not None and len(box_info.labels) > 0:
            box_info.labels = box_info.labels[valid_index]
        return self.affine(box_info)


class MosaicWrapper(Mosaic):
    def __init__(self, sizes, **kwargs):
        super(MosaicWrapper, self).__init__(**kwargs)
        self.sizes = sizes

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        rand_size = np.random.choice(self.sizes)
        self.target_size = rand_size
        self.affine.reset(target_size=(rand_size, rand_size))
        self.scale_max.reset(max_threshes=[rand_size])
        return super(MosaicWrapper, self).aug(box_info)


class RandCrop(BasicTransform):
    def __init__(self, min_thresh=0.5, max_thresh=0.8, **kwargs):
        kwargs['p'] = 1.0
        super(RandCrop, self).__init__(**kwargs)
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh

    def get_crop_area(self, h, w):
        h_min = self.min_thresh * h if self.min_thresh <= 1 else min(self.min_thresh, h)
        h_max = self.max_thresh * h if self.max_thresh <= 1 else min(self.max_thresh, h)
        h_min, h_max = min(h_min, h_max), max(h_min, h_max)
        w_min = self.min_thresh * w if self.min_thresh <= 1 else min(self.min_thresh, w)
        w_max = self.max_thresh * w if self.max_thresh <= 1 else min(self.max_thresh, w)
        w_min, w_max = min(w_min, w_max), max(w_min, w_max)
        crop_h = int(np.random.uniform(h_min, h_max)) - 1
        crop_w = int(np.random.uniform(w_min, w_max)) - 1
        x0 = int(np.random.uniform(0, w - crop_w))
        y0 = int(np.random.uniform(0, h - crop_h))
        return x0, y0, crop_w, crop_h

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        h, w = box_info.img.shape[:2]
        x0, y0, crop_w, crop_h = self.get_crop_area(h, w)
        box_info.img = box_info.img[y0:y0 + crop_h, x0:x0 + crop_w, :]
        if box_info.boxes is not None and len(box_info.boxes) > 0:
            assert box_info.xyxy and not box_info.normalized_box, "box form should be xyxy and not normalized coord"
            cropped_boxes = box_info.boxes - np.array([x0, y0, x0, y0])
            cropped_boxes[..., [0, 2]] = cropped_boxes[..., [0, 2]].clip(min=0, max=crop_w)
            cropped_boxes[..., [1, 3]] = cropped_boxes[..., [1, 3]].clip(min=0, max=crop_h)
            c_w, c_h = (cropped_boxes[:, [2, 3]] - cropped_boxes[:, [0, 1]]).T
            area0 = (box_info.boxes[:, 2] - box_info.boxes[:, 0]) * (box_info.boxes[:, 3] - box_info.boxes[:, 1])
            area = c_w * c_h
            ar = np.maximum(c_w / (c_h + 1e-16), c_h / (c_w + 1e-16))
            i = (c_w > 2) & (c_w > 2) & (ar < 20) & (area / (area0 + 1e-16) > 0.2)
            box_info.boxes = cropped_boxes[i]
            if box_info.labels is not None and len(box_info.labels) > 0:
                box_info.labels = box_info.labels[i]
        return box_info


class NanoPerspective(BasicTransform):
    """
    reference to nanodet/data/warp.py
    """

    def __init__(self,
                 keep_ratio: bool = True,
                 dst_shape: Tuple[int, int] = (416, 416),
                 divisible: int = 0,
                 perspective: float = 0.0,
                 scale: Tuple[float, float] = (0.5, 1.4),
                 stretch: Tuple = ((1, 1), (1, 1)),
                 rotation: float = 0.0,
                 shear: float = 0.0,
                 translate: float = 0.2,
                 flip: float = 0.5,
                 **kwargs):
        kwargs['p'] = 1.0
        super(NanoPerspective, self).__init__(**kwargs)
        self.keep_ratio = keep_ratio
        self.divisible = divisible
        self.perspective = perspective
        self.scale_ratio = scale
        self.stretch_ratio = stretch
        self.rotation_degree = rotation
        self.shear_degree = shear
        self.flip_prob = flip
        self.translate_ratio = translate
        self.dst_shape = dst_shape

    @staticmethod
    def get_flip_matrix(prob=0.5):
        F = np.eye(3)
        if random.random() < prob:
            F[0, 0] = -1
        return F

    @staticmethod
    def get_perspective_matrix(perspective=0.0):
        """
        :param perspective:
        :return:
        """
        P = np.eye(3)
        P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
        return P

    @staticmethod
    def get_rotation_matrix(degree=0.0):
        """
        :param degree:
        :return:
        """
        R = np.eye(3)
        a = random.uniform(-degree, degree)
        R[:2] = cv.getRotationMatrix2D(angle=a, center=(0, 0), scale=1)
        return R

    @staticmethod
    def get_scale_matrix(ratio=(1, 1)):
        """

        :param ratio:
        """
        Scl = np.eye(3)
        scale = random.uniform(*ratio)
        Scl[0, 0] *= scale
        Scl[1, 1] *= scale
        return Scl, scale

    @staticmethod
    def get_stretch_matrix(width_ratio=(1, 1), height_ratio=(1, 1)):
        """

        :param width_ratio:
        :param height_ratio:
        """
        Str = np.eye(3)
        Str[0, 0] *= random.uniform(*width_ratio)
        Str[1, 1] *= random.uniform(*height_ratio)
        return Str

    @staticmethod
    def get_shear_matrix(degree):
        """

        :param degree:
        :return:
        """
        Sh = np.eye(3)
        Sh[0, 1] = math.tan(
            random.uniform(-degree, degree) * math.pi / 180
        )  # x shear (deg)
        Sh[1, 0] = math.tan(
            random.uniform(-degree, degree) * math.pi / 180
        )  # y shear (deg)
        return Sh

    @staticmethod
    def get_translate_matrix(translate, width, height):
        """

        :param translate:
        :return:
        """
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation
        T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation
        return T

    @staticmethod
    def get_resize_matrix(raw_shape, dst_shape, keep_ratio):
        """
        Get resize matrix for resizing raw img to input size
        :param raw_shape: (width, height) of raw image
        :param dst_shape: (width, height) of input image
        :param keep_ratio: whether keep original ratio
        :return: 3x3 Matrix
        """
        r_w, r_h = raw_shape
        d_w, d_h = dst_shape
        Rs = np.eye(3)
        if keep_ratio:
            C = np.eye(3)
            C[0, 2] = -r_w / 2
            C[1, 2] = -r_h / 2

            if r_w / r_h < d_w / d_h:
                ratio = d_h / r_h
            else:
                ratio = d_w / r_w
            Rs[0, 0] *= ratio
            Rs[1, 1] *= ratio

            T = np.eye(3)
            T[0, 2] = 0.5 * d_w
            T[1, 2] = 0.5 * d_h
            return T @ Rs @ C
        else:
            Rs[0, 0] *= d_w / r_w
            Rs[1, 1] *= d_h / r_h
            return Rs

    @staticmethod
    def get_minimum_dst_shape(
            src_shape: Tuple[int, int],
            dst_shape: Tuple[int, int],
            divisible: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Calculate minimum dst shape"""
        src_w, src_h = src_shape
        dst_w, dst_h = dst_shape

        if src_w / src_h < dst_w / dst_h:
            ratio = dst_h / src_h
        else:
            ratio = dst_w / src_w

        dst_w = int(ratio * src_w)
        dst_h = int(ratio * src_h)

        if divisible and divisible > 0:
            dst_w = max(divisible, int((dst_w + divisible - 1) // divisible * divisible))
            dst_h = max(divisible, int((dst_h + divisible - 1) // divisible * divisible))
        return dst_w, dst_h

    @staticmethod
    def warp_boxes(boxes, M, width, height):
        n = len(boxes)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            return xy.astype(np.float32)
        else:
            return boxes

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        raw_img = box_info.img
        height = raw_img.shape[0]
        width = raw_img.shape[1]
        C = np.eye(3)
        C[0, 2] = -width / 2
        C[1, 2] = -height / 2
        P = self.get_perspective_matrix(self.perspective)
        C = P @ C

        Scl, scale = self.get_scale_matrix(self.scale_ratio)
        C = Scl @ C

        Str = self.get_stretch_matrix(*self.stretch_ratio)
        C = Str @ C

        R = self.get_rotation_matrix(self.rotation_degree)
        C = R @ C

        Sh = self.get_shear_matrix(self.shear_degree)
        C = Sh @ C

        F = self.get_flip_matrix(self.flip_prob)
        C = F @ C

        T = self.get_translate_matrix(self.translate_ratio, width, height)
        M = T @ C
        dst_shape = self.dst_shape
        if self.keep_ratio:
            dst_shape = self.get_minimum_dst_shape(
                (width, height), dst_shape, self.divisible
            )
        ResizeM = self.get_resize_matrix((width, height), dst_shape, self.keep_ratio)
        M = ResizeM @ M
        img = cv.warpPerspective(raw_img, M, dsize=tuple(dst_shape), borderValue=box_info.padding_val)
        box_info.img = img
        if box_info.boxes is not None and len(box_info.boxes):
            assert box_info.xyxy and not box_info.normalized_box, "box form should be xyxy and not normalized coord"
            xy = self.warp_boxes(box_info.boxes, M, dst_shape[0], dst_shape[1])
            box_info.boxes = xy
            # w = xy[:, 2] - xy[:, 0]
            # h = xy[:, 3] - xy[:, 1]
            # area = w * h
            # area0 = (box_info.boxes[:, 2] - box_info.boxes[:, 0]) * (box_info.boxes[:, 3] - box_info.boxes[:, 1])
            # ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            # i = (w > 2) & (h > 2) & (area / (area0 * scale + 1e-16) > 0.2) & (ar < 20)
            # box_info.boxes = xy[i]
            # box_info.labels = box_info.labels[i]

        return box_info


class PixelNormalize(BasicTransform):
    def __init__(self, mean=(103.53, 116.28, 123.675), std=(57.375, 57.12, 58.395), **kwargs):
        """
        :param mean: bgr
        :param std: bgr
        :param kwargs:
        """
        kwargs['p'] = 1.0
        super(PixelNormalize, self).__init__(**kwargs)
        self.mean = mean
        self.std = std

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        mean = np.array(self.mean, dtype=np.float32)[None, None, :]
        std = np.array(self.std, dtype=np.float32)[None, None, :]
        box_info.img = (box_info.img.astype(np.float32) - mean) / std
        return box_info


class CoordNormalize(BasicTransform):
    def __init__(self, norm=True, **kwargs):
        kwargs['p'] = 1.0
        super(CoordNormalize, self).__init__(**kwargs)
        self.norm = norm

    @staticmethod
    def normalize(box_info):
        if not box_info.normalized_box:
            h, w = box_info.img.shape[:2]
            box_info.boxes = box_info.boxes / np.array([w, h, w, h])[None, :]
            box_info.normalized_box = True
        return box_info

    @staticmethod
    def un_normalize(box_info):
        if box_info.normalized_box:
            h, w = box_info.img.shape[:2]
            box_info.boxes = box_info.boxes * np.array([w, h, w, h])[None, :]
            box_info.normalized_box = False
        return box_info

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        return self.normalize(box_info) if self.norm else self.un_normalize(box_info)


class CoordTransform(BasicTransform):
    def __init__(self, c_type="xyxy", **kwargs):
        kwargs['p'] = 1.0
        super(CoordTransform, self).__init__(**kwargs)
        assert c_type.strip() in ['xyxy', 'xywh'], "only support xyxy,xywh"
        self.c_type = c_type

    @staticmethod
    def xyxy2xywh(box_info: BoxInfoCV):
        if box_info.boxes is None or len(box_info.boxes) == 0:
            return box_info
        if box_info.xyxy:
            box_info.boxes[:, 2:] = box_info.boxes[:, 2:] - box_info.boxes[:, :2]
            box_info.boxes[:, :2] = box_info.boxes[:, :2] + 0.5 * box_info.boxes[:, 2:]
            box_info.xyxy = False
        return box_info

    @staticmethod
    def xywh2xyxy(box_info: BoxInfoCV):
        if box_info.boxes is None or len(box_info.boxes) == 0:
            return box_info
        if not box_info.xyxy:
            box_info.boxes[:, :2] = box_info.boxes[:, :2] - 0.5 * box_info.boxes[:, 2:]
            box_info.boxes[:, 2:] = box_info.boxes[:, :2] + box_info.boxes[:, 2:]
            box_info.xyxy = True
        return box_info

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        return self.xyxy2xywh(box_info) if self.c_type.strip() == "xywh" else self.xywh2xyxy(box_info)


class ChannelSwitch(BasicTransform):
    def __init__(self, channel_order=(2, 1, 0), **kwargs):
        kwargs['p'] = 1.0
        super(ChannelSwitch, self).__init__(**kwargs)
        self.channel_order = channel_order

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        box_info.img = box_info.img[..., list(self.channel_order)]
        return box_info


class OneOf(BasicTransform):
    def __init__(self, transforms, **kwargs):
        kwargs['p'] = 1.0
        super(OneOf, self).__init__(**kwargs)
        if isinstance(transforms[0], BasicTransform):
            prob = float(1 / len(transforms))
            transforms = [(prob, transform) for transform in transforms]
        probs, transforms = zip(*transforms)
        probs, transforms = list(probs), list(transforms)
        self.probs = probs
        self.transforms = transforms

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        index = np.random.choice(a=range(len(self.probs)), p=self.probs)
        box_info = self.transforms[index](box_info)
        return box_info


class Sequence(BasicTransform):
    def __init__(self, transforms, **kwargs):
        kwargs['p'] = 1.0
        super(Sequence, self).__init__(**kwargs)
        self.transforms = transforms

    def aug(self, box_info: BoxInfoCV) -> BoxInfoCV:
        for transform in self.transforms:
            box_info = transform(box_info)
        return box_info
