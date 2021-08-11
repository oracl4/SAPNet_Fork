import random

import cv2
import numpy as np
import torch

from detection.data.container import Container
from detection.structures import PolygonMasks

import albumentations
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox
from collections import namedtuple

class CustomCutout(DualTransform):
    """
    Custom Cutout augmentation with handling of bounding boxes 
    Note: (only supports square cutout regions)
    
    Author: Kaushal28
    Reference: https://arxiv.org/pdf/1708.04552.pdf
    """
    
    def __init__(
        self,
        fill_value=0,
        bbox_removal_threshold=0.5,
        min_cutout_size=128,
        max_cutout_size=256,
        always_apply=False,
        p=0.5
    ):
        """
        Class construstor
        
        :param fill_value: Value to be filled in cutout (default is 0 or black color)
        :param bbox_removal_threshold: Bboxes having content cut by cutout path more than this threshold will be removed
        :param min_cutout_size: minimum size of cutout (192 x 192)
        :param max_cutout_size: maximum size of cutout (512 x 512)
        """
        super(CustomCutout, self).__init__(always_apply, p)  # Initialize parent class
        self.fill_value = fill_value
        self.bbox_removal_threshold = bbox_removal_threshold
        self.min_cutout_size = min_cutout_size
        self.max_cutout_size = max_cutout_size
        
    def _get_cutout_position(self, img_height, img_width, cutout_size):
        """
        Randomly generates cutout position as a named tuple
        
        :param img_height: height of the original image
        :param img_width: width of the original image
        :param cutout_size: size of the cutout patch (square)
        :returns position of cutout patch as a named tuple
        """
        position = namedtuple('Point', 'x y')
        return position(
            np.random.randint(0, img_width - cutout_size + 1),
            np.random.randint(0, img_height - cutout_size + 1)
        )
        
    def _get_cutout(self, img_height, img_width):
        """
        Creates a cutout pacth with given fill value and determines the position in the original image
        
        :param img_height: height of the original image
        :param img_width: width of the original image
        :returns (cutout patch, cutout size, cutout position)
        """
        cutout_size = np.random.randint(self.min_cutout_size, self.max_cutout_size + 1)
        cutout_position = self._get_cutout_position(img_height, img_width, cutout_size)
        return np.full((cutout_size, cutout_size, 3), self.fill_value), cutout_size, cutout_position
        
    def apply(self, image, **params):
        """
        Applies the cutout augmentation on the given image
        
        :param image: The image to be augmented
        :returns augmented image
        """
        image = image.copy()  # Don't change the original image
        self.img_height, self.img_width, _ = image.shape
        cutout_arr, cutout_size, cutout_pos = self._get_cutout(self.img_height, self.img_width)
        
        # Set to instance variables to use this later
        self.image = image
        self.cutout_pos = cutout_pos
        self.cutout_size = cutout_size
        
        image[cutout_pos.y:cutout_pos.y+cutout_size, cutout_pos.x:cutout_size+cutout_pos.x, :] = cutout_arr
        return image
    
    def apply_to_bbox(self, bbox, **params):
        """
        Removes the bounding boxes which are covered by the applied cutout
        
        :param bbox: A single bounding box coordinates in pascal_voc format
        :returns transformed bbox's coordinates
        """

        # Denormalize the bbox coordinates
        bbox = denormalize_bbox(bbox, self.img_height, self.img_width)
        x_min, y_min, x_max, y_max = tuple(map(int, bbox))

        bbox_size = (x_max - x_min) * (y_max - y_min)  # width * height
        overlapping_size = np.sum(
            (self.image[y_min:y_max, x_min:x_max, 0] == self.fill_value) &
            (self.image[y_min:y_max, x_min:x_max, 1] == self.fill_value) &
            (self.image[y_min:y_max, x_min:x_max, 2] == self.fill_value)
        )

        # Remove the bbox if it has more than some threshold of content is inside the cutout patch
        if overlapping_size / bbox_size > self.bbox_removal_threshold:
            return normalize_bbox((0, 0, 0, 0), self.img_height, self.img_width)

        return normalize_bbox(bbox, self.img_height, self.img_width)

    def get_transform_init_args_names(self):
        """
        Fetches the parameter(s) of __init__ method
        :returns: tuple of parameter(s) of __init__ method
        """
        return ('fill_value', 'bbox_removal_threshold', 'min_cutout_size', 'max_cutout_size', 'always_apply', 'p')

class cutout(object):

    def __init__(self, cutout_ratio=1):
        self.cutout_ratio = cutout_ratio

    def __call__(self, results):
        cutout = True if np.random.rand() < self.cutout_ratio else False
        results['cutout'] = cutout
        
        if results['cutout']:
            image = results['img']
            bboxes = results['boxes']
            labels = results['labels']

            # print("Before===========================================")
            # print(len(results['img_shape']))

            augmentation = albumentations.Compose([
                                    albumentations.LongestMaxSize(max_size=1024),
                                    CustomCutout(p=1),
                                    # albumentations.OneOf([  # One of blur or adding gauss noise
                                    #     albumentations.Blur(p=0.50),  # Blurs the image
                                    #     albumentations.GaussNoise(var_limit=5.0 / 255.0, p=0.50)  # Adds Gauss noise to image
                                    # ], p=1)
                                ], bbox_params = {
                                    'format': 'pascal_voc',
                                    'label_fields': ['labels']
                                })

            aug_result = augmentation(image=image, bboxes=bboxes, labels=labels)
            
            aug_img = aug_result['image']
            aug_boxes= np.array(aug_result['bboxes'])
            aug_labels = np.array(aug_result['labels'])

            results['img'] = aug_img.astype('uint8')
            results['boxes'] = aug_boxes.astype('int64')
            results['labels'] = aug_labels

            img = aug_result['image']
            aug_width, aug_height = img.shape[:2]
            aug_shape = (aug_width, aug_height)

            results['img_shape'] = aug_shape
            
            # if(len(results['boxes']) == 0):
            #     print("There is empty bounding box")

            # print("After===========================================")
            # print(len(results['img_shape']))

            # if(results['boxes'].size == 0):
            #     print("sheet")

            # # Visualize
            # img = results['img']
            # boxes = np.array(results['boxes'])
            # labels = np.array(results['labels'])
            # ann = np.concatenate((boxes, labels[:,None]), axis=1)
            # for box in ann:
            #     box = box.astype(int)
            #     print(box)
            #     x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            #     labels = box[4]
            #     img = cv2.rectangle(img, (x1, y1), (x2, y2), (36,255,12), 1)
            #     img = cv2.putText(img, str(labels), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)

            # # Using cv2.imshow() method 
            # cv2.imshow("window", img)
            # cv2.imwrite("random_image/" + str(random.randint(1, 100)) + ".jpg", img)
            # cv2.waitKey(0) 
            # cv2.destroyAllWindows() 
        
        return results
        

class random_flip(object):
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def flip_boxes(self, results):
        if 'boxes' in results:
            w, h = results['img_shape']
            boxes = results['boxes']
            flipped = boxes.copy()
            flipped[..., 0] = w - boxes[..., 2] - 1
            flipped[..., 2] = w - boxes[..., 0] - 1
            results['boxes'] = flipped

    def flip_masks(self, results):
        if 'masks' in results:  # list[list[ndarray[double]]]
            w, h = results['img_shape']
            masks = results['masks']
            for mask in masks:
                for polygon in mask:
                    # np.array([x0, y0, x1, y1, ..., xn, yn]) (n >= 3)
                    polygon[0::2] = w - polygon[0::2] - 1
            results['masks'] = masks

    def __call__(self, results):
        flip = True if np.random.rand() < self.flip_ratio else False
        results['flip'] = flip
        if results['flip']:
            results['img'] = np.flip(results['img'], axis=1)
            self.flip_boxes(results)
            self.flip_masks(results)
        return results

class resize(object):
    def __init__(self, min_size, max_size=None, keep_ratio=True):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.keep_ratio = keep_ratio

    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return w, h
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return ow, oh

    def resize_boxes(self, results):
        w, h = results['img_shape']
        boxes = results['boxes'].copy() * results['scale_factor']
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h - 1)
        results['boxes'] = boxes

    def resize_masks(self, results):
        if 'masks' in results:
            masks = results['masks']
            for mask in masks:
                for polygon in mask:
                    scale_y = scale_x = results['scale_factor']
                    # inplace modify
                    polygon[0::2] *= scale_x
                    polygon[1::2] *= scale_y
            results['masks'] = masks

    def __call__(self, results):
        w, h = results['img_shape']
        new_w, new_h = self.get_size((w, h))
        image = cv2.resize(results['img'], dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)

        results['img'] = image

        results['img_shape'] = (new_w, new_h)
        results['origin_img_shape'] = (w, h)
        results['scale_factor'] = float(new_w) / w

        self.resize_boxes(results)
        self.resize_masks(results)

        return results

class normalize(object):
    """Normalize the image.
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_bgr (bool): Whether to convert the image from RGB to BGR,
            default is true.
    """

    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1), to_01=False, to_bgr=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_01 = to_01
        self.to_bgr = to_bgr

    def __call__(self, results):
        img = results['img'].astype(np.float32)
        if self.to_01:
            img = img / 255.0
        if self.to_bgr:
            img = img[:, :, [2, 1, 0]]
        img = (img - self.mean) / self.std
        results['img'] = img
        results['img_norm'] = dict(mean=self.mean, std=self.std, to_01=self.to_01, to_bgr=self.to_bgr)
        return results


class pad(object):
    def __init__(self, size_divisor=0, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def __call__(self, results):
        img = results['img']
        pad_shape = img.shape
        if self.size_divisor > 0:
            pad_shape = list(pad_shape)
            pad_shape[0] = int(np.ceil(img.shape[0] / self.size_divisor)) * self.size_divisor
            pad_shape[1] = int(np.ceil(img.shape[1] / self.size_divisor)) * self.size_divisor
            pad_shape = tuple(pad_shape)
            pad = np.full(pad_shape, self.pad_val, dtype=img.dtype)
            pad[:img.shape[0], :img.shape[1], ...] = img
        else:
            pad = img
        results['img'] = pad
        results['pad_shape'] = pad_shape
        return results


def de_normalize(image, img_meta):
    assert 'img_norm' in img_meta
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = image * img_meta['img_norm']['std'] + img_meta['img_norm']['mean']
    if img_meta['img_norm']['to_01']:
        image *= 255
    if img_meta['img_norm']['to_bgr']:
        image = image[:, :, [2, 1, 0]]
    image = image.astype('uint8')
    return image

class collect(object):

    def __init__(self, meta_keys=('img_shape', 'crop', 'flip', 'cutout', 'origin_img_shape', 'scale_factor', 'img_norm', 'img_info', 'pad_shape')):
        self.meta_keys = meta_keys

    def __call__(self, results):
        img = torch.from_numpy(results['img'].transpose(2, 0, 1))
        
        # print("Something===========================================")
        # print(results)
        
        target = {
            'boxes': torch.from_numpy(results['boxes'].astype(np.float32)),
            'labels': torch.from_numpy(results['labels']),
        }
        # if 'masks' in results:
        #     target['masks'] = PolygonMasks(results['masks'])

        img_meta = {
        }
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]

        target = Container(target)
        return img, img_meta, target

class compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, results):
        for transform in self.transforms:
            results = transform(results)
        return results
