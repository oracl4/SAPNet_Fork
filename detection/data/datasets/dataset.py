import os
from collections import defaultdict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

import random
import cv2
import torch

from ..transforms import build_transforms


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False

        # TODO: remove
    flags = []
    for obj in anno:
        x, y, w, h = obj['bbox']
        flag = w >= 5 and h >= 5

        # segm = [0 for polygon in obj['segmentation'] if len(polygon) >= 6]
        flag = flag # and len(segm) > 0

        flags.append(flag)
    if not any(flags):
        return False

    return True


class ABSDataset(Dataset):
    def __init__(self, images_dir, transforms=(), dataset_name='', is_train=False):
        self.images_dir = images_dir
        self.transforms = build_transforms(transforms)
        self.dataset_name = dataset_name
        self.ids = []
        self.label2cat = {}
        self.is_train = is_train
        self.mosaic = False
        self.img_size = 480
        self.augment = True
        

    def __getitem__(self, idx):
        
        results = []
        
        # Training (do mosaic)
        if self.is_train == True:
            # Add mosaic
            mosaic = self.mosaic and random.random() < 1.0
            if mosaic:
                # Load mosaic
                results = load_mosaic(self, idx)
            else:
                results = load_image(self, idx)
        else:
            results = load_image(self, idx)
        
        # return with augmentation
        results = self.transforms(results)
        return results

    def get_annotations_by_image_id(self, img_id):
        """
        Args:
            img_id: image id
        Returns: dict with keys (img_info, boxes, labels)
        """
        raise NotImplementedError

    def get_annotations(self, idx):
        """
        Args:
            idx: dataset index
        Returns: dict with keys (img_info, boxes, labels)
        """
        img_id = self.ids[idx]
        return self.get_annotations_by_image_id(img_id)

    def __repr__(self):
        return '{} Dataset(size: {})'.format(self.dataset_name, len(self))

    def __len__(self):
        return len(self.ids)


class COCODataset(ABSDataset):
    def __init__(self, ann_file, root, transforms=(), remove_empty=False, dataset_name='', is_train=False):
        super().__init__(root, transforms, dataset_name, is_train)
        self.ann_file = ann_file

        from pycocotools.coco import COCO
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        n = len(self.ids)
        self.indices = range(int(n/2))

        if remove_empty:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.coco.getCatIds())
        }
        self.label2cat = {
            v: k for k, v in self.cat2label.items()
        }

    def get_annotations_by_image_id(self, img_id):
        coco = self.coco
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if len(anns) > 0 and 'iscrowd' in anns[0]:
            anns = [obj for obj in anns if obj["iscrowd"] == 0]

        boxes = []
        labels = []
        masks = []
        for obj in anns:
            x, y, w, h = obj["bbox"]

            # TODO: remove
            if not (w >= 5 and h >= 5):
                continue
            # segm = [0 for polygon in obj['segmentation'] if len(polygon) >= 6]
            # if len(segm) == 0:
            #     continue

            box = [x, y, x + w - 1, y + h - 1]
            
            label = self.cat2label[obj["category_id"]]
            boxes.append(box)
            labels.append(label)
            # segm = [np.array(polygon, dtype=np.float64) for polygon in obj['segmentation'] if len(polygon) >= 6]
            # masks.append(segm)
        boxes = np.array(boxes).reshape((-1, 4))
        labels = np.array(labels)

        return {'img_info': img_info, 'boxes': boxes, 'labels': labels, 'masks': masks}


class VOCDataset(ABSDataset):
    CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root, split='train', base_dir='.', transforms=(), keep_difficult=True, img_ext='.jpg', dataset_name=''):
        self.root = root
        self.split = split
        self.keep_difficult = keep_difficult
        self.img_ext = img_ext

        voc_root = os.path.join(self.root, base_dir)
        images_dir = os.path.join(voc_root, 'JPEGImages')
        self.annotation_dir = os.path.join(voc_root, 'Annotations')
        super().__init__(images_dir, transforms, dataset_name)

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, split + '.txt')
        with open(os.path.join(split_f), "r") as f:
            ids = [x.strip() for x in f.readlines()]
        self.ids = ids

        cat_ids = list(range(len(VOCDataset.CLASSES)))
        self.label2cat = {
            label: cat for label, cat in enumerate(cat_ids)
        }

    def get_annotations_by_image_id(self, img_id):
        ann_path = os.path.join(self.annotation_dir, img_id + '.xml')
        target = self.parse_voc_xml(ET.parse(ann_path).getroot())['annotation']
        img_info = {
            'width': target['size']['width'],
            'height': target['size']['height'],
            'id': img_id,
            'file_name': img_id + self.img_ext,
        }

        boxes = []
        labels = []
        difficult = []
        for obj in target['object']:
            is_difficult = bool(int(obj['difficult']))
            if is_difficult and not self.keep_difficult:
                continue
            label_name = obj['name']
            if label_name not in self.CLASSES:
                continue
            difficult.append(is_difficult)
            label_id = self.CLASSES.index(label_name)
            box = obj['bndbox']
            box = list(map(lambda x: float(x) - 1, [box['xmin'], box['ymin'], box['xmax'], box['ymax']]))
            boxes.append(box)
            labels.append(label_id)
        boxes = np.array(boxes).reshape((-1, 4))
        labels = np.array(labels)
        difficult = np.array(difficult)

        return {'img_info': img_info, 'boxes': boxes, 'labels': labels, 'difficult': difficult}

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag: {k: v if k == 'object' else v[0] for k, v in def_dic.items()}
            }
        elif node.text:
            text = node.text.strip()
            voc_dict[node.tag] = text
        return voc_dict

def load_image(self, idx):
    target = self.get_annotations(idx)
    img_info = target['img_info']
    file_name = img_info['file_name']
    img = Image.open(os.path.join(self.images_dir, file_name)).convert('RGB')

    results = {
        'img': np.array(img),
        'boxes': np.array(target['boxes']),
        'labels': np.array(target['labels']),
        'img_shape': (img.width, img.height),
        'img_info': img_info,
    }

    # print(results['boxes'])
    # print(results['labels'])

    if 'masks' in target:
        results['masks'] = target['masks']
    
    return results

def load_mosaic(self, index):
    # loads images in a 4-mosaic
    labels4 = []
    s = self.img_size
    mosaic_border = [-s // 2, -s // 2]
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices

    for i, index in enumerate(indices):
        # print(index)
        # Load image
        target = self.get_annotations(index)
        img_info = target['img_info']
        file_name = img_info['file_name']
        im = Image.open(os.path.join(self.images_dir, file_name)).convert('RGB')
        
        im = np.array(im)

        h0, w0 = im.shape[:2] # orig hw
        
        r = s / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)),
                            interpolation=cv2.INTER_LINEAR)

        h, w = im.shape[:2]
        img = im

        # Labels and boxes
        boxes =  (np.array(target['boxes']) * r).astype(int)    # n x 4
        labels = np.array(target['labels'])                     # n x ,
        
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        
        padw = x1a - x1b
        padh = y1a - y1b
        
        # print(labels.shape)

        ann = np.concatenate((boxes, labels[:,None]), axis=1)

        # x1 y1 x2 y2
        ann[:, 0] = (ann[:, 0]) + padw  # top left x
        ann[:, 1] = (ann[:, 1]) + padh  # top left y
        ann[:, 2] = (ann[:, 2]) + padw  # bottom right x
        ann[:, 3] = (ann[:, 3]) + padh  # bottom right y

        # print(ann)

        labels4.append(ann)

    # collect boxes
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, :3]):
        np.clip(x, 0, 2 * s, out=x)
    
    # final_boxes  = np.vstack(labels4)
    
    # # final_labels = np.hstack(final_labels)

    # # clip boxes to the image area
    # final_boxes[:, 0:] = np.clip(final_boxes[:, 0:], 0, s).astype(np.int32)
    # w = (final_boxes[:,2] - final_boxes[:,0])
    # h = (final_boxes[:,3] - final_boxes[:,1])
    
    # discard boxes where w or h <5
    w = (labels4[:,2] - labels4[:,0])
    h = (labels4[:,3] - labels4[:,1])
    labels4 = labels4[(w>=5) & (h>=5)]

    # print(labels4)

    # img_w, img_h = img4.shape[:2]

    # for box in labels4:

    #     # print(box)

    #     x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

    #     img4 = cv2.rectangle(img4, (x1, y1), (x2, y2), (36,255,12), 1)
    #     img4 = cv2.putText(img4, str(box[4]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
    
    boxes = labels4[:, 0:4]
    labels = labels4[:, 4]

    # print(boxes)
    # print(labels)

    # # Using cv2.imshow() method 
    # # Displaying the image 
    # cv2.imshow("window", img4)

    # cv2.imwrite("random_image/" + str(random.randint(1, 100)) + ".jpg", img4)
    
    # #waits for user to press any key 
    # #(this is necessary to avoid Python kernel form crashing)
    # cv2.waitKey(0) 
    
    # #closing all open windows 
    # cv2.destroyAllWindows() 

    # # Concat/clip labels
    # labels4 = np.concatenate(labels4, 0)
    # for x in (labels4[:, 1:], *segments4):
    #     np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    # img4, labels4, segments4 = copy_paste(img4, labels4, segments4, p=self.hyp['copy_paste'])
    # img4, labels4 = random_perspective(img4, labels4, segments4,
    #                                 degrees=self.hyp['degrees'],
    #                                 translate=self.hyp['translate'],
    #                                 scale=self.hyp['scale'],
    #                                 shear=self.hyp['shear'],
    #                                 perspective=self.hyp['perspective'],
    #                                 border=self.mosaic_border)  # border to remove
    
    img4_width, img4_height = img4.shape[:2]

    results = {
        'img': img4,
        'boxes': boxes,
        'labels': labels,
        'img_shape': (img4_width, img4_height),
        'img_info': img_info,
    }

    return results

