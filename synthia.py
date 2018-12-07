"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage:  run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python synthia.py train --dataset=/path/to/synthia/ --model=coco

    # Train a new model starting from ImageNet weights
    python synthia.py train --dataset=/path/to/synthia/ --model=imagenet

    # Continue training a model that you had trained earlier
    python synthia.py train --dataset=/path/to/synthia/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python synthia.py train --dataset=/path/to/synthia/ --model=last

    # Run evaluatoin on the last model you trained
    python synthia.py evaluate --dataset=/path/to/synthia/ --model=last
"""

import os
import sys
import random
import math
import re
import time
import json
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.color
import skimage.io
import skimage.transform


import zipfile
import urllib.request
import shutil

from config import Config
import utils
import model as modellib

import torch

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class synthiaConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "synthia"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 22  # background + 22 shapes

    IMAGE_MIN_DIM = 760
    IMAGE_MAX_DIM = 1280


############################################################
#  Dataset
############################################################

class synthiaDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_synthia(self, dataset_dir,subset):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("synthia", 1, "sky")
        self.add_class("synthia", 2, "Building")
        self.add_class("synthia", 3, "Road")
        self.add_class("synthia", 4, "Sidewalk")
        self.add_class("synthia", 5, "Fence")
        self.add_class("synthia", 6, "Vegetation")
        self.add_class("synthia", 7, "Pole")
        self.add_class("synthia", 8, "Car")
        self.add_class("synthia", 9, "Traffic sign")
        self.add_class("synthia", 10, "Pedestrian")
        self.add_class("synthia", 11, "Bicycle")
        self.add_class("synthia", 12, "Motorcycle")
        self.add_class("synthia", 13, "Parking-slot")
        self.add_class("synthia", 14, "Road-work")
        self.add_class("synthia", 15, "Traffic light")
        self.add_class("synthia", 16, "Terrain")
        self.add_class("synthia", 17, "Rider")
        self.add_class("synthia", 18, "Truck")
        self.add_class("synthia", 19, "Bus")
        self.add_class("synthia", 20, "Train")
        self.add_class("synthia", 21, "Wall")
        self.add_class("synthia", 22, "Lanemarking")        

        if subset == "test":
            fname="test.txt"
        else:
            fname="train.txt"
        
        # obtain the image ids
        with open(fname) as f:
            content = f.readlines()
        image_ids = [x.strip() for x in content]
        
        for image_id in image_ids:
            if int(image_id)<201:
                Path=os.path.join(dataset_dir, "val","{}.png".format(image_id))
                self.add_image(
                    "synthia",
                    image_id=image_id,
                    path=Path)
            else:
                Path=os.path.join(dataset_dir, "train","{}.png".format(image_id))
                self.add_image(
                    "synthia",
                    image_id=image_id,
                    path=Path)
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,4] Numpy array.
        """
        # Load image
        imgPath = self.image_info[image_id]['path']
        img=skimage.io.imread(imgPath)
        return img

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "synthia":
            return info["synthia"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        path=info['path']
        mpath=path.replace("RGB","GT")
        label=cv2.imread(mpath,cv2.IMREAD_UNCHANGED)
        raw_mask=label[:,:,1]
        number=np.unique(raw_mask)
        number=number[1:]
        # you should change the mask shape according to the image shape
        mask = np.zeros([760, 1280, len(number)],dtype=np.uint8)
        class_ids=np.zeros([len(number)],dtype=np.uint32)
        for i,p in enumerate(number):
            location=np.argwhere(raw_mask==p)
            mask[location[:,0], location[:,1], i] = 1
            class_ids[i]=label[location[0,0],location[0,1],2]
#        mask = [m for m in mask if set(np.unique(m).flatten()) != {0}]
        return mask.astype(np.bool), class_ids.astype(np.int32)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Synthia.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on Synthia")
    parser.add_argument('--dataset', required=False,
                        default="/mnt/backup/jianyuan/synthia/RAND_CITYSCAPES/RGB",
                        metavar="/path/to/coco/",
                        help='Directory of the Synthia dataset')
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file ")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/mnt/backup/jianyuan/pytorch-mask-rcnn/logs",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--lr', required=False,
                        default=0.001,
#                        metavar="/mnt/backup/jianyuan/pytorch-mask-rcnn/logs",
                        help='Learning rate')
    parser.add_argument('--batchsize', required=False,
                        default=4,
                        help='Batch size')
    parser.add_argument('--steps', required=False,
                        default=200,
                        help='steps per epoch')    
    parser.add_argument('--device', required=False,
                        default="gpu",
                        help='gpu or cpu')                         
    args = parser.parse_args()

    # Configurations
    if args.command == "train":
        config = synthiaConfig()
    else:
        class InferenceConfig(synthiaConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
        inference_config = InferenceConfig()
    config.display()

    # Select Device
    if args.device == "gpu":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)

    model = model.to(device)
        
    # Select weights file to load
    if args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
            # load pre-trained weights from coco or imagenet
            model.load_pre_weights(model_path)
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
            model.load_weights(model_path)
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
            # load pre-trained weights from coco or imagenet
            model.load_pre_weights(model_path)
        else:
            model_path = args.model
            model.load_weights(model_path)
    else:
        model_path = ""
        model.load_weights(model_path)
#
##     Load weights
    print("Loading weights ", model_path)
    

    # For Multi-gpu training, please uncomment the following part
    # Notably, in the following codes, the model will be wrapped in DataParallel()
    # it means you need to change the model. to model.module
    # for example, model.train_model --> model.module.train_model
    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #    model = torch.nn.DataParallel(model)
    

    data_dir=args.dataset
    # Training dataset
    dataset_train = synthiaDataset()
    dataset_train.load_synthia(data_dir,"train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = synthiaDataset()
    dataset_val.load_synthia(data_dir,"test")
    dataset_val.prepare()

    # input parameters
    lr=float(args.lr)
    batchsize=int(args.batchsize)
    steps=int(args.steps)
    
    # Train or evaluate
    if args.command == "train":


        print(" Training Image Count: {}".format(len(dataset_train.image_ids)))
        print("Class Count: {}".format(dataset_train.num_classes))
        print("Validation Image Count: {}".format(len(dataset_val.image_ids)))
        print("Class Count: {}".format(dataset_val.num_classes))
        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=lr,
                    epochs=1,
                    BatchSize=batchsize,
                    steps=steps,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=lr/2,
                    epochs=3,
                    BatchSize=batchsize,
                    steps=steps,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=lr / 10,
                    epochs=15,
                    BatchSize=batchsize,
                    steps=steps,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        image_ids = np.random.choice(dataset_val.image_ids, 1)
        model.eval()
        APs = []
        for image_id in image_ids:
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_val, inference_config,
                                       image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
            # Run object detection
            results = model.detect([image],device)
            r = results[0]
            # Compute AP
            AP, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
            APs.append(AP)
    
        print("mAP: ", np.mean(APs))
        

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
