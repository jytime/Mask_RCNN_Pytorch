# Mask_RCNN_Pytorch

This is an implementation of the instance segmentation model [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Pytorch, based on the previous work of [Matterport](https://github.com/matterport/Mask_RCNN) and [lasseha](https://github.com/multimodallearning/pytorch-mask-rcnn).  Matterport's repository is an implementation on Keras and TensorFlow while lasseha's repository is an implementation on Pytorch.

## Features
This implementation has the following features:



## Requirements
* Python 3
* Pytorch 0.4
* matplotlib, scipy, skimage, h5py, numpy

## Compilation
Some of the instrctions come from lasseha's repository.
* We use the [Non-Maximum Suppression](https://github.com/ruotianluo/pytorch-faster-rcnn) from ruotianluo and the [RoiAlign](https://github.com/longcw/RoIAlign.pytorch) from longcw. Please follow the instrctions below to build the functions.

        cd nms/src/cuda/
        nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=arch
        cd ../../
        python build.py
        cd ../

        cd roialign/roi_align/src/cuda/
        nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=arch
        cd ../../
        python build.py
        cd ../../
        
         
    where arch is determined by your GPU model: 
    
    | GPU  | TitanX | GTX 960M | GTX 1070 | GTX 1080 (Ti) |
    | :--: | :--:   | :--:     | :--:     | :--: |
    | arch | sm_52  |sm_50     |sm_61     |sm_61 |
* If you want to train the network on the [COCO dataset](http://cocodataset.org/#home), please install the [Python COCO API](https://github.com/cocodataset/cocoapi) and create a symlink.

        ln -s /path/to/coco/cocoapi/PythonAPI/pycocotools/ pycocotools
* The pretrained models on COCO and ImageNet are available [here](https://drive.google.com/open?id=1LXUgC2IZUYNEoXr05tdqyKFZY0pZyPDc).  
