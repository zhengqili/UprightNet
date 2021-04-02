# UprightNet
PyTorch implementation of paper "UprightNet: Geometry-Aware Camera Orientation Estimation from Single Images", ICCV 2019
[[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Xian_UprightNet_Geometry-Aware_Camera_Orientation_Estimation_From_Single_Images_ICCV_2019_paper.pdf) 

## Dependency
The code is tested with Python3, Pytorch >= 1.0 and CUDA >= 10.0, the dependencies includes 
* tensorboardX
* matplotlib
* opencv
* scikit-image
* scipy

## Dataset
* Download pre-processed InteriorNet and ScanNet, as well as their corresponding training/validation/testing txt files from [link](https://drive.google.com/drive/folders/1WdNAESqDYcUPQyXAW6PvlcdQIYlOEXIw?usp=sharing)
* Modify the paths in train.py, test.py and txt files to match the dataset path in your machine.

## Coordindate system
Our upright and local coordindate systems are defined as follows (corresponding to the normal images in the pre-processed datasets):
* Z upward, Y right, X backward , equivalent to
* Roll negative -> image rotate counterclockwise, Pitch positive -> camera rotate up


## Training

* To train the network on the InteriorNet, run 
```bash
	python train.py --mode ResNet --dataset interiornet --w_grad 0.25 --w_pose 2.0
```

* To train the network on the ScanNet, run 
```bash
	python train.py --mode ResNet --dataset scannet --w_grad 0.25 --w_pose 0.5
```

## Testing: 
* Download checkpoints.zip from [link](https://drive.google.com/drive/folders/1WdNAESqDYcUPQyXAW6PvlcdQIYlOEXIw?usp=sharing), unzip it and make sure checkpoints folder is in the root directory of codebase.

* To evaluate InteriorNet pretrained network on the InteriorNet testset, run
```bash
	python test.py --mode ResNet --dataset interiornet
```

* To evaluate ScanNet pretrained network on the ScanNet testset, run 
```bash
	python test.py --mode ResNet --dataset scannet
```


