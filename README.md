# UprightNet
PyTorch implementation of paper "UprightNet: Geometry-Aware Camera Orientation Estimation from Single Images", ICCV 2019


1. Training

* To train the network on the InteriorNet, run 
```bash
	python train.py --mode ResNet --dataset interiornet --w_grad 0.25 --w_pose 2.0
```

 	
* To train the network on the ScanNet, run 
```bash
	python train.py --mode ResNet --dataset scannet --w_grad 0.25 --w_pose 0.5
```


2. Testing: 
* To evaluate InteriorNet pretrained network on the InteriorNet testset, run
```bash
	python test.py --mode ResNet --dataset interiornet
```

* To evaluate ScanNet pretrained network on the ScanNet testset, run 
```bash
	python test.py --mode ResNet --dataset scannet
```





