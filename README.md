# BPFINet
# [BPFINet: Boundary-aware Progressive Feature Integration Network for Salient Object Detection](https://doi.org/10.1016/j.neucom.2021.04.078)

This repo. is an official implementation of the *BPFINet* , which has been accepted in the journal *Neurocomputing, 2021*. 

The main pipeline is shown as the following, 
![BPFINet](figures/network.png)

And some visualization results are listed 
![results](figures/results.png)

![modules](figures/modules.png)

## Dependencies 
```
>= Pytorch 1.0.0
OpenCV-Python
[optional] matlab
```

## Training
```
python main.py 
```

## Test
```
 python3 main.py --mode=test 
```
We provide the trained model file ([Baidu](https://drive.google.com/file/d/1bXERDgTKfzkZfXKs8z5vj1QNM3zL-QTL/view?usp=sharing)) [code:]

The saliency maps are also available ([Baidu](https://drive.google.com/file/d/1sIqEKDCi_rSY4t1THPlBSyAd05F2ve_Q/view?usp=sharing)). [code:]

## Citation
Please cite the `BPFINet` in your publications if it helps your research:
```
@article{CHEN2021,
  title = {BPFINet: Boundary-aware Progressive Feature Integration Network for Salient Object Detection},
  author = {Tianyou Chen and Xiaoguang Hu and Jin Xiao and Guofeng Zhang},
  journal = {Neurocomputing},
  year = {2021},
}
```
