# Joint_360depth
This repository contains the code of the paper "[Improving 360 monocular depth estimation via non-local dense prediction transformer and Joint supervised and self-supervised learning](https://arxiv.org/abs/2109.10563)".

Some of our codes are based on the following repositories: [EBS](https://github.com/gdlg/panoramic-depth-estimation), [monodepth2](https://github.com/nianticlabs/monodepth2), [Omnidepth](https://github.com/meder411/OmniDepth-PyTorch), [DPT](https://github.com/isl-org/DPT) and [Non-local neural network](https://github.com/AlexHex7/Non-local_pytorch).

We'd like to thank the authors and users providing the codes.

Due to personal reasons, other parts of the codes will be uploaded in a next few weeks.
## Changelog
[2021-10-09 ] Release inference code and pre-trained models

## 1. Setup

This codes are tested under PyTorch (1.7) with a single NVIDIA v100 GPU (418.56, cuda 10.0).


### Installation
#### 1) clone repository
~~~bash
git clone https://github.com/yuniw18/Joint_360depth.git
~~~
#### 2) Set up dependencies
Using Anaconda virtual env, environment can be set as follows.

~~~bash
conda env create --file depth.yaml
conda activate depth_1.7
~~~

## 2. Inreference
#### 1) Download pretrained model

Download the pretrained model from this [link](https://drive.google.com/drive/folders/1IcyB1tgvs_U2KgzAVM9Qo861RmKNCnUd?usp=sharing). 

- Ours_w_NLFB : model used in the paper
- Super_S3D_Fres & Joint_S3D_Fres : models trained with [Structure3D](https://github.com/bertjiazheng/Structured3D) dataset, which are not used in the paper. 

#### Note on the models
We observe that Ours_w_NLFB works well for public dataset, however, it does not works well for complicated real world scenes relatively.
Therefore, Super_S3D_Fres & Joint_S3D_Fres are additionally trained to support complicated real world scenes with full angular resolutions. Super_S3D_Fres & Joint_S3D_Fres provide similar results except some cases (e.g., window) as shown in the figure below. By refering to the figure & table below, select the model according to the purpose .

<img src="./Assets/Comp.png">

The quantitative results in this table are calculated for 360&deg; x 90&deg; image resulutions.

| Model               | Input angular resolution  | Joint learning? | Dataset for supervised learning |Matterpot abs. rel. |Matterpot Sq.rel |Matterpot RMS |Matterpot RMSlog | Matterpot delta < 1.25  |
|---------------------|--------------------|----------------|--------------------------|-----------------|------|------|------|----------------|
| Ours_w_NLFB     |360&deg;x90&deg; | Yes             | [3D60](https://github.com/VCL3D/3D60) | 0.0700                     | 0.0287                 | 0.3032 |0.1051|0.9599|
| Super_S3D_Fres  |360&deg;x180&deg; |   No            |[Structure3D](https://github.com/bertjiazheng/Structured3D) | 0.0631              | 0.0400                 | 0.3454          |0.1216|0.9433
| Joint_S3D_Fres|360&deg;x180&deg; |Yes     | [Structure3D](https://github.com/bertjiazheng/Structured3D) | 0.0642                | 0.0389                 | 0.3388          |0.1207|0.9447


#### 2) To start quickly, go to inference folder & run the following command
Do not use `--Input_Full` option when test 'Ours_w_NLFB' model
~~~bash
python3 inference_main.py --checkpoint_path [pretrained model path] --Input_Full
~~~
Estimated depths of images in `sample` folder will be saved in the `output` folder

#### 3) To estimate the depths of custom images, run the following command

~~~bash
python3 inference_main.py --data_path [image path] --output_path [path where results will be saved]
~~~

## To do list
- [x] Code for inference  
- [ ] Code for quantitavie evaluation 
- [ ] Code for training
- [ ] Video set used in the paper

## Citation
```
@article{yun2021improving,
  title={Improving 360 Monocular Depth Estimation via Non-local Dense Prediction Transformer and Joint Supervised and Self-supervised Learning},
  author={Yun, IlWi and Lee, Hyuk-Jae and Rhee, Chae Eun},
  journal={arXiv preprint arXiv:2109.10563},
  year={2021}
}
``` 
## License
MIT License
