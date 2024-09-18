# WHALES
This is the official repository of WHALES.

Achieving high levels of safety and reliability in autonomous driving remains a critical challenge, especially due to occlusion and limited perception ranges in stand-alone systems. Cooperative perception among vehicles offers a promising solution, but existing research is hindered by datasets with a limited number of agents. Scaling up the number of cooperating agents is non-trivial and introduces significant computational and technical hurdles that have not been addressed in previous works.

To bridge this gap, we present **W**ireless en**H**anced **A**utonomous vehicles with **L**arge number of **E**ngaged agent**S** (**WHALES**), a dataset generated using CARLA simulator that features an unprecedented average of 8.4 agents per driving sequence. In addition to providing the largest number of agents and viewpoints among autonomous driving datasets, WHALES records agent behaviors, enabling cooperation across multiple tasks. This expansion allows for new supporting tasks in cooperative perception.

As a demonstration, we conduct experiments on agent scheduling task, where the ego agent selects one of multiple candidate agents to cooperate with, optimizing perception gains in autonomous driving. The WHALES dataset and codebase can be found at [WHALES GitHub](https://github.com/chensiweiTHU/WHALES).
# Installation

1. Clone our repository:

``
git clone https://github.com/chensiweiTHU/WHALES.git
``

2. Install mmetection3D

Please refer to https://github.com/open-mmlab/mmdetection3d and install mmdetecction3d==0.17.1

3. (Optional) Install OpenCOOD

Please refer to https://github.com/DerrickXuNu/OpenCOOD and install opencood.

# Preparing
1. Download the dataset

Download the dataset in Baidu netdisk https://pan.baidu.com/s/1dintX-d1T-m2uACqDlAM9A with a code `gduh`.

We also provide a mini version in https://pan.baidu.com/s/1b5JuUsGgBT3IaPoPmNpuAA with a code `nnqw`.
 
3. Preprocess the dataset

Like other datasets in mmdetection3D, first we put the dataset in `./data/whales/` directory.
Then run the following command to preprocess data.

``
python tools/create_data.py whales --root-path ./data/whales/ --out-dir ./data/whales/ --extra-tag whales
``

Like other datasets, you will find generated `*.pkl` files in `./data/whales/`

# Trianing

We use mmdetection3D format config files in ./configs_CooperativePerception/ to run our experiments. Run the following command to train the models.

``
bash tools/dist_train.sh your_config_file.py number_of_gpus
``
# Testing 

 Run the following command to test the trained models.

``
bash tools/dist_test.sh your_config_file.py your_model_file.pth number_of_gpus --eval bbox
``

We use mAP and NDS as our benchmark.

# Scheduling Algorithms

We focus on agent scheduling in our experiments, We implement scheduling algorithms in `./mmdet3d_plugin/datasets/pipelines/cooperative_perception.py`. You can write customized scheduling algorithms in this file.


# Experimental Results

## Stand-alone 3D Object Detection Benchmark (50m/100m)

| Method | $\text{AP}_{Veh}\uparrow$ | $\text{AP}_{Ped}\uparrow$ | $\text{AP}_{Cyc}\uparrow$ | mAP$\uparrow$ | mATE$\downarrow$ | mASE$\downarrow$ | mAOE$\downarrow$ | mAVE$\downarrow$ | NDS$\uparrow$ |
|---------|--------------------------|--------------------------|--------------------------|----------------|-------------------|-------------------|-------------------|-------------------|--------------|
| Pointpillars | 67.1/41.5 | 38.0/6.3 | 37.3/11.6 | 47.5/19.8 | 0.117/0.247 | 0.876/0.880 | 1.069/1.126 | 1.260/1.625 | 33.8/18.6 |
| SECOND | 58.5/38.8 | 27.1/12.1 | 24.1/12.9 | 36.6/21.2 | 0.106/0.156 | 0.875/0.878 | 1.748/1.729 | 1.005/1.256 | 28.5/20.3 |
| RegNet | 66.9/42.3 | 38.7/8.4 | 32.9/11.7 | 46.2/20.8 | 0.119/0.240 | 0.874/0.881 | 1.079/1.158 | 1.231/1.421 | 33.2/19.2 |
| VoxelNeXt | 64.7/42.3 | 52.2/27.4 | 35.9/9.0 | 50.9/26.2 | 0.075/0.142 | 0.877/0.877 | 1.212/1.147 | 1.133/1.348 | 36.0/22.9 |

## Cooperative 3D Object Detection Benchmark (50m/100m)
| Method | $\text{AP}_{Veh}\uparrow$ | $\text{AP}_{Ped}\uparrow$ | $\text{AP}_{Cyc}\uparrow$ | $mAP\uparrow$ | $mATE\downarrow$ | $mASE\downarrow$ | $mAOE\downarrow$ | $mAVE\downarrow$ | $NDS\uparrow$ |
|---------|--------------------------|--------------------------|--------------------------|----------------|-------------------|-------------------|-------------------|-------------------|--------------|
| No Fusion | 67.1/41.5 | 38.0/6.3 | 37.3/11.6 | 47.5/19.8 | 0.117/0.247 | 0.876/0.880 | 1.069/1.126 | 1.260/1.625 | 33.8/18.6 |
| F-Cooper | **75.4/52.8** | 50.1/9.1 | 44.7/20.4 | 56.8/27.4 | 0.117/0.205 | **0.874/0.879** | 1.074/1.206 | 1.358/1.449 | 38.5/22.9 |
| Raw-level Fusion | 71.3/48.9 | 38.1/8.5 | 40.7/16.3 | 50.0/24.6 | 0.135/0.242 | 0.875/0.882 | **1.062/1.242** | 1.308/1.469 | 34.9/21.1 |
| *VoxelNeXt | 71.5/50.6 | **60.1/35.4** | **47.6/21.9** | **59.7/35.9** | **0.085/0.159** | 0.877/0.878 | 1.070/1.204 | 1.262/1.463 | **40.2/27.6** |

* We use sparse convolution to fuse the VoxelNext features

## mAP Scores on 3D Object Detection using Different Scheduling Policies (50m/100m)

| Inference \ Training | No Fusion | Closest Agent | Single Random | Multiple Random | Full Communication |
|-----------------------|-----------|---------------|---------------|-----------------|--------------------|
| **No Fusion**         | 50.9/26.2  | 50.9/23.3     | 51.3/25.3     | 50.3/22.9       | 45.6/18.8          |
| **Closest Agent**     | 39.9/20.3  | 58.4/30.2     | 58.3/32.6     | 57.7/30.5       | **55.4**/10.8      |
| **Single Random**     | 43.3/22.8  | 57.9/31.0     | 58.4/33.3     | 57.7/31.4       | 55.0/14.6          |
| **MASS**        | **55.5**/11.0 | **58.8**/33.7 | **58.9**/34.0 | 57.3/32.3       | 54.1/27.4          |
| **Historical Best**   | 54.8/29.6  | 58.6/31.7     | **58.9**/34.0 | **58.3**/32.6   | 54.1/27.4          |
| **Multiple Random**   | **34.5**/16.9 | 60.7/35.1     | 61.2/37.1     | 61.4/36.4       | 58.8/12.9          |
| **Full Communication**| 29.1/10.5  | **63.7**/38.4  | **64.0**/39.9 | **64.7**/41.3   | **65.1**/39.2     |


