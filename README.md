# WHALES
This is the official repository of WHALES.

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

comming soon
