# NTIRE 2025 CD-FSOD Challenge-HUSTLab-14

## Setup
1. Clone this repository.

```bash
git clone https://github.com/Weston111/cdfsod.git && cd cdfsod/
```

2. Install the required dependencies.

```bash
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
pip install -v -e .
pip install -r requirements.txt
```

具体环境：

* for pytorch,we use version 2.0.0+cu118 for torch and 0.15.1+cu118 for torchvision
* for cuda, we use cuda11.8

3. Download pre-trained model
    * Model1 MM-Grounding-DINO https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth and https://huggingface.co/fushh7/LLMDet/blob/main/large.pth .
    for later one ,we use for its some module trained from llmdet. part of mmgroundingdino ,you need to download [bert](https://huggingface.co/bert-base-uncased) (We separate it just to save storage)(bert是mmgroundingdino的一个模块，分开来存放可以节省空间)
    * Qwen2.5VL we just use it through api.
```bash
mkdir model
cd model
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth

pip install -U huggingface_hub
huggingface-cli download --resume-download fushh7/LLMDet --local-dir .
mv LLMDet/large.pth ./
huggingface-cli download --resume-download google-bert/bert-base-uncased --local-dir .
```



## Main Results
1. you need to download the trained model by [link](https://pan.baidu.com/s/1yNgvUq6iD_kE_OkIiYvP5Q?pwd=4aym)

2. 然后放到文件夹checkpoints下


```bash
bash main_results_test.sh
```

```
｜--cdfsod
｜  |--mmdet and others folders
｜  |--checkpoints
｜  |  |--dataset1_{k}shot.pth
｜  |  |--dataset2_{k}shot.pth
｜  |  |--dataset3_{k}shot.pth
｜  |--data
｜  |  |--cdfsod
｜  |  |  |--dataset1
｜  |  |  |--dataset2
｜  |  |  |--dataset3
｜  |--configs
｜  |  |--mm_grounding_dino
｜  |  |  |--cdfsod
｜  |  |  |  |--dataset1
｜  |  |  |  |  |--dataset1_{k}shot.py
｜  |  |  |  |--dataset2
｜  |  |  |  |  |--dataset2_{k}shot.py
｜  |  |  |  |--dataset3
｜  |  |  |  |  |--dataset3_{k}shot.py
｜  |--model
｜  |  |--grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
｜  |  |--large.pth
｜  |--test_results
｜  |  |--dataset{n}_{k}shot.json
｜  |--main_results_test.sh
｜  |--main_results_train.sh
```