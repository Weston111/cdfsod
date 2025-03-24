# NTIRE 2025 CD-FSOD挑战赛-HUSTLab-14

## 环境配置
1. 克隆此存储库。

```bash
git clone https://github.com/Weston111/cdfsod.git && cd cdfsod/
```

2. 安装所需的依赖项。

```bash
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"
pip install -v -e .
pip install -r requirements.txt
```

具体环境：

* 对于pytorch，我们使用版本2.0.0+cu118的torch和0.15.1+cu118的torchvision。
* 对于cuda，我们使用cuda11.8。

3. 下载预训练模型
    * 模型1 MM-Grounding-DINO https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth 和 https://huggingface.co/fushh7/LLMDet/blob/main/large.pth。对于后者，我们使用它是因为它的某些模块来自llmdet。mmgroundingdino的一部分，你需要下载[bert](https://huggingface.co/bert-base-uncased)（我们分开存放只是为了节省存储空间）(bert是mmgroundingdino的一个模块，分开来存放可以节省空间)。
    * Qwen2.5VL我们仅通过api使用它。


## 主要结果
1. 通过[百度网盘链接](https://pan.baidu.com/s/1yNgvUq6iD_kE_OkIiYvP5Q?pwd=4aym)下载训练好的模型。

2. 然后将其放入checkpoints文件夹中。
```
｜--cdfsod
｜  |--mmdet和其他文件夹
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
3. 执行main_results_tesh.sh脚本复现结果。
```bash
bash main_results_test.sh
```