#!/bin/bash

# 这是一个简单的测试脚本，用于验证prompt_tester.py工具的功能
# 它只测试一个类别的少量描述词，以快速验证工具是否正常工作

# CONFIG_FILE="configs/mm_grounding_dino/cross/dataset2/prompt.py"
# CONFIG_FILE="configs/mm_grounding_dino/upload/dataset1/prompt.py"
# WORK_DIR="output_prompt/prompt_test_dataset1"

# CONFIG_FILE="configs/mm_grounding_dino/upload_fixed_seed/dataset3/grounding_dino_swin-l_dataset3_1shot.py"
# WORK_DIR="output_prompt/prompt_test_dataset3_2"

# # 创建工作目录
# mkdir -p $WORK_DIR

# echo "开始测试prompt_tester.py工具..."
# echo "使用配置文件: $CONFIG_FILE"
# echo "结果将保存在: $WORK_DIR"

# # 测试car类别的两个不同描述词
# python tools/prompt_tester.py $CONFIG_FILE \
#     --class-names dent  \
#     --descriptions "Bump" "deformed"\
#     --max-combinations 3 \
#     --use-prefix \
#     --work-dir $WORK_DIR

# echo "测试完成，请查看 $WORK_DIR 目录中的结果" 


# CONFIG=$1
GPUS=$1
CONFIG_FILE="configs/mm_grounding_dino/upload_fixed_seed/dataset3/prompt_1shot.py"
# CHECKPOINT="output/upload/dataset3/1shot/best_coco_bbox_mAP_epoch_15.pth"
CHECKPOINT="output/upload/dataset3/10shot/best_coco_bbox_mAP_epoch_25.pth"
WORK_DIR="output_prompt/prompt_test_dataset3_shot10"
mkdir -p $WORK_DIR

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
echo "开始测试prompt_tester.py工具..."
echo "使用配置文件: $CONFIG_FILE"
echo "结果将保存在: $WORK_DIR"
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test_prompt.py \
    $CONFIG_FILE \
    $CHECKPOINT \
    --launcher pytorch \
    --class-names "dent" "scratch" "crack" "glass shatter" "lamp broken" "tire flat"  \
    --descriptions "deformed" "metal" "panel deformation" "surface"\
    --descriptions "abrasion" "surface abrasion" \
    --descriptions "Material Fissure" "Structural" \
    --descriptions "broken" "Fragmented"\
    --descriptions "Cracked" \
    --descriptions "Deflated" \
    --max-combinations 10000 \
    --use-prefix \
    --work-dir $WORK_DIR
    ${@:4}
