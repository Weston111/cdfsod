bash tools/dist_test.sh configs/mm_grounding_dino/coco/grounding_dino_swin-t_finetune_16xb4_1x_coco.py  output/ifsod/coco_base/epoch_1.pth 4 --work-dir ./output/ifsod/test/stage1

bash tools/dist_test.sh configs/mm_grounding_dino/ifsod/coco_novel_test.py  output/ifsod/coco_base/epoch_1.pth 4 --work-dir ./output/ifsod/test/stage1

bash tools/dist_test.sh configs/mm_grounding_dino/ifsod/coco_novel_test.py  output/ifsod/coco_base/epoch_1.pth 4 --work-dir ./output/ifsod/test/stage1