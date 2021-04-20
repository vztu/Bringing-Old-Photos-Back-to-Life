
# input_folder="./data/pascal_val_crack"
# # input_folder="/home/ztu/Desktop/InstColorization/train_data/pascal_clean_more_val/val"
# input_folder_gray='./data/pascal_val_crack_yyy'


# ## First convert RGB to YYY images
# python folder_y2yyy.py --input_folder ${input_folder} \
#                        --output_folder ${input_folder_gray}


# ## Run test
# output_folder="./results/pascal_val_crack"
# python run.py --input_folder ${input_folder_gray} \
#               --output_folder ${output_folder} \
#               --GPU 0


## Eval metrics
output_folder="./results/pascal_val_crack/final_output"
gt_folder="/home/ztu/Desktop/InstColorization/train_data/pascal_clean_more_val/val"
python folder_eval_metrics.py --output_folder ${output_folder} \
              --gt_folder ${gt_folder}