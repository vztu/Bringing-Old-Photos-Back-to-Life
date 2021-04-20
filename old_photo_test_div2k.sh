
input_folder="/home/ztu/Desktop/InstColorization/train_data/DIV2K_resize_val_HR"
# input_folder="/home/ztu/Desktop/InstColorization/train_data/pascal_clean_more_val/val"
input_folder_gray='./data/DIV2K_resize_val_HR'


## First convert RGB to YYY images
python folder_rgb2gray.py --input_folder ${input_folder} \
                       --output_folder ${input_folder_gray}


## Run test
output_folder="./results/DIV2K_resize_val_HR"
python run.py --input_folder ${input_folder_gray} \
              --output_folder ${output_folder} \
              --GPU 0,1