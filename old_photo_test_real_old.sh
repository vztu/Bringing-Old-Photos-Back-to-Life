# test images with scratch
input_folder="./data/real_old_resize/input_lll"
output_folder="./results/real_old_resize/input_lll"

python run.py --input_folder ${input_folder} \
               --output_folder ${output_folder} \
               --GPU -1 --with_scratch