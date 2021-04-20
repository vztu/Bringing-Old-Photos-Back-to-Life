# test images without scratch
input_folder="./test_images/old"
output_folder="./results/test_images/old"
python run.py --input_folder ${input_folder} \
              --output_folder ${output_folder} \
              --GPU 0

# test images with scratch
input_folder="./test_images/old_w_scratch"
output_folder="./results/test_images/old_w_scratch"

python run.py --input_folder ${input_folder} \
               --output_folder ${output_folder} \
               --GPU 0 --with_scratch