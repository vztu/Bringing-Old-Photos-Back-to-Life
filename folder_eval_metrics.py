import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from os.path import basename
import cv2
import argparse
import numpy as np
from torchvision.transforms import functional as TF
from tqdm import tqdm
from utils import ssim, loss, color_space_convert

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./results/real_old_resize/input_lll/final_output")
    parser.add_argument(
        "--gt_folder",
        type=str,
        default="./data/real_old_resize/gt")
    opts = parser.parse_args()

    # resolve relative paths before changing directory
    opts.output_folder = os.path.abspath(opts.output_folder)
    opts.gt_folder = os.path.abspath(opts.gt_folder)
    # if not os.path.exists(opts.output_folder):
    #     os.makedirs(opts.output_folder)

    img_names = sorted([x for x in os.listdir(opts.output_folder) 
                         if x.endswith('.jpg') or x.endswith('.png')])

    count_empty = 0
    time_test = 0
    ssim_acc_rgb = 0.
    psnr_acc_rgb = 0.
    ssim_acc_gray = 0.
    psnr_acc_gray = 0.
    lpips_acc = 0.

    # use SSIM and PSNR to evluate image
    ssim_metric = ssim.SSIM().cuda()
    lpips_metric = loss.lipis_eval('alex').cuda()
    hsv = color_space_convert.RgbToHsv(training=False)
    dataset_size = len(img_names)
    for name in tqdm(img_names):
        # for DIV2k pascal
        # gt = cv2.imread(os.path.join(opts.gt_folder, os.path.splitext(name)[0]+'.jpg'))
        # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        # for real old
        try:
            gt = cv2.imread(os.path.join(opts.gt_folder, os.path.splitext(name)[0].replace('o','t')+'.jpg'))
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        except:
            gt = cv2.imread(os.path.join(opts.gt_folder, os.path.splitext(name)[0].replace('o','t')+'.png'))
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt_image = TF.to_tensor(gt).unsqueeze_(0).cuda()
        
        img = cv2.resize(cv2.imread(os.path.join(opts.output_folder, name)),
                         (gt.shape[1], gt.shape[0]) ,interpolation=cv2.INTER_CUBIC)
        rgb_image = TF.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze_(0).cuda()

        ssim_acc_rgb += ssim_metric(rgb_image, gt_image)
        ssim_acc_gray += ssim_metric(hsv(rgb_image), hsv(gt_image))

        psnr_acc_rgb += loss.batch_psnr(rgb_image, gt_image, 1.)
        psnr_acc_gray += loss.batch_psnr(hsv(rgb_image), hsv(gt_image), 1.)

        lpips_acc += lpips_metric(rgb_image, gt_image)

        
    with open(os.path.join(opts.output_folder, 'log.txt'), 'w') as f:
        print('The average RGB SSIM is %.3f,  RGB PSNR is %.3f, '
        'The average GRAY SSIM is %.3f,  GRAY PSNR is %.3f,'
        'lpips is %.3f'
        % (ssim_acc_rgb / dataset_size, psnr_acc_rgb / dataset_size,
            ssim_acc_gray / dataset_size, psnr_acc_gray / dataset_size,
            lpips_acc / dataset_size),  file=f)
