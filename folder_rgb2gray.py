import os
from os.path import basename
import cv2
import argparse
from tqdm import tqdm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="./test_images/old", help="Test images")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output",
    )
    opts = parser.parse_args()

    # resolve relative paths before changing directory
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)

    img_names = sorted([os.path.join(opts.input_folder, x)
                         for x in os.listdir(opts.input_folder) 
                         if x.endswith('.jpg') or x.endswith('.png')])

    for name in tqdm(img_names):
        
        img_bgr = cv2.imread(name)
        # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_fakebgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        # img_fakebgr = img_gray

        cv2.imwrite(os.path.join(opts.output_folder, os.path.basename(name)), img_fakebgr)