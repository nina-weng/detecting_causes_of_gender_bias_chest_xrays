import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from skimage.io import imread, imsave
from skimage.transform import resize
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--dataset",
        default="both",
        help="Dataset",
        choices=["NIH", "chexpert", "both"],
    )
    parser.add_argument(
        "-p", "--img_dir", help="the folder path of the images", type=str
    )
    args = parser.parse_args()
    print("args:")
    print(args)

    # preprocess NIH
    if args.dataset == "both" or args.dataset == "NIH":
        print("start to pre-process NIH ...")
        img_data_dir = args.img_dir + "NIH/"
        mata_csv_file = "./datafiles/Data_Entry_2017_v2020_clean_split.csv"

        df = pd.read_csv(mata_csv_file, header=0)

        preproc_dir = "preproc_224x224/"
        out_dir = img_data_dir + preproc_dir

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for idx, file_name in enumerate(tqdm(df["Image Index"])):
            out_path = out_dir + file_name

            if not os.path.exists(out_path):
                image = imread(img_data_dir + "/images/" + file_name)
                image = resize(image, output_shape=(224, 224), preserve_range=True)
                imsave(out_path, image.astype(np.uint8))

        print("pre-process NIH done.")

    # preprocess CheXpert
    if args.dataset == "both" or args.dataset == "chexpert":
        print("start to pre-process CheXpert ...")

        df_file = "./datafiles/chexpert.sample.allrace.csv"
        df = pd.read_csv(df_file)

        img_data_dir = args.img_dir + "chexpert/"
        preproc_dir = "preproc_224x224/"
        out_dir = img_data_dir + preproc_dir

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for idx, p in enumerate(tqdm(df["Path"])):
            split = p.split("/")
            preproc_filename = split[2] + "_" + split[3] + "_" + split[4]

            out_path = out_dir + preproc_filename

            if not os.path.exists(out_path):
                image = imread(img_data_dir + p)
                image = resize(image, output_shape=(224, 224), preserve_range=True)
                imsave(out_path, image.astype(np.uint8))

        print("pre-process CheXpert done.")
