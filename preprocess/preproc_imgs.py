import pandas as pd
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import resize
import os
from tqdm import tqdm
import numpy as np


if __name__ == '__main__':

    df_file = '../datafiles/chexpert/train.csv'
    df = pd.read_csv(df_file)

    img_data_dir = '/work3/ninwe/dataset/'
    preproc_dir = 'preproc_224x224/'
    out_dir = img_data_dir + preproc_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for idx, p in enumerate(tqdm(df['Path'])):
        split = p.split("/")
        preproc_filename = split[2] + '_' + split[3] + '_' + split[4]

        out_path = out_dir + preproc_filename

        if not os.path.exists(out_path):
            print('*'*30)
            print(out_path)
            print('*' * 30)
        else:
            image = imread(out_path)
            if image.shape[0] != 224 or image.shape[1] != 224:
                print('-'*30)
                print(out_path,image.shape)
                print('-' * 30)
