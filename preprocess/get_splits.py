import sys
sys.path.append('../../detecting_causes_of_gender_bias_chest_xrays')


from dataloader.dataloader import DISEASE_LABELS_CHE,DISEASE_LABELS_NIH,NIHDataResampleModule, CheXpertDataResampleModule
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore")




def sampling_ds(dataset:str,
    label:str,
    rs:int,
    npp:int,
    out_dir:str):

    assert dataset in ['NIH','chexpert'], 'Unknown dataset'
    if dataset == 'NIH':
        ChestXRayModule = NIHDataResampleModule
        csv_file_img ='../datafiles/'+'Data_Entry_2017_v2020_clean_split.csv'
    elif dataset == 'chexpert':
        ChestXRayModule = CheXpertDataResampleModule
        csv_file_img = '../datafiles/chexpert.sample.allrace.csv'

    # other hps

    #  for various the gender ratios:
    for f_per in [0,50,100]:
        print('.'*30+'female percentage: '+str(f_per))
        # mkdir
        output_dir = out_dir + f'/{dataset}-{label}-{npp}-{f_per}-{rs}/'
        if os.path.exists(output_dir) == False:
            os.mkdir(output_dir)


        data = ChestXRayModule(img_data_dir='/work3/ninwe/dataset/{}/preproc_224x224/'.format(dataset),
                                csv_file_img=csv_file_img,
                                image_size=224,
                                pseudo_rgb=False,
                                batch_size=64, # do not matter here
                                num_workers=0, # do not matter here
                                augmentation=False,# do not matter here
                                outdir=output_dir,
                                version_no=0, # do not matter here
                                female_perc_in_training=f_per,
                                chose_disease=label,
                                random_state=rs,
                                num_classes=1,
                                num_per_patient=npp,
                                prevalence_setting = 'separate',)


        del data








if __name__ == '__main__':
    print('Starting automatically create sampled splits ... ')


    out_dir = '../datafiles/sampling_splits/'
    if os.path.exists(out_dir) == False:
        os.mkdir(out_dir)


    parser = ArgumentParser()


    parser.add_argument('-s','--dataset',default='chexpert',help='Dataset', choices =['NIH','chexpert','all'])
    parser.add_argument('-d','--disease_label',default=['Cardiomegaly'], help='Chosen disease label', type=str, nargs='*')
    parser.add_argument('-r', '--random_state', default='0-10', help='random state')
    args = parser.parse_args()


    print('hyper-parameters:')
    print(args)


    if len(args.random_state.split('-')) != 2:
        if len(args.random_state.split('-')) == 1:
            rs_min, rs_max = int(args.random_state), int(args.random_state)+1
        else:
            raise Exception('Something wrong with args.random_states : {}'.format(args.random_states))
    else:
        rs_min, rs_max = int(args.random_state.split('-')[0]),int(args.random_state.split('-')[1])




    # disease_label_list = [''.join(each) for each in args.disease_label]
    # if len(disease_label_list) ==1 and disease_label_list[0] == 'all':
    # disease_label_list = DISEASE_LABELS_NIH if args.dataset == 'NIH' else DISEASE_LABELS_CHE
    disease_label_list = args.disease_label
    print('disease_label_list:{}'.format(disease_label_list))




    # NIH
    if args.dataset == 'NIH' or args.dataset == 'all':
        if len(disease_label_list) ==1 and disease_label_list[0] == 'all':
            loop_list = DISEASE_LABELS_NIH
        else:
            loop_list = disease_label_list
        
        for each_d in loop_list:
            print('^'*30+'Sampling {} - {}'.format('NIH',each_d))
            for i in tqdm(np.arange(rs_min, rs_max)):
                print('-'*30+'rs:{}'.format(i))
                sampling_ds(dataset='NIH',label=each_d,rs=i,npp=1,out_dir=out_dir)


    # chexpert
    if args.dataset == 'chexpert' or args.dataset == 'all':
        if len(disease_label_list) ==1 and disease_label_list[0] == 'all':
            loop_list = DISEASE_LABELS_CHE
        else:
            loop_list = disease_label_list
        
        for each_d in loop_list:
            print('^'*30+'Sampling {} - {}'.format('CheXpert',each_d))
            for i in tqdm(np.arange(rs_min, rs_max)):
                print('-'*30+'rs:{}'.format(i))
                sampling_ds(dataset='chexpert',label=each_d,rs=i,npp=1,out_dir=out_dir)

