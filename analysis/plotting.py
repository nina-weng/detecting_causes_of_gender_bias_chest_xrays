

#####
# re-store the prediction results and plot Figure 2 in main paper and Figure 2 in suppplementary materials
#####
import os.path

from dataloader.dataloader import DISEASE_LABELS_NIH, DISEASE_LABELS_CHE
from analysis.utils import get_gender_df, no_bs, load_demographic_data

from argparse import ArgumentParser
import numpy as np


import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.utils import resample
from tabulate import tabulate
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')


def plotting_all():
    result_dir = 'D:\\ninavv\\phd\\research\\run_results\\'
    font_size=25

    per_plot_each_row = 6
    z = math.ceil(len(disease_list)/per_plot_each_row)
    print(per_plot_each_row,z)
    nrows = int(len(datasets)*z)
    ncols = per_plot_each_row


    fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(ncols*4,nrows*4),
                            dpi=100)

    print(axes.shape)

    for i,each_d in enumerate(disease_list):
        for j,each_ds in enumerate(datasets):
            i_new = i %per_plot_each_row
            j_new = int(i//per_plot_each_row)*2+j


            print(j_new,i_new,each_d,each_ds)
            if j_new==7:
                axes[j_new][i_new].set_xlabel('%female in training',fontsize=10)
            else:
                axes[j_new][i_new].set_xlabel(None)


            axes[j_new][i_new].set_title('{}'.format(each_d),fontsize=font_size, weight='bold')

            axes[j_new][i_new].set_ylim(0.6,0.95)


            if i_new == 0:
                axes[j_new][i_new].set_ylabel('{}\nAUROC'.format(each_ds),fontsize=font_size)
            else:
                axes[j_new][i_new].set_ylabel(None)



            try:
                # normal settings (sample one with priority in diesease one)
                csv_file = result_dir+'{}_results'.format(each_ds)+'\\plotting\\'+'updated_{}.csv'.format(each_d)
                # sam prevalence between different groups
                #csv_file = result_dir+'{}_results'.format(each_ds)+'\\plotting\\'+'equal_updated_{}.csv'.format(each_d)
                df = pd.read_csv(csv_file)
                print(csv_file)
            except:
                continue

            df['TrainOnFemalePerc'] = df['exp_suffix'].apply(lambda x: 0
            if 'Train on Male' in x
            else 50
            if 'Train on Both' in x
            else 100)
            df['TestOn'] =  df['exp_suffix'].apply(lambda x: 'Test on Male'
            if 'Test on M' in x
            else 'Test on Female')
            df = df[df['rs']<10]
            print(df.shape)
            sns.boxplot(x="TrainOnFemalePerc", y="auroc", hue="TestOn",
                        hue_order=['Test on Male','Test on Female',],
                        palette=['blue','gold'],
                        data=df,
                        linewidth=1,
                        width=0.8,
                        ax=axes[j_new][i_new])
            #         sns.stripplot(x="TrainOnFemalePerc", y='auroc', data=df,ax=axes[j_new][i_new],
            #                      size=2,jitter=0.2,
            #                       hue="TestOn",
            #                      hue_order=['Test on Male','Test on Female',],
            #                      palette=['yellowgreen','red'],dodge=True)



            if j_new==0 and i_new == ncols-1:
                axes[j_new][i_new].legend(bbox_to_anchor=(1, 2.2),fontsize=15)
            else:
                axes[j_new][i_new].get_legend().remove()
            if j_new!=nrows-1:
                axes[j_new][i_new].set_xlabel(None)
            if i_new != 0:
                #axes[j_new][i_new].set_ticklabels([])
                axes[j_new][i_new].set_ylabel(None)
                plt.setp(axes[j_new][i_new].get_yticklabels(), visible=False)
                #axes[j_new][i_new].grid(True,axis='y')

            else:
                axes[j_new][i_new].set_ylabel('{}\nAUROC'.format(each_ds),fontsize=font_size)
            if j_new==nrows-1 and i_new==0:
                axes[j_new][i_new].set_xlabel('%female in training',fontsize=font_size)
            else:
                axes[j_new][i_new].set_xlabel(None)
            axes[j_new][i_new].tick_params(axis='both', labelsize=22)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.4)
    plt.tight_layout()
    plt.savefig('D:\\ninavv\\phd\\research\\projects\\p1_bias_reasoning\\all_per_common_labels_updated_cheds.png')





def re_store_results(disease_label_list,list_rs,female_perc_in_training_sets):
    for d in disease_label_list: #disease_labels:
        print('*'*30)
        print('D:{}'.format(d))
        result_varioustrain = []
        for f_per in female_perc_in_training_sets:
            print('-'*30)
            print('F_PERC:{}'.format(f_per))
            pred_df_list=[]
            for rs in tqdm(list_rs):
                run_config=f'{args.dataset}-{d}-fp{f_per}-npp{args.npp}-rs{rs}'
                version_no = 0
                print(run_config)

                pred_test = pd.read_csv(data_dir + run_config+'\\predictions.test.version_{}.csv'.format(version_no))
                # load demographic data according to dataset
                df_test = pd.read_csv(data_dir + run_config+'\\test.version_{}.csv'.format(version_no))
                df_test.reset_index(inplace = True)
                assert df_test.shape[0] == pred_test.shape[0],'df_test:{},preds:{}'.format(df_test.shape,pred_test.shape)

                pred_test = load_demographic_data(args,pred_test,df_test)
                pred_df_list.append(pred_test)

            # print(df_test.shape)
            all_roc_auc_nobs_df = no_bs(pred_df_list,d)
            gender_df = get_gender_df(all_roc_auc_nobs_df)

            gender_df_this_f_perc = gender_df.copy()
            if f_per == 100:
                trainon = 'Train on Female'
            elif f_per == 0:
                trainon = 'Train on Male'
            elif f_per == 50:
                trainon = 'Train on Both'
            gender_df_this_f_perc['exp_suffix'] = gender_df_this_f_perc['exp_suffix'].apply(lambda x: trainon+' '+x )
            result_varioustrain.append(gender_df_this_f_perc)

        gender_df_all = pd.concat([result_varioustrain[0],result_varioustrain[1],result_varioustrain[2]])
        # print(gender_df_all.shape)
        # print(gender_df_all)
        gender_df_all.to_csv(args.out_dir+'{}-{}.csv'.format(args.dataset,d))




def main(args,disease_label_list,list_rs):
    female_perc_in_training_sets = [0,50,100]

    ##############
    # re-store the prediction result in the form of auroc for different sensitive group in different gender ratios
    re_store_results(args,disease_label_list,list_rs,female_perc_in_training_sets)

    ################
    # plotting
    # a) plotting in a big picture
    plotting












if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s','--dataset',default='NIH',help='Dataset', choices =['NIH','chexpert'])
    parser.add_argument('-d','--disease_label',default='all', help='Chosen disease label', type=list, nargs='+')
    parser.add_argument('-n', '--npp',default=1,help='Number per patient, could be integer or None (no sampling)',type=int)
    parser.add_argument('-r', '--random_state', default='0-10', help='random state')
    args = parser.parse_args()
    print('args:')
    print(args)

    # interpret random states
    if len(args.random_state.split('-')) != 2:
        if len(args.random_state.split('-')) == 1:
            rs_min, rs_max = int(args.random_state), int(args.random_state)+1
        else:
            raise Exception('Something wrong with args.random_states : {}'.format(args.random_states))
    rs_min, rs_max = int(args.random_state.split('-')[0]),int(args.random_state.split('-')[1])
    list_rs = np.arange(rs_min,rs_max)

    # interpret disease labels
    disease_label_list = [''.join(each) for each in args.disease_label]

    if len(disease_label_list) ==1 and disease_label_list[0] == 'all':
        disease_label_list = DISEASE_LABELS_CHE if args.dataset == 'chexpert' else DISEASE_LABELS_NIH

    if args.dataset == 'NIH':
        args.male='M'
        args.female='F'
    elif args.dataset == 'chexpert':
        args.male='Male'
        args.female='Female'


    # define the path of run result files
    data_dir = '/work3/ninwe/run/cause_bias/' # where to read the run results
    out_dir = '/work3/ninwe/run/cause_bias/plotting/'
    if os.path.exists(out_dir) == False:
        os.mkdir(out_dir)

    args.data_dir = data_dir
    args.out_dir = out_dir

    #
    main(args,disease_label_list,list_rs)


