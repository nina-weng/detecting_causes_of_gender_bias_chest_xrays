import sys
sys.path.append('../../detecting_causes_of_gender_bias_chest_xrays')

from dataloader.dataloader import DISEASE_LABELS_NIH,NIHDataResampleModule
from prediction.models import ResNet,DenseNet


import os
import torch
import pandas as pd
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser
import shutil
import numpy as np

# test

num_classes = len(DISEASE_LABELS)
image_size = (1024, 1024)
image_size = (224, 224)

# parameters that could change
if image_size[0] == 224:
    batch_size = 64
elif image_size[0] == 1024:
    batch_size = 16
else:
    raise Exception('wrong image size')

epochs = 20
num_workers = 2 ###
model_choose = 'resnet' # or 'densenet'
model_scale = '50'
lr=1e-6
pretrained = True
augmentation = True

gi_split=True
# gender_setting='100%_female'  # '0%_female', '100%_female'
fold_nums=np.arange(0,20)

resam=True
female_perc_in_training_set = [0,50,100]#
random_state_set = np.arange(0,10)
num_per_patient = 1
disease_list=['Cardiomegaly']
# chose_disease_str =  'Effusion' #'Pneumonia','Pneumothorax','Cardiomegaly'
random_state = 2022
if resam: num_classes = 1
# print('multi label training')
# num_classes = len(DISEASE_LABELS)
isMultilabel = True if num_classes!=1 else False
crop = None # None or [0,1]
prevalence_setting = 'total' #['separate','total','equal']

save_model_para = False
loss_func_type='BCE'


if image_size[0] == 224:
    img_data_dir = '/work3/ninwe/dataset/NIH/preproc_224x224/'
elif image_size[0] == 1024:
    img_data_dir = '/work3/ninwe/dataset/NIH/images/'


csv_file_img = '../datafiles/'+'Data_Entry_2017_v2020_clean_split.csv'
#csv_file_img = 'D:/ninavv/phd/data/NIH/'+'Data_Entry_2017_v2020_clean_split_fake.csv'


def get_cur_version(dir_path):
    i = 0
    while os.path.exists(dir_path+'/version_{}'.format(i)):
        i+=1
    return i


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def test_func(model, data_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0,num_classes):
            t = targets[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()

def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            emb = model(img)
            embeds.append(emb)
            targets.append(lab)

        embeds = torch.cat(embeds, dim=0)
        targets = torch.cat(targets, dim=0)

    return embeds.cpu().numpy(), targets.cpu().numpy()




def main(hparams,gender_setting=None,fold_num=None,female_perc_in_training=None,random_state=None,chose_disease_str=None):



    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")
    print('DEVICE:{}'.format(device))

    # get run_config
    if not gi_split:
        view_position = 'AP'  # 'AP','PA','all'
        vp_sample = False
        if vp_sample and view_position != 'all':
            raise Exception('Could not sample the train set anymore for VP=AP/PA!')
        only_gender = None  # 'F' , 'M', None

        run_config = '{}{}-lr{}-ep{}-pt{}-aug{}-VP{}-sam{}-SEX{}-imgs{}-mpara{}'.format(model_choose, model_scale, lr, epochs,
                                                                                int(pretrained),
                                                                                int(augmentation), str(view_position),
                                                                                int(vp_sample),
                                                                                str(only_gender),
                                                                                image_size[0],
                                                                                int(save_model_para))
    else:
        view_position = 'all'  # 'AP','PA','all'
        vp_sample = False
        only_gender = None  # 'F' , 'M', None
        gender_setting = '{}%_female'.format(female_perc_in_training)
        run_config = '{}{}-lr{}-ep{}-pt{}-aug{}-VP{}-GIsplit-{}-Fold{}-imgs{}-mpara{}'.format(model_choose, model_scale, lr,
                                                                                   epochs,
                                                                                   int(pretrained),
                                                                                   int(augmentation), view_position,
                                                                                   gender_setting,
                                                                                   fold_num,
                                                                                   image_size[0],
                                                                                   int(save_model_para))

    if resam:
        if loss_func_type == 'BCE':
            run_config = '{}{}-lr{}-ep{}-pt{}-aug{}-{}%female-D{}-npp{}-ml{}-rs{}-imgs{}-crop{}-ps{}-mpara{}'.format(model_choose, model_scale, lr,
                                                                                      epochs,
                                                                                      int(pretrained),
                                                                                      int(augmentation),
                                                                                      female_perc_in_training,
                                                                                      chose_disease_str,
                                                                                      num_per_patient,
                                                                                      int(isMultilabel),
                                                                                      random_state,
                                                                                      image_size[0],
                                                                                      crop,
                                                                                      prevalence_setting,
                                                                                      int(save_model_para))
        else:
            run_config = '{}{}-lr{}-ep{}-pt{}-aug{}-{}%female-D{}-npp{}-ml{}-rs{}-loss{}-imgs{}-crop{}-ps{}-mpara{}'.format(model_choose, model_scale, lr,
                                                                                      epochs,
                                                                                      int(pretrained),
                                                                                      int(augmentation),
                                                                                      female_perc_in_training,
                                                                                      chose_disease_str,
                                                                                      num_per_patient,
                                                                                      int(isMultilabel),
                                                                                      random_state,
                                                                                      loss_func_type,
                                                                                      image_size[0],
                                                                                      crop,
                                                                                      prevalence_setting,
                                                                                      int(save_model_para))

    print('------------------------------------------\n'*3)
    print(run_config)
    # Create output directory
    # out_name = str(model.model_name)
    run_dir = '/work3/ninwe/run/NIH/disease/'
    out_dir = run_dir + run_config
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cur_version = get_cur_version(out_dir)




    # data
    if resam != True:
        data = NIHDataModule(img_data_dir=img_data_dir,
                                csv_file_img=csv_file_img,
                                image_size=image_size,
                                pseudo_rgb=False,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                augmentation=augmentation,
                                view_position = view_position,
                                vp_sample = vp_sample,
                                only_gender=only_gender,
                             save_split=True,
                             outdir=out_dir,
                             version_no=cur_version,
                             gi_split=gi_split,
                             gender_setting=gender_setting,
                             fold_num=fold_num)
    else:
        data = NIHDataResampleModule(img_data_dir=img_data_dir,
                                csv_file_img=csv_file_img,
                                image_size=image_size,
                                pseudo_rgb=False,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                augmentation=augmentation,
                                outdir=out_dir,
                                version_no=cur_version,
                                female_perc_in_training=female_perc_in_training,
                                chose_disease=chose_disease_str,
                                random_state=random_state,
                                num_classes=num_classes,
                                num_per_patient=num_per_patient,
                                crop=crop,
                                prevalence_setting = prevalence_setting,

            )

    if data.df_train is None:
        # end this run
        return


    # model
    if model_choose == 'resnet':
        model_type = ResNet
    elif model_choose == 'densenet':
        model_type = DenseNet
    model = model_type(num_classes=num_classes,lr=lr,pretrained=pretrained,model_scale=model_scale,
                       loss_func_type = loss_func_type)


    temp_dir = os.path.join(out_dir, 'temp_version_{}'.format(cur_version))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0,5):
        if augmentation:
            sample = data.train_set.exam_augmentation(idx)
            sample = np.asarray(sample)
            sample = np.transpose(sample, (2, 1, 0))
            imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.png'), sample)
        else:
            sample = data.train_set.get_sample(idx) #PIL
            sample = np.asarray(sample['image'])
            imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.png'), sample.astype(np.uint8))

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')



    # train
    trainer = pl.Trainer(
        # callbacks=[checkpoint_callback],
        callbacks=[EarlyStopping(monitor="val_loss", mode="min",patience=3)],
        log_every_n_steps = 1,
        max_epochs=epochs,
        gpus=hparams.gpus,
        accelerator="auto",
        logger=TensorBoardLogger(run_dir, name=run_config,version=cur_version),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                            num_classes=num_classes,lr=lr,pretrained=pretrained,
                                            model_scale=model_scale,
                                            loss_func_type=loss_func_type
                                            )

    model.to(device)

    cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
    cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
    cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

    print('VALIDATION')
    preds_val, targets_val, logits_val = test_func(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=preds_val, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.val.version_{}.csv'.format(cur_version)), index=False)

    print('TESTING')
    preds_test, targets_test, logits_test = test_func(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.test.version_{}.csv'.format(cur_version)), index=False)


    if (True and resam):
        print('TESTING on train set')
        # trainloader need to be non shuffled!
        preds_test, targets_test, logits_test = test_func(model, data.train_dataloader_nonshuffle(), device)
        df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
        df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
        df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
        df = pd.concat([df, df_logits, df_targets], axis=1)
        df.to_csv(os.path.join(out_dir, 'predictions.train.version_{}.csv'.format(cur_version)), index=False)

    # print('EMBEDDINGS')
    #
    # model.remove_head()
    #
    # embeds_val, targets_val = embeddings(model, data.val_dataloader(), device)
    # df = pd.DataFrame(data=embeds_val)
    # df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    # df = pd.concat([df, df_targets], axis=1)
    # df.to_csv(os.path.join(out_dir, 'embeddings.val.version_{}.csv'.format(cur_version)), index=False)
    #
    # embeds_test, targets_test = embeddings(model, data.test_dataloader(), device)
    # df = pd.DataFrame(data=embeds_test)
    # df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    # df = pd.concat([df, df_targets], axis=1)
    # df.to_csv(os.path.join(out_dir, 'embeddings.test.version_{}.csv'.format(cur_version)), index=False)

    # delete the model parameters

    if save_model_para == False:
        model_para_dir = os.path.join(out_dir,'version_{}'.format(cur_version))
        shutil.rmtree(model_para_dir)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)

    parser.add_argument('-ds','--dataset',default='NIH',help='Dataset, choose from [\'NIH\',\'chexpert\']')
    parser.add_argument('-d','--disease_choose',default='Pneumothorax', help='chosen disease label')

    args = parser.parse_args()



    if resam:
        print('***********RESAMPLING EXPERIMENT**********\n'*5)
        for d in disease_list:
            for female_perc_in_training in female_perc_in_training_set:
                for i in random_state_set:
                    main(args, female_perc_in_training=female_perc_in_training,random_state = i,chose_disease_str=d)
    elif gi_split:
        print('***********GI SPLIT EXPERIMENT**********\n' * 5)
        for f_perc in female_perc_in_training_set:
            for fold_i in fold_nums:
                main(args, female_perc_in_training=f_perc, fold_num=fold_i)