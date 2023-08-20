import os
import sys
sys.path.append('../../NIH_classifer')
import torch
import pandas as pd

from prediction.models import ResNet,DenseNet
from dataloader.dataloader import NIHDataset,NIHDataModule,DISEASE_LABELS
from prediction.disease_prediction import test_func,embeddings

if __name__ == '__main__':

    run_folder = '/work3/ninwe/run/NIH/disease/'
    run_config = 'resnet18-lr1e-05-ep20-pt1-aug1-GIsplit-imgs224'
    version_no = 0
    ckpts = os.listdir(run_folder+run_config+f'/version_{version_no}/checkpoints')
    assert  len(ckpts) == 1

    ckpt_path = run_folder+run_config+f'/version_{version_no}/checkpoints/'+ckpts[0]

    # get info from run_config
    model_name = run_config.split('-')[0]
    if 'resnet' in model_name:
        model_type = ResNet

    lr = 1e-5
    pretrained = True
    model_scale='18'
    image_size = int(run_config.split('-')[-1][4:])
    if image_size == 224:
        img_data_dir = '/work3/ninwe/dataset/NIH/preproc_224x224/'
    elif image_size == 1024:
        img_data_dir = '/work3/ninwe/dataset/NIH/images/'

    num_classes = len(DISEASE_LABELS)


    model = model_type.load_from_checkpoint(ckpt_path,
                                            num_classes=num_classes, lr=lr, pretrained=pretrained,
                                            model_scale=model_scale,
                                            )

    csv_file_img = '../datafiles/' + 'Data_Entry_2017_v2020_clean_split.csv'
    batch_size=64
    num_workers=2
    augmentation=True
    view_position='AP'
    vp_sample=False
    only_gender=None
    save_split=True
    out_dir = '/work3/ninwe/run/NIH/disease/' + run_config +'/orisplit/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cur_version=version_no
    gi_split=False
    fold_num=0

    # define the test set

    data = NIHDataModule(img_data_dir=img_data_dir,
                         csv_file_img=csv_file_img,
                         image_size=image_size,
                         pseudo_rgb=False,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         augmentation=augmentation,
                         view_position=view_position,
                         vp_sample=vp_sample,
                         only_gender=only_gender,
                         save_split=save_split,
                         outdir=out_dir,
                         version_no=cur_version,
                         gi_split=gi_split,
                         fold_num=fold_num)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(0) if use_cuda else "cpu")

    model.to(device)

    cols_names_classes = ['class_' + str(i) for i in range(0, num_classes)]
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

    print('EMBEDDINGS')

    model.remove_head()

    embeds_val, targets_val = embeddings(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=embeds_val)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'embeddings.val.version_{}.csv'.format(cur_version)), index=False)

    embeds_test, targets_test = embeddings(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=embeds_test)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'embeddings.test.version_{}.csv'.format(cur_version)), index=False)