import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch

from dataloader.dataloader import CheXpertDataResampleModule, NIHDataResampleModule
from prediction.disease_prediction import get_cur_version, hp_default_value, test_func
from prediction.models import ResNet


def load_model(ckpt_dir):
    model_choose = hp_default_value.model
    num_classes = hp_default_value.num_classes
    lr = hp_default_value.lr
    pretrained = disease_label_list.pretrained
    model_scale = hp_default_value.model_scale

    if model_choose == "resnet":
        model_type = ResNet

    file_list = os.listdir(ckpt_dir)
    assert len(file_list) == 1
    ckpt_path = ckpt_dir + file_list[0]
    model = model_type.load_from_checkpoint(
        ckpt_path,
        num_classes=num_classes,
        lr=lr,
        pretrained=pretrained,
        model_scale=model_scale,
    )

    return model


def main(args, d, rs, f_perc):
    run_config_NIH = f"NIH-{d}-fp{f_perc}-npp{args.npp}-rs{rs}"
    run_config_chexpert = f"chexpert-{d}-fp{f_perc}-npp{args.npp}-rs{rs}"
    version_no_NIH = 0
    version_no_chexpert = 0

    run_dir = args.run_dir

    # load the models
    print("Loading the models ...")

    ckpt_dir_NIH = run_dir + run_config_NIH + f"/version_{version_no_NIH}/checkpoints/"
    NIH_model = load_model(ckpt_dir_NIH)

    ckpt_dir_chexpert = (
        run_dir + run_config_chexpert + f"/version_{version_no_chexpert}/checkpoints/"
    )
    chexpert_model = load_model(ckpt_dir_chexpert)
    print("Loading the models done.")

    # load the dataset
    print("Loading the NIH dataset ...")
    img_data_dir = args.data_dir + "NIH/preproc_224x224/"
    csv_file_img = "../datafiles/Data_Entry_2017_v2020_clean_split.csv"
    image_size = args.image_size
    bs = args.bs
    num_workers = args.number_workers
    augmentation = args.augmentation

    chose_disease_str = d

    # output run config
    run_dir = args.run_dir
    run_config = f"cross_ds_{d}_fpec{f_perc}_rs{rs}"
    out_dir = run_dir + run_config
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_dir_NIH = out_dir + "/NIH_datasplit/"
    if not os.path.exists(out_dir_NIH):
        os.makedirs(out_dir_NIH)

    cur_version = get_cur_version(out_dir)

    data_NIH = NIHDataResampleModule(
        img_data_dir=img_data_dir,
        csv_file_img=csv_file_img,
        image_size=image_size,
        pseudo_rgb=False,
        batch_size=bs,
        num_workers=num_workers,
        augmentation=augmentation,
        outdir=out_dir_NIH,
        version_no=cur_version,
        female_perc_in_training=f_perc,
        chose_disease=chose_disease_str,
        random_state=rs,
        num_classes=hp_default_value.num_classes,
        num_per_patient=hp_default_value.npp,
        crop=hp_default_value.crop,
        prevalence_setting=hp_default_value.prevalence_setting,
    )

    print("Loading the NIH dataset done.")

    print("Loading the chexpert dataset ...")
    img_data_dir = args.data_dir + "chexpert/preproc_224x224/"
    csv_file_img = "../datafiles/chexpert.sample.allrace.csv"
    out_dir_chexpert = out_dir + "/chexpert_datasplit/"
    if not os.path.exists(out_dir_chexpert):
        os.makedirs(out_dir_chexpert)

    cur_version = get_cur_version(out_dir)

    data_chexpert = CheXpertDataResampleModule(
        img_data_dir=img_data_dir,
        csv_file_img=csv_file_img,
        image_size=image_size,
        pseudo_rgb=False,
        batch_size=bs,
        num_workers=num_workers,
        augmentation=augmentation,
        outdir=out_dir_chexpert,
        version_no=cur_version,
        female_perc_in_training=f_perc,
        chose_disease=chose_disease_str,
        random_state=rs,
        num_classes=hp_default_value.num_classes,
        num_per_patient=hp_default_value.npp,
        prevalence_setting=hp_default_value.prevalence_setting,
    )
    print("Loading the chexpert dataset done")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(0) if use_cuda else "cpu")

    NIH_model.to(device)
    chexpert_model.to(device)

    num_classes = hp_default_value.num_classes

    cols_names_classes = ["class_" + str(i) for i in range(0, num_classes)]
    cols_names_logits = ["logit_" + str(i) for i in range(0, num_classes)]
    cols_names_targets = ["target_" + str(i) for i in range(0, num_classes)]

    print("1.TESTING NIH test set Using NIH model")
    preds_test, targets_test, logits_test = test_func(
        NIH_model, data_NIH.test_dataloader(), device
    )
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(
        os.path.join(
            out_dir, "predictions.testNIHtrainNIH.version_{}.csv".format(cur_version)
        ),
        index=False,
    )

    print("2.TESTING NIH test set Using chexpert model")
    preds_test, targets_test, logits_test = test_func(
        chexpert_model, data_NIH.test_dataloader(), device
    )
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(
        os.path.join(
            out_dir,
            "predictions.testNIHtrainChexpert.version_{}.csv".format(cur_version),
        ),
        index=False,
    )

    print("3.TESTING chexpert test set Using chexpert model")
    preds_test, targets_test, logits_test = test_func(
        chexpert_model, data_chexpert.test_dataloader(), device
    )
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(
        os.path.join(
            out_dir,
            "predictions.testChexperttrainChexpert.version_{}.csv".format(cur_version),
        ),
        index=False,
    )

    print("4.TESTING chexpert test set Using NIH model")
    preds_test, targets_test, logits_test = test_func(
        NIH_model, data_chexpert.test_dataloader(), device
    )
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(
        os.path.join(
            out_dir,
            "predictions.testChexperttrainNIH.version_{}.csv".format(cur_version),
        ),
        index=False,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    # the results from which datasets that we'd like to re-store to csv files
    parser.add_argument(
        "-d",
        "--disease_label",
        default="Pneumothorax",
        help="Chosen disease label",
        type=list,
        nargs="+",
    )
    parser.add_argument(
        "-n",
        "--npp",
        default=1,
        help="Number per patient, could be integer or None (no sampling)",
        type=int,
    )
    parser.add_argument("-r", "--random_state", default="0", help="random state")
    parser.add_argument(
        "-p",
        "--run_dir",
        default="/work3/ninwe/run/cause_bias/",
        help="your run dir",
        type=str,
    )
    parser.add_argument(
        "-q",
        "--data_dir",
        default="/work3/ninwe/dataset/",
        help="your data dir",
        type=str,
    )
    args = parser.parse_args()
    print("args:")
    print(args)

    # interpret random states
    if len(args.random_state.split("-")) != 2:
        if len(args.random_state.split("-")) == 1:
            rs_min, rs_max = int(args.random_state), int(args.random_state) + 1
        else:
            raise Exception(
                "Something wrong with args.random_states : {}".format(
                    args.random_states
                )
            )
    rs_min, rs_max = int(args.random_state.split("-")[0]), int(
        args.random_state.split("-")[1]
    )
    list_rs = np.arange(rs_min, rs_max)

    # interpret disease labels
    disease_label_list = ["".join(each) for each in args.disease_label]

    for d in disease_label_list:
        for rs in list_rs:
            for f_perc in [0, 50, 100]:
                main(args, d, rs, f_perc)
