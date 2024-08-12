import sys

sys.path.append("../../detecting_causes_of_gender_bias_chest_xrays")

import os
import shutil
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from skimage.io import imsave
from tqdm import tqdm

from dataloader.dataloader import (
    DISEASE_LABELS_CHE,
    DISEASE_LABELS_NIH,
    CheXpertDataResampleModule,
    NIHDataResampleModule,
)
from prediction.models import DenseNet, ResNet

hp_default_value = {
    "model": "resnet",
    "model_scale": "50",
    "lr": 1e-6,
    "bs": 64,
    "epochs": 20,
    "pretrained": True,
    "augmentation": True,
    "is_multilabel": False,
    "image_size": (224, 224),
    "crop": None,
    "prevalence_setting": "separate",
    "save_model": False,
    "num_workers": 2,
    "num_classes": 1,
    "embedding_size": None,
    "csv_file_img": None,
}


def nullable_string(val):
    if not val:
        return None
    return str(val)


def get_cur_version(dir_path):
    i = 0
    while os.path.exists(dir_path + "/version_{}".format(i)):
        i += 1
    return i


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def test_func(args, model, data_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc="Test-loop")):
            img, lab = batch["image"].to(device), batch["label"].to(device)
            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0, args.num_classes):
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
        for batch in tqdm(data_loader, desc="Test-loop"):
            img, lab = batch["image"].to(device), batch["label"].to(device)
            emb = model(img)
            embeds.append(emb)
            targets.append(lab)

        embeds = torch.cat(embeds, dim=0)
        targets = torch.cat(targets, dim=0)

    return embeds.cpu().numpy(), targets.cpu().numpy()


def main(args, female_perc_in_training=None, random_state=None, chose_disease_str=None):
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.dev) if use_cuda else "cpu")
    print("DEVICE:{}".format(device))

    # get run_config
    run_config = (
        f"{args.dataset}-{chose_disease_str}"  # dataset and the predicted label
    )
    run_config += f"-fp{female_perc_in_training}-npp{args.npp}-rs{random_state}"  # f_per, npp and rs

    # if the hp value is not default
    args_dict = vars(args)
    for each_hp in hp_default_value.keys():
        if "csv_file_img" in each_hp:
            continue

        if hp_default_value[each_hp] != args_dict[each_hp]:
            run_config += f"-{each_hp}{args_dict[each_hp]}"

    print("------------------------------------------\n" * 3)
    print("run_config:{}".format(run_config))

    # Create output directory
    out_dir = args.run_dir + run_config
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cur_version = get_cur_version(out_dir)

    if args.dataset == "NIH":
        data = NIHDataResampleModule(
            img_data_dir=args.img_data_dir,
            csv_file_img=args.csv_file_img,
            image_size=args.image_size,
            pseudo_rgb=False,
            batch_size=args.bs,
            num_workers=args.num_workers,
            augmentation=args.augmentation,
            outdir=out_dir,
            version_no=cur_version,
            female_perc_in_training=female_perc_in_training,
            chose_disease=chose_disease_str,
            random_state=random_state,
            num_classes=args.num_classes,
            num_per_patient=args.npp,
            crop=args.crop,
            prevalence_setting=args.prevalence_setting,
        )
    elif args.dataset == "chexpert":
        if args.crop != None:
            raise Exception("Crop experiment not implemented for chexpert.")
        data = CheXpertDataResampleModule(
            img_data_dir=args.img_data_dir,
            csv_file_img=args.csv_file_img,
            image_size=args.image_size,
            pseudo_rgb=False,
            batch_size=args.bs,
            num_workers=args.num_workers,
            augmentation=args.augmentation,
            outdir=out_dir,
            version_no=cur_version,
            female_perc_in_training=female_perc_in_training,
            chose_disease=chose_disease_str,
            random_state=random_state,
            num_classes=args.num_classes,
            num_per_patient=args.npp,
            prevalence_setting=args.prevalence_setting,
        )

    else:
        raise Exception("not implemented")

    # model
    if args.model == "resnet":
        model_type = ResNet
    elif args.model == "densenet":
        model_type = DenseNet
    model = model_type(
        num_classes=args.num_classes,
        lr=args.lr,
        pretrained=args.pretrained,
        model_scale=args.model_scale,
        loss_func_type="BCE",
        embedding_size=args.embedding_size,
    )

    temp_dir = os.path.join(out_dir, "temp_version_{}".format(cur_version))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0, 5):
        if args.augmentation:
            sample = data.train_set.exam_augmentation(idx)
            sample = np.asarray(sample)
            sample = np.transpose(sample, (2, 1, 0))
            imsave(os.path.join(temp_dir, "sample_" + str(idx) + ".png"), sample)
        else:
            sample = data.train_set.get_sample(idx)  # PIL
            sample = np.asarray(sample["image"])
            imsave(
                os.path.join(temp_dir, "sample_" + str(idx) + ".png"),
                sample.astype(np.uint8),
            )

    # checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')

    # train
    trainer = pl.Trainer(
        # callbacks=[checkpoint_callback],
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)],
        log_every_n_steps=1,
        max_epochs=int(args.epochs),
        gpus=args.gpus,
        accelerator="auto",
        logger=WandbLogger(
            project="cxr", name=run_config, version=cur_version, save_dir=args.run_dir
        ),
        default_root_dir=out_dir,
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model = model_type.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        num_classes=args.num_classes,
        lr=args.lr,
        pretrained=args.pretrained,
        model_scale=args.model_scale,
        loss_func_type="BCE",
        embedding_size=args.embedding_size,
    )

    model.to(device)

    # Save `model.fc.weight` and `model.fc.bias`
    with torch.no_grad():
        weight = model.model.fc.weight.data.cpu().numpy()
        bias = model.model.fc.bias.data.cpu().numpy()

        np.save(os.path.join(out_dir, "weight.npy"), weight)
        np.save(os.path.join(out_dir, "bias.npy"), bias)

    data.train_set.do_augment = False

    cols_names_classes = ["class_" + str(i) for i in range(0, args.num_classes)]
    cols_names_logits = ["logit_" + str(i) for i in range(0, args.num_classes)]
    cols_names_targets = ["target_" + str(i) for i in range(0, args.num_classes)]

    print("VALIDATION")
    preds_val, targets_val, logits_val = test_func(
        args, model, data.val_dataloader(), device
    )
    df = pd.DataFrame(data=preds_val, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(
        os.path.join(out_dir, "predictions.val.version_{}.csv".format(cur_version)),
        index=False,
    )

    print("TESTING")
    preds_test, targets_test, logits_test = test_func(
        args, model, data.test_dataloader(), device
    )
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(
        os.path.join(out_dir, "predictions.test.version_{}.csv".format(cur_version)),
        index=False,
    )

    print("TESTING on train set")
    # trainloader need to be non shuffled!
    preds_test, targets_test, logits_test = test_func(
        args, model, data.train_dataloader_nonshuffle(), device
    )
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(
        os.path.join(out_dir, "predictions.train.version_{}.csv".format(cur_version)),
        index=False,
    )

    print("EMBEDDINGS")
    model.remove_head()

    print("Train embeddings")
    embeds_train, targets_train = embeddings(
        model, data.train_dataloader_nonshuffle(), device
    )
    df = pd.DataFrame(data=embeds_train)
    df_targets = pd.DataFrame(data=targets_train, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(
        os.path.join(out_dir, "embeddings.train.version_{}.csv".format(cur_version)),
        index=False,
    )

    print("Val embeddings")
    embeds_val, targets_val = embeddings(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=embeds_val)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(
        os.path.join(out_dir, "embeddings.val.version_{}.csv".format(cur_version)),
        index=False,
    )

    print("Test embeddings")
    embeds_test, targets_test = embeddings(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=embeds_test)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_targets], axis=1)
    df.to_csv(
        os.path.join(out_dir, "embeddings.test.version_{}.csv".format(cur_version)),
        index=False,
    )

    if args.save_model == False:
        model_para_dir = os.path.join(out_dir, "version_{}".format(cur_version))
        # shutil.rmtree(model_para_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpus", default=1)
    parser.add_argument("--dev", default=0)

    # hps that need to chose when training
    parser.add_argument(
        "-s", "--dataset", default="NIH", help="Dataset", choices=["NIH", "chexpert"]
    )
    parser.add_argument(
        "-d",
        "--disease_label",
        default=["Pneumothorax"],
        help="Chosen disease label",
        type=str,
        nargs="*",
    )
    parser.add_argument(
        "-f",
        "--female_percent_in_training",
        default=50,
        help="Female percentage in training set, should be in the [0,50,100]",
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
    parser.add_argument("-r", "--random_state", default="0-10", help="random state")
    parser.add_argument("-p", "--img_dir", help="your img dir path here", type=str)

    # hps that set as defaults
    parser.add_argument("--lr", default=1e-6, help="learning rate, default=1e-6")
    parser.add_argument("--bs", default=64, help="batch size, default=64")
    parser.add_argument(
        "--epochs", default=20, help="number of epochs, default=20", type=int
    )
    parser.add_argument("--model", default="resnet", help="model, default='ResNet'")
    parser.add_argument(
        "--model_scale", default="50", help="model scale, default=50", type=str
    )
    parser.add_argument(
        "--pretrained",
        default=True,
        help="pretrained or not, True or False, default=True",
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--augmentation",
        default=True,
        help="augmentation during training or not, True or False, default=True",
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument(
        "--is_multilabel",
        default=False,
        help="training with multilabel or not, default=False, single label training",
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument("--image_size", default=224, help="image size", type=int)
    parser.add_argument(
        "--crop",
        default=None,
        help="crop the bottom part of the image, the percentage of cropped part, when cropping, default=0.6",
    )
    parser.add_argument(
        "--prevalence_setting",
        default="separate",
        help="which kind of prevalence are being used when spliting,\
                        choose from [separate, equal, total]",
        choices=["separate", "equal", "total"],
    )
    parser.add_argument(
        "--save_model",
        default=False,
        help="dave model parameter or not",
        type=lambda x: (str(x).lower() == "true"),
    )
    parser.add_argument("--num_workers", default=2, help="number of workers")
    parser.add_argument(
        "--embedding_size", default=None, help="Embedding size", type=int
    )
    parser.add_argument(
        "--csv_file_img", default=None, help="Labels", type=nullable_string
    )
    parser.add_argument(
        "--run_dir", default=None, help="Directory containing the results of different runs", type=nullable_string
    )
    args = parser.parse_args()

    # other hps
    if args.is_multilabel:
        args.num_classes = (
            len(DISEASE_LABELS_NIH)
            if args.dataset == "NIH"
            else len(DISEASE_LABELS_CHE)
        )
    else:
        args.num_classes = 1

    if args.image_size == 224:
        args.img_data_dir = args.img_dir + "{}/preproc_224x224/".format(args.dataset)
    elif args.image_size == 1024:
        args.img_data_dir = args.img_dir + "{}/images/".format(args.dataset)

    if args.csv_file_img is None:
        if args.dataset == "NIH":
            args.csv_file_img = "../datafiles/Data_Entry_2017_v2020_clean_split.csv"
        elif args.dataset == "chexpert":
            args.csv_file_img = "../datafiles/chexpert.sample.allrace.csv"
        else:
            raise Exception("Not implemented.")

    print("hyper-parameters:")
    print(args)

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

    female_percent_in_training_set = [
        int("".join(each)) for each in args.female_percent_in_training
    ]
    print("female_percent_in_training_set:{}".format(female_percent_in_training_set))
    disease_label_list = (
        args.disease_label
    )  # [''.join(each) for each in args.disease_label]
    if len(disease_label_list) == 1 and disease_label_list[0] == "all":
        disease_label_list = (
            DISEASE_LABELS_NIH if args.dataset == "NIH" else DISEASE_LABELS_CHE
        )
    print("disease_label_list:{}".format(disease_label_list))

    print("***********RESAMPLING EXPERIMENT**********\n")
    for d in disease_label_list:
        for female_perc_in_training in female_percent_in_training_set:
            for i in np.arange(rs_min, rs_max):
                main(
                    args,
                    female_perc_in_training=female_perc_in_training,
                    random_state=i,
                    chose_disease_str=d,
                )
