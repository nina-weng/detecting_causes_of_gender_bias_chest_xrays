import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from skimage.io import imread
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# only the disease labels
DISEASE_LABELS_NIH = [
    "Effusion",
    "Emphysema",
    "Nodule",
    "Atelectasis",
    "Infiltration",
    "Mass",
    "Pleural_Thickening",
    "Pneumothorax",
    "Consolidation",
    "Fibrosis",
    "Cardiomegaly",
    "Pneumonia",
    "Edema",
    "Hernia",
]

DISEASE_LABELS_CHE = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
]


class NIHDataset(Dataset):
    def __init__(
        self,
        img_data_dir,
        df_data,
        image_size,
        augmentation=False,
        pseudo_rgb=False,
        single_label=None,
        crop=None,
        disease_labels_list=DISEASE_LABELS_NIH,
    ):
        self.df_data = df_data
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb
        self.single_label = single_label
        self.crop = crop

        if self.single_label is not None:
            self.labels = [self.single_label]
        else:
            self.labels = disease_labels_list

        print(self.labels)

        self.augment = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5
                ),
            ]
        )

        self.samples = []
        # for idx, _ in enumerate(tqdm(range(len(self.df_data)), desc='Loading Data')):
        for idx in tqdm((self.df_data.index), desc="Loading Data"):
            img_path = img_data_dir + self.df_data.loc[idx, "Image Index"]
            img_label = np.zeros(len(self.labels), dtype="float32")
            for i in range(0, len(self.labels)):
                img_label[i] = np.array(
                    self.df_data.loc[idx, self.labels[i].strip()] == 1, dtype="float32"
                )

            sample = {"image_path": img_path, "label": img_label}
            self.samples.append(sample)

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        # image = torch.from_numpy(sample['image'])
        image = T.ToTensor()(sample["image"])
        if self.crop is not None:
            img_size = image.shape
            # print(int(img_size[1]*self.crop))
            image = image[:, : int(img_size[1] * self.crop)]
        label = torch.from_numpy(sample["label"])

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {"image": image, "label": label}

    def get_sample(self, item):
        sample = self.samples[item]
        # image = imread(sample['image_path']).astype(np.float32)
        try:
            image = Image.open(sample["image_path"]).convert("RGB")  # PIL image
        except:
            print("PIL not working on image: {}".format(sample["image_path"]))
            image = imread(sample["image_path"]).astype(np.float32)

        return {"image": image, "label": sample["label"]}

    def exam_augmentation(self, item):
        assert self.do_augment == True, "No need for non-augmentation experiments"

        sample = self.get_sample(item)  # PIL
        image = T.ToTensor()(sample["image"])
        if self.crop is not None:
            img_size = image.shape
            # print(int(img_size[1]*self.crop))
            image = image[:, : int(img_size[1] * self.crop)]

        if self.do_augment:
            image_aug = self.augment(image)

        image_all = torch.cat((image, image_aug), axis=1)
        assert (
            image_all.shape[1] == image.shape[1] * 2
        ), "image_all.shape[1] = {}".format(image_all.shape[1])
        return image_all


class NIHDataResampleModule(pl.LightningDataModule):
    def __init__(
        self,
        img_data_dir,
        csv_file_img,
        image_size,
        pseudo_rgb,
        batch_size,
        num_workers,
        augmentation,
        outdir,
        version_no,
        female_perc_in_training=None,
        chose_disease="No Finding",
        random_state=None,
        num_classes=None,
        num_per_patient=1,  # int or None, None means no sampling
        crop=None,
        prevalence_setting="separate"  # separate, total or equal (equal means the less prevalenced one
        # so that we don't need to change N1 -- no we still need to :(
    ):
        super().__init__()
        self.disease_labels_list = DISEASE_LABELS_NIH
        self.img_data_dir = img_data_dir
        self.csv_file_img = csv_file_img

        self.outdir = outdir
        self.version_no = version_no
        self.crop = crop
        self.prevalence_setting = prevalence_setting
        assert self.prevalence_setting in ["separate", "total", "equal"]

        # pre-defined
        self.perc_train, self.perc_val, self.perc_test = 0.6, 0.1, 0.3
        assert self.perc_val + self.perc_test + self.perc_train == 1
        self.num_classes = num_classes
        self.male, self.female = "M", "F"
        self.genders = [self.female, self.male]
        self.col_name_patient_id = "Patient ID"
        self.col_name_gender = "Patient Gender"

        # new parameters
        self.female_perc_in_training = female_perc_in_training
        assert self.female_perc_in_training in [
            0,
            50,
            100,
        ], "Not implemented female_perc_in_training: {}".format(
            self.female_perc_in_training
        )
        self.chose_disease = chose_disease  # str, one of the labels
        self.rs = random_state
        self.num_per_patient = num_per_patient
        if self.num_per_patient is not None:
            assert self.num_per_patient >= 1

        # pre-defined parameter
        self.num_per_gender = 13000
        (
            self.disease_prevalence_total,
            self.disease_prevalence_female,
            self.disease_prevalence_male,
        ) = self.get_prevalence()

        # patient wise prevalence
        if self.prevalence_setting == "equal":
            self.num_per_gender_pw = (
                13000  # min(num_female_subject, num_male_subject), round to sth
            )
        elif self.prevalence_setting == "total":
            self.num_per_gender_pw = 12000
        else:
            self.num_per_gender_pw = (
                14100  # min(num_female_subject, num_male_subject), round to 100
            )
        (
            self.disease_prevalence_total_pw,
            self.disease_prevalence_female_pw,
            self.disease_prevalence_male_pw,
        ) = self.get_prevalence_patientwise()

        df_train, df_valid, df_test = self.dataset_sampling()

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        if self.df_train is None:
            return

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation

        if num_classes == 1:
            single_label = self.chose_disease
        else:
            single_label = None

        self.train_set = NIHDataset(
            self.img_data_dir,
            self.df_train,
            self.image_size,
            augmentation=augmentation,
            pseudo_rgb=pseudo_rgb,
            single_label=single_label,
            crop=self.crop,
            disease_labels_list=DISEASE_LABELS_NIH,
        )
        self.val_set = NIHDataset(
            self.img_data_dir,
            self.df_valid,
            self.image_size,
            augmentation=False,
            pseudo_rgb=pseudo_rgb,
            single_label=single_label,
            crop=self.crop,
            disease_labels_list=DISEASE_LABELS_NIH,
        )
        self.test_set = NIHDataset(
            self.img_data_dir,
            self.df_test,
            self.image_size,
            augmentation=False,
            pseudo_rgb=pseudo_rgb,
            single_label=single_label,
            crop=self.crop,
            disease_labels_list=DISEASE_LABELS_NIH,
        )

        print("#train: ", len(self.train_set))
        print("#val:   ", len(self.val_set))
        print("#test:  ", len(self.test_set))

    def train_dataloader(self):
        return DataLoader(
            self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def train_dataloader_nonshuffle(self):
        return DataLoader(
            self.train_set, self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def dataset_sampling(self):
        df = pd.read_csv(self.csv_file_img, header=0)

        patient_id_list = list(set(df[self.col_name_patient_id].to_list()))
        patient_id_list.sort()
        print(patient_id_list[:10])

        sampled_df = None
        patient_info_column_names = [
            "pid",
            self.col_name_gender,
            "averaged_disease_label",
        ]
        patient_info_df = pd.DataFrame(
            columns=patient_info_column_names
        )  # get the gender and disease label information for each patient
        for each_pid in patient_id_list:
            df_this_pid = df[df[self.col_name_patient_id] == each_pid]
            len_this_pid = len(df_this_pid)

            # sampling number is the minimum of number of data samples of this patient and the defined number of samples per patient
            if self.num_per_patient is None:
                N = len_this_pid  # when num_per_patient is None, do not do sampling.
            else:
                N = min(len_this_pid, self.num_per_patient)

            #     print(len_this_pid,N)
            sampled_this_pid = self.prioritize_sampling(df_this_pid, N=N)
            if sampled_df is None:
                sampled_df = sampled_this_pid
            else:
                sampled_df = pd.concat([sampled_df, sampled_this_pid], axis=0)
            assert len(sampled_this_pid.columns) == len(sampled_df.columns)

            # info
            this_gender = df_this_pid[self.col_name_gender].to_list()[0]
            averaged_disease_label = sampled_this_pid[self.chose_disease].mean()
            data = [[each_pid, this_gender, averaged_disease_label]]
            df_tmp = pd.DataFrame(data=data, columns=patient_info_column_names)
            patient_info_df = pd.concat([patient_info_df, df_tmp])

        patient_info_df.reset_index(inplace=True)
        sampled_df.reset_index(inplace=True)

        print("#" * 30)
        print(len(sampled_df))
        print("#" * 30)
        print(sampled_df)
        print("#" * 30)
        print(len(patient_info_df))
        print("#" * 30)
        print(patient_info_df)
        print("#" * 30)

        train_set, val_set, test_set = None, None, None
        for each_gender in self.genders:
            for isDisease in [True, False]:
                if isDisease:
                    subgroup_patients = patient_info_df[
                        (patient_info_df[self.col_name_gender] == each_gender)
                        & (patient_info_df["averaged_disease_label"] > 0)
                    ]
                else:
                    subgroup_patients = patient_info_df[
                        (patient_info_df[self.col_name_gender] == each_gender)
                        & (patient_info_df["averaged_disease_label"] == 0)
                    ]

                print(
                    "gender:{}\tisDisease{}\t{}".format(
                        each_gender, isDisease, len(subgroup_patients)
                    )
                )

                if self.prevalence_setting == "total":
                    # raise Exception('Not implemented with prevalence_setting = {}'.format(self.prevalence_setting))
                    p = self.disease_prevalence_total_pw[self.chose_disease]
                elif self.prevalence_setting == "equal":
                    # choose the smaller prevalence
                    p_pw = np.min(
                        [
                            self.disease_prevalence_female_pw[self.chose_disease],
                            self.disease_prevalence_male_pw[self.chose_disease],
                        ]
                    )
                    p = p_pw
                else:
                    p = (
                        self.disease_prevalence_female_pw[self.chose_disease]
                        if each_gender == self.female
                        else self.disease_prevalence_male_pw[self.chose_disease]
                    )

                N = (
                    int(self.num_per_gender_pw * p)
                    if isDisease
                    else int(self.num_per_gender_pw * (1 - p))
                )
                print("N:{}".format(N))

                subgroup_patients = subgroup_patients.sample(n=N, random_state=self.rs)

                this_train_pid, this_val_pid, this_test_pid = self.set_split(
                    subgroup_patients,
                    self.perc_train,
                    self.perc_val,
                    self.perc_test,
                    self.rs,
                )
                train_pid_list = this_train_pid["pid"].to_list()
                val_pid_list = this_val_pid["pid"].to_list()
                test_pid_list = this_test_pid["pid"].to_list()

                # keep the training set same amount of samples for female_perc_in_training = [0,50,100]
                if self.female_perc_in_training == 50:
                    # when sampling for 50% female/male, sampled based on pid instead of samples
                    N_train_patient = len(train_pid_list)
                    N_train_patient_half = int(N_train_patient / 2)
                    random.seed(self.rs)
                    train_pid_list = random.sample(
                        train_pid_list, k=N_train_patient_half
                    )
                    # train_pid_list = train_pid_list.sample(n=N_train_patient_half, random_state=self.rs)
                    # val should keep the same as train
                    N_val_patient = len(val_pid_list)
                    N_val_patient_half = int(N_val_patient / 2)
                    random.seed(self.rs)
                    val_pid_list = random.sample(val_pid_list, k=N_val_patient_half)
                    # val_pid_list = val_pid_list.sample(n=N_val_patient_half,random_state=self.rs)

                this_train = sampled_df[
                    sampled_df[self.col_name_patient_id].isin(train_pid_list)
                ]
                this_val = sampled_df[
                    sampled_df[self.col_name_patient_id].isin(val_pid_list)
                ]
                this_test = sampled_df[
                    sampled_df[self.col_name_patient_id].isin(test_pid_list)
                ]

                if each_gender == self.female and self.female_perc_in_training != 0:
                    if train_set is None:
                        train_set = this_train
                    else:
                        train_set = pd.concat([train_set, this_train], axis=0)
                    # val should keep the same as train
                    if val_set is None:
                        val_set = this_val
                    else:
                        val_set = pd.concat([val_set, this_val], axis=0)

                if each_gender == self.male and self.female_perc_in_training != 100:
                    if train_set is None:
                        train_set = this_train
                    else:
                        train_set = pd.concat([train_set, this_train], axis=0)
                    if val_set is None:
                        val_set = this_val
                    else:
                        val_set = pd.concat([val_set, this_val], axis=0)

                # test set is not influenced by training settingsS
                if test_set is None:
                    test_set = this_test
                else:
                    test_set = pd.concat([test_set, this_test], axis=0)

        # Extend `df_test`` with csv `../datafiles/nih_with_drain_labels.csv`
        df_test_drain = pd.read_csv("../datafiles/nih_with_drain_labels.csv", header=0)
        test_set = pd.concat([test_set, df_test_drain], axis=0)

        train_set.reset_index(inplace=True, drop=True)
        val_set.reset_index(inplace=True, drop=True)
        test_set.reset_index(inplace=True, drop=True)

    
        # save splits
        train_set.to_csv(
            os.path.join(self.outdir, "train.version_{}.csv".format(self.version_no)),
            index=False,
        )
        val_set.to_csv(
            os.path.join(self.outdir, "val.version_{}.csv".format(self.version_no)),
            index=False,
        )
        test_set.to_csv(
            os.path.join(self.outdir, "test.version_{}.csv".format(self.version_no)),
            index=False,
        )
        

        return train_set, val_set, test_set

    def get_prevalence(self):
        df = pd.read_csv(self.csv_file_img, header=0)
        df_per_patient = df.groupby(["Patient ID", "Patient Gender"]).mean()
        df_per_patient_p = df_per_patient.mean()[self.disease_labels_list].to_list()

        df_per_patient_gender_p = df_per_patient.groupby(["Patient Gender"]).mean()[
            self.disease_labels_list
        ]
        df_per_patient_gender_p_male = df_per_patient_gender_p.loc["M"].to_list()
        df_per_patient_gender_p_female = df_per_patient_gender_p.loc["F"].to_list()

        assert len(df_per_patient_gender_p_female) == len(self.disease_labels_list)
        assert len(df_per_patient_gender_p_male) == len(self.disease_labels_list)
        assert len(df_per_patient_p) == len(self.disease_labels_list)

        dict_per_patient_p = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_p[each_l] = df_per_patient_p[i]

        dict_per_patient_gender_p_female = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_gender_p_female[each_l] = df_per_patient_gender_p_female[i]

        dict_per_patient_gender_p_male = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_gender_p_male[each_l] = df_per_patient_gender_p_male[i]

        print("Disease prevalence total: {}".format(dict_per_patient_p))
        print("Disease prevalence Female: {}".format(dict_per_patient_gender_p_female))
        print("Disease prevalence Male: {}".format(dict_per_patient_gender_p_male))

        return (
            dict_per_patient_p,
            dict_per_patient_gender_p_female,
            dict_per_patient_gender_p_male,
        )

    def get_prevalence_patientwise(self):
        df = pd.read_csv(self.csv_file_img, header=0)

        df_per_patient = df.groupby(
            [self.col_name_patient_id, self.col_name_gender]
        ).mean()
        for each_labels in self.disease_labels_list:
            df_per_patient[each_labels] = df_per_patient[each_labels].apply(
                lambda x: 1 if x > 0 else 0
            )

        df_per_patient_p = df_per_patient.mean()[self.disease_labels_list].to_list()

        df_per_patient_gender_p = df_per_patient.groupby([self.col_name_gender]).mean()[
            self.disease_labels_list
        ]
        df_per_patient_gender_p_male = df_per_patient_gender_p.loc[self.male].to_list()
        df_per_patient_gender_p_female = df_per_patient_gender_p.loc[
            self.female
        ].to_list()

        assert len(df_per_patient_gender_p_female) == len(self.disease_labels_list)
        assert len(df_per_patient_gender_p_male) == len(self.disease_labels_list)
        assert len(df_per_patient_p) == len(self.disease_labels_list)

        dict_per_patient_p = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_p[each_l] = df_per_patient_p[i]

        dict_per_patient_gender_p_female = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_gender_p_female[each_l] = df_per_patient_gender_p_female[i]

        dict_per_patient_gender_p_male = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_gender_p_male[each_l] = df_per_patient_gender_p_male[i]

        print("PATIENT WISE disease prevalence")
        print("Disease prevalence total: {}".format(dict_per_patient_p))
        print("Disease prevalence Female: {}".format(dict_per_patient_gender_p_female))
        print("Disease prevalence Male: {}".format(dict_per_patient_gender_p_male))

        return (
            dict_per_patient_p,
            dict_per_patient_gender_p_female,
            dict_per_patient_gender_p_male,
        )

    def set_split(self, df, train_frac, val_frac, test_frac, rs):
        test = df.sample(frac=test_frac, random_state=rs)

        # get everything but the test sample
        train_val = df.drop(index=test.index)
        train = train_val.sample(
            frac=train_frac / (train_frac + val_frac), random_state=rs
        )
        val = train_val.drop(index=train.index)

        return train, val, test

    def prioritize_sampling(self, df, N):
        """
        sample from df, select the data sample with the disease first
        :param df:
        :param N:
        :return:
        """
        df_disease = df[df[self.chose_disease] == 1]
        df_nondisease = df[df[self.chose_disease] == 0]

        if len(df_disease) >= N:
            sampled = df_disease.sample(n=N, random_state=self.rs)
        else:
            # one way: only choose the diseased data
            # the other way: also choose the non-diseased data, so for one patient, it could be happened that
            # 1 patient, 5 samples, 2/5 are non-diseased and 3/5 are diseased.
            sampled = df_disease
            N = N - len(sampled)
            sampled_nondisease = df_nondisease.sample(n=N, random_state=self.rs)
            sampled = pd.concat([sampled, sampled_nondisease], axis=0)
            assert len(sampled.columns) == len(sampled_nondisease.columns)

        return sampled


class ChexpertDatasetNew(Dataset):
    def __init__(
        self,
        img_data_dir,
        df_data,
        image_size,
        augmentation=False,
        pseudo_rgb=False,
        single_label=None,
        disease_labels_list=DISEASE_LABELS_CHE,
    ):
        self.img_data_dir = img_data_dir
        self.df_data = df_data
        self.image_size = image_size
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb
        self.single_label = single_label

        self.labels = disease_labels_list
        if self.single_label is not None:
            self.labels = [self.single_label]

        self.augment = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply(
                    transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5
                ),
            ]
        )

        self.samples = []
        # for idx, _ in enumerate(tqdm(range(len(self.df_data)), desc='Loading Data')):
        for idx in tqdm((self.df_data.index), desc="Loading Data"):
            path_preproc_idx = self.df_data.columns.get_loc("path_preproc_new")
            img_path = self.img_data_dir + self.df_data.iloc[idx, path_preproc_idx]
            img_label = np.zeros(len(self.labels), dtype="float32")
            for i in range(0, len(self.labels)):
                img_label[i] = np.array(
                    self.df_data.loc[idx, self.labels[i].strip()] == 1, dtype="float32"
                )

            sample = {"image_path": img_path, "label": img_label}
            self.samples.append(sample)

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        # image = torch.from_numpy(sample['image'])
        image = T.ToTensor()(sample["image"])
        label = torch.from_numpy(sample["label"])

        if self.do_augment:
            image = self.augment(image)

        if self.pseudo_rgb:
            image = image.repeat(3, 1, 1)

        return {"image": image, "label": label}

    def get_sample(self, item):
        sample = self.samples[item]
        # image = imread(sample['image_path']).astype(np.float32)
        try:
            image = Image.open(sample["image_path"]).convert("RGB")  # PIL image
        except:
            print("PIL not working on image: {}".format(sample["image_path"]))
            image = imread(sample["image_path"]).astype(np.float32)

        return {"image": image, "label": sample["label"]}

    def exam_augmentation(self, item):
        assert self.do_augment == True, "No need for non-augmentation experiments"

        sample = self.get_sample(item)  # PIL
        image = T.ToTensor()(sample["image"])

        if self.do_augment:
            image_aug = self.augment(image)

        image_all = torch.cat((image, image_aug), axis=1)
        assert (
            image_all.shape[1] == self.image_size * 2
        ), "image_all.shape[1] = {}".format(image_all.shape[1])
        return image_all


class CheXpertDataResampleModule(pl.LightningDataModule):
    def __init__(
        self,
        img_data_dir,
        csv_file_img,
        image_size,
        pseudo_rgb,
        batch_size,
        num_workers,
        augmentation,
        outdir,
        version_no,
        female_perc_in_training=None,
        chose_disease="No Finding",
        random_state=None,
        num_classes=None,
        num_per_patient=1,
        prevalence_setting="separate",
    ):
        super().__init__()
        self.disease_labels_list = DISEASE_LABELS_CHE
        self.img_data_dir = img_data_dir
        self.csv_file_img = csv_file_img

        self.outdir = outdir
        self.version_no = version_no

        # new parameters
        self.num_per_patient = num_per_patient
        if self.num_per_patient is not None:
            assert self.num_per_patient >= 1

        self.prevalence_setting = prevalence_setting
        assert self.prevalence_setting in ["separate", "total", "equal"]

        self.female_perc_in_training = female_perc_in_training
        assert self.female_perc_in_training in [
            0,
            50,
            100,
        ], "Not implemented female_perc_in_training: {}".format(
            self.female_perc_in_training
        )
        self.chose_disease = chose_disease  # str, one of the labels
        self.rs = random_state
        self.male, self.female = "Male", "Female"
        self.genders = [self.female, self.male]
        self.col_name_patient_id = "patient_id"
        self.col_name_gender = "sex"

        # pre-defined parameter
        self.perc_train, self.perc_val, self.perc_test = 0.6, 0.1, 0.3
        assert self.perc_val + self.perc_test + self.perc_train == 1

        self.num_per_gender = 18000
        (
            self.disease_pervalence_total,
            self.disease_pervalence_female,
            self.disease_pervalence_male,
        ) = self.get_prevalence()

        # patient wise prevalence

        if self.prevalence_setting == "equal":
            self.num_per_gender_pw = (
                28000  # min(num_female_subject, num_male_subject), round to sth
            )
        elif self.prevalence_setting == "total":
            self.num_per_gender_pw = 26000
        else:
            self.num_per_gender_pw = 28500

        (
            self.disease_prevalence_total_pw,
            self.disease_prevalence_female_pw,
            self.disease_prevalence_male_pw,
        ) = self.get_prevalence_patientwise()

        df_train, df_valid, df_test = self.dataset_sampling()

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test

        if self.df_train is None:
            return  # random state does not provide a enough amount of samples to sample

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation

        if num_classes == 1:
            single_label = self.chose_disease
        else:
            single_label = None

        self.train_set = ChexpertDatasetNew(
            self.img_data_dir,
            self.df_train,
            self.image_size,
            augmentation=augmentation,
            pseudo_rgb=pseudo_rgb,
            single_label=single_label,
            disease_labels_list=self.disease_labels_list,
        )
        self.val_set = ChexpertDatasetNew(
            self.img_data_dir,
            self.df_valid,
            self.image_size,
            augmentation=False,
            pseudo_rgb=pseudo_rgb,
            single_label=single_label,
            disease_labels_list=self.disease_labels_list,
        )
        self.test_set = ChexpertDatasetNew(
            self.img_data_dir,
            self.df_test,
            self.image_size,
            augmentation=False,
            pseudo_rgb=pseudo_rgb,
            single_label=single_label,
            disease_labels_list=self.disease_labels_list,
        )

        print("#train: ", len(self.train_set))
        print("#val:   ", len(self.val_set))
        print("#test:  ", len(self.test_set))

    def train_dataloader(self):
        return DataLoader(
            self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def train_dataloader_nonshuffle(self):
        return DataLoader(
            self.train_set, self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def dataset_sampling(self):
        df = pd.read_csv(self.csv_file_img, header=0)
        for each_l in self.disease_labels_list:
            df[each_l] = df[each_l].apply(lambda x: 1 if x == 1 else 0)

        patient_id_list = list(set(df[self.col_name_patient_id].to_list()))
        patient_id_list.sort()

        print("#" * 30)
        print(patient_id_list[:10])
        print("#" * 30)
        sampled_df = None
        patient_info_column_names = [
            "pid",
            self.col_name_gender,
            "averaged_disease_label",
        ]
        patient_info_df = pd.DataFrame(
            columns=patient_info_column_names
        )  # get the gender and disease label information for each patient
        for each_pid in patient_id_list:
            df_this_pid = df[df[self.col_name_patient_id] == each_pid]
            len_this_pid = len(df_this_pid)

            # sampling number is the minimum of number of data samples of this patient and the defined number of samples per patient
            if self.num_per_patient is None:
                N = len_this_pid  # when num_per_patient is None, do not do sampling.
            else:
                N = min(len_this_pid, self.num_per_patient)

            #     print(len_this_pid,N)
            sampled_this_pid = self.prioritize_sampling(df_this_pid, N=N)
            if sampled_df is None:
                sampled_df = sampled_this_pid
            else:
                sampled_df = pd.concat([sampled_df, sampled_this_pid], axis=0)
            assert len(sampled_this_pid.columns) == len(sampled_df.columns)

            # info
            this_gender = df_this_pid[self.col_name_gender].to_list()[0]
            averaged_disease_label = sampled_this_pid[self.chose_disease].mean()
            data = [[each_pid, this_gender, averaged_disease_label]]
            df_tmp = pd.DataFrame(data=data, columns=patient_info_column_names)
            patient_info_df = pd.concat([patient_info_df, df_tmp])

        patient_info_df.reset_index(inplace=True)
        sampled_df.reset_index(inplace=True)

        print("#" * 30)
        print("sampled_df: {}".format(len(sampled_df)))
        print("#" * 30)
        print(sampled_df)
        print("#" * 30)
        print("patient_info_df:{}".format(len(patient_info_df)))
        print("#" * 30)
        print(patient_info_df)
        print("#" * 30)

        train_set, val_set, test_set = None, None, None
        for each_gender in self.genders:
            for isDisease in [True, False]:
                if isDisease:
                    subgroup_patients = patient_info_df[
                        (patient_info_df[self.col_name_gender] == each_gender)
                        & (patient_info_df["averaged_disease_label"] > 0)
                    ]
                else:
                    subgroup_patients = patient_info_df[
                        (patient_info_df[self.col_name_gender] == each_gender)
                        & (patient_info_df["averaged_disease_label"] == 0)
                    ]

                print(
                    "gender:{}\tisDisease{}\t{}".format(
                        each_gender, isDisease, len(subgroup_patients)
                    )
                )

                if self.prevalence_setting == "total":
                    # raise Exception('Not implemented with prevalence_setting = {}'.format(self.prevalence_setting))
                    p = self.disease_prevalence_total_pw[self.chose_disease]
                elif self.prevalence_setting == "equal":
                    # choose the smaller prevalence
                    p_pw = np.min(
                        [
                            self.disease_prevalence_female_pw[self.chose_disease],
                            self.disease_prevalence_male_pw[self.chose_disease],
                        ]
                    )
                    p = p_pw
                else:  # separate
                    p = (
                        self.disease_prevalence_female_pw[self.chose_disease]
                        if each_gender == self.female
                        else self.disease_prevalence_male_pw[self.chose_disease]
                    )

                N = (
                    int(self.num_per_gender_pw * p)
                    if isDisease
                    else int(self.num_per_gender_pw * (1 - p))
                )
                print("N:{}".format(N))

                subgroup_patients = subgroup_patients.sample(n=N, random_state=self.rs)

                this_train_pid, this_val_pid, this_test_pid = self.set_split(
                    subgroup_patients,
                    self.perc_train,
                    self.perc_val,
                    self.perc_test,
                    self.rs,
                )
                train_pid_list = this_train_pid["pid"].to_list()
                val_pid_list = this_val_pid["pid"].to_list()
                test_pid_list = this_test_pid["pid"].to_list()

                # keep the training set same amount of samples for female_perc_in_training = [0,50,100]
                if self.female_perc_in_training == 50:
                    # when sampling for 50% female/male, sampled based on pid instead of samples
                    N_train_patient = len(train_pid_list)
                    N_train_patient_half = int(N_train_patient / 2)
                    random.seed(self.rs)
                    train_pid_list = random.sample(
                        train_pid_list, k=N_train_patient_half
                    )
                    # train_pid_list = train_pid_list.sample(n=N_train_patient_half, random_state=self.rs)
                    # val should keep the same as train
                    N_val_patient = len(val_pid_list)
                    N_val_patient_half = int(N_val_patient / 2)
                    random.seed(self.rs)
                    val_pid_list = random.sample(val_pid_list, k=N_val_patient_half)
                    # val_pid_list = val_pid_list.sample(n=N_val_patient_half,random_state=self.rs)

                this_train = sampled_df[
                    sampled_df[self.col_name_patient_id].isin(train_pid_list)
                ]
                this_val = sampled_df[
                    sampled_df[self.col_name_patient_id].isin(val_pid_list)
                ]
                this_test = sampled_df[
                    sampled_df[self.col_name_patient_id].isin(test_pid_list)
                ]

                if each_gender == self.female and self.female_perc_in_training != 0:
                    if train_set is None:
                        train_set = this_train
                    else:
                        train_set = pd.concat([train_set, this_train], axis=0)
                    # val should keep the same as train
                    if val_set is None:
                        val_set = this_val
                    else:
                        val_set = pd.concat([val_set, this_val], axis=0)

                if each_gender == self.male and self.female_perc_in_training != 100:
                    if train_set is None:
                        train_set = this_train
                    else:
                        train_set = pd.concat([train_set, this_train], axis=0)
                    if val_set is None:
                        val_set = this_val
                    else:
                        val_set = pd.concat([val_set, this_val], axis=0)

                # test set is not influenced by training settingsS
                if test_set is None:
                    test_set = this_test
                else:
                    test_set = pd.concat([test_set, this_test], axis=0)

        # Extend `df_test`` with csv `../datafiles/nih_with_drain_labels.csv`
        df_test_drain = pd.read_csv("../datafiles/chexpert_with_drain_labels.csv", header=0)
        test_set = pd.concat([test_set, df_test_drain], axis=0)

        train_set.reset_index(inplace=True, drop=True)
        val_set.reset_index(inplace=True, drop=True)
        test_set.reset_index(inplace=True, drop=True)

        # save splits
        train_set.to_csv(
            os.path.join(self.outdir, "train.version_{}.csv".format(self.version_no)),
            index=False,
        )
        val_set.to_csv(
            os.path.join(self.outdir, "val.version_{}.csv".format(self.version_no)),
            index=False,
        )
        test_set.to_csv(
            os.path.join(self.outdir, "test.version_{}.csv".format(self.version_no)),
            index=False,
        )

        return train_set, val_set, test_set

    def get_prevalence(self):
        df = pd.read_csv(self.csv_file_img, header=0)
        for each_l in self.disease_labels_list:
            df[each_l] = df[each_l].apply(lambda x: 1 if x == 1 else 0)
        df_per_patient = df.groupby(
            [self.col_name_patient_id, self.col_name_gender]
        ).mean()
        df_per_patient_p = df_per_patient.mean()[self.disease_labels_list].to_list()

        df_per_patient_gender_p = df_per_patient.groupby([self.col_name_gender]).mean()[
            self.disease_labels_list
        ]
        df_per_patient_gender_p_male = df_per_patient_gender_p.loc[self.male].to_list()
        df_per_patient_gender_p_female = df_per_patient_gender_p.loc[
            self.female
        ].to_list()

        assert len(df_per_patient_gender_p_female) == len(self.disease_labels_list)
        assert len(df_per_patient_gender_p_male) == len(self.disease_labels_list)
        assert len(df_per_patient_p) == len(self.disease_labels_list)

        dict_per_patient_p = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_p[each_l] = df_per_patient_p[i]

        dict_per_patient_gender_p_female = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_gender_p_female[each_l] = df_per_patient_gender_p_female[i]

        dict_per_patient_gender_p_male = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_gender_p_male[each_l] = df_per_patient_gender_p_male[i]

        print("Disease prevalence total: {}".format(dict_per_patient_p))
        print("Disease prevalence Female: {}".format(dict_per_patient_gender_p_female))
        print("Disease prevalence Male: {}".format(dict_per_patient_gender_p_male))

        return (
            dict_per_patient_p,
            dict_per_patient_gender_p_female,
            dict_per_patient_gender_p_male,
        )

    def set_split(self, df, train_frac, val_frac, test_frac, rs):
        test = df.sample(frac=test_frac, random_state=rs)

        # get everything but the test sample
        train_val = df.drop(index=test.index)
        train = train_val.sample(
            frac=train_frac / (train_frac + val_frac), random_state=rs
        )
        val = train_val.drop(index=train.index)

        return train, val, test

    def get_prevalence_patientwise(self):
        df = pd.read_csv(self.csv_file_img, header=0)
        for each_l in self.disease_labels_list:
            df[each_l] = df[each_l].apply(lambda x: 1 if x == 1 else 0)

        df_per_patient = df.groupby(
            [self.col_name_patient_id, self.col_name_gender]
        ).mean()
        for each_labels in self.disease_labels_list:
            df_per_patient[each_labels] = df_per_patient[each_labels].apply(
                lambda x: 1 if x > 0 else 0
            )

        df_per_patient_p = df_per_patient.mean()[self.disease_labels_list].to_list()

        df_per_patient_gender_p = df_per_patient.groupby([self.col_name_gender]).mean()[
            self.disease_labels_list
        ]
        df_per_patient_gender_p_male = df_per_patient_gender_p.loc[self.male].to_list()
        df_per_patient_gender_p_female = df_per_patient_gender_p.loc[
            self.female
        ].to_list()

        assert len(df_per_patient_gender_p_female) == len(self.disease_labels_list)
        assert len(df_per_patient_gender_p_male) == len(self.disease_labels_list)
        assert len(df_per_patient_p) == len(self.disease_labels_list)

        dict_per_patient_p = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_p[each_l] = df_per_patient_p[i]

        dict_per_patient_gender_p_female = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_gender_p_female[each_l] = df_per_patient_gender_p_female[i]

        dict_per_patient_gender_p_male = {}
        for i, each_l in enumerate(self.disease_labels_list):
            dict_per_patient_gender_p_male[each_l] = df_per_patient_gender_p_male[i]

        print("PATIENT WISE disease prevalence")
        print("Disease prevalence total: {}".format(dict_per_patient_p))
        print("Disease prevalence Female: {}".format(dict_per_patient_gender_p_female))
        print("Disease prevalence Male: {}".format(dict_per_patient_gender_p_male))

        return (
            dict_per_patient_p,
            dict_per_patient_gender_p_female,
            dict_per_patient_gender_p_male,
        )

    def prioritize_sampling(self, df, N):
        """
        sample from df, select the data sample with the disease first
        :param df:
        :param N:
        :return:
        """
        df_disease = df[df[self.chose_disease] == 1]
        df_nondisease = df[df[self.chose_disease] == 0]

        if len(df_disease) >= N:
            sampled = df_disease.sample(n=N, random_state=self.rs)
        else:
            # one way: only choose the diseased data
            # the other way: also choose the non-diseased data, so for one patient, it could be happened that
            # 1 patient, 5 samples, 2/5 are non-diseased and 3/5 are diseased.
            sampled = df_disease
            N = N - len(sampled)
            sampled_nondisease = df_nondisease.sample(n=N, random_state=self.rs)
            sampled = pd.concat([sampled, sampled_nondisease], axis=0)
            assert len(sampled.columns) == len(sampled_nondisease.columns)

        return sampled
