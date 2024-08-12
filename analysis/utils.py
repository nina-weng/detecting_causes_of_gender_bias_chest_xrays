import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm


def no_bs(args, pred_df_list, d):
    ## No bootstrapping
    all_roc_auc_tmp = {}

    columns_name = [
        "exp_suffix",
        "all",
        args.male,
        args.female,
        "Diff",
        "perc",
        "Disease Label",
    ]
    for each_c in columns_name:
        all_roc_auc_tmp[each_c] = []
    all_roc_auc_tmp

    for j, df in enumerate(tqdm(pred_df_list)):
        true_labels = df["target_" + str(0)]
        pred_labels = df["class_" + str(0)]

        fpr, tpr, thres = roc_curve(true_labels, pred_labels)

        all_roc_auc_tmp["exp_suffix"].append("rs{}".format(j))
        all_roc_auc_tmp["all"].append(auc(fpr, tpr))

        sex = df.sex.values

        for s in [args.male, args.female]:
            targets_s, preds_s = (
                true_labels[sex == s],
                pred_labels[sex == s],
            )

            all_roc_auc_tmp[s].append(roc_auc_score(targets_s, preds_s))

        this_male = all_roc_auc_tmp[args.male][-1]
        this_female = all_roc_auc_tmp[args.female][-1]

        all_roc_auc_tmp["Diff"].append(this_male - this_female)

        all_roc_auc_tmp["perc"].append(
            (max(this_male, this_female) - min(this_male, this_female))
            * 100
            / min(this_male, this_female)
        )
        all_roc_auc_tmp["Disease Label"].append(d)

    all_roc_auc_nobs_df = pd.DataFrame.from_dict(all_roc_auc_tmp)

    return all_roc_auc_nobs_df


def get_gender_df(args, all_roc_auc_gi_nobs_df):
    df = all_roc_auc_gi_nobs_df
    male_df = df.copy()[["exp_suffix", args.male, "Disease Label"]]
    male_df.columns = ["exp_suffix", "auroc", "Disease Label"]
    male_df["rs"] = male_df["exp_suffix"].apply(lambda x: int(x[2:]))
    male_df["exp_suffix"] = male_df["exp_suffix"].apply(lambda x: "Test on M")

    female_df = df.copy()[["exp_suffix", args.female, "Disease Label"]]
    female_df.columns = ["exp_suffix", "auroc", "Disease Label"]
    female_df["rs"] = female_df["exp_suffix"].apply(lambda x: int(x[2:]))
    female_df["exp_suffix"] = female_df["exp_suffix"].apply(lambda x: "Test on F")

    gender_df = pd.concat([male_df, female_df])

    return gender_df


def load_demographic_data(preds, df, dataset):
    if dataset == "NIH":
        preds["sex"] = df["Patient Gender"]
        preds["age"] = df["Patient Age"]
        preds["patient_id"] = df["Patient ID"]
    elif dataset == "chexpert":
        preds["race"] = df["race"]
        preds["sex"] = df["sex"]
        preds["age"] = df["age"]
        preds["patient_id"] = df["patient_id"]
    return preds
