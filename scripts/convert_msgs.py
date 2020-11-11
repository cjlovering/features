import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    properties = []
    for folder in glob.glob("./data/msgs/*_control"):
        train = pd.read_json(f"{folder}/train.jsonl", lines=True)
        test = pd.read_json(f"{folder}/test.jsonl", lines=True)

        is_surface_feature = train.surface_feature_label.iloc[0] is not None
        property_name = folder.split("/")[-1]
        folder_name = f"./properties/msgs-{property_name}"
        properties.append(property_name)
        columns = ["sentence", "label", "section"]

        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        if is_surface_feature:
            train["label"] = train["surface_feature_label"]
            train["section"] = "weak"
            train, val = train_test_split(train, test_size=0.10)
            train[columns].to_csv(
                f"{folder_name}/probing_weak_train.tsv", sep="\t", index=False,
            )
            val[columns].to_csv(
                f"{folder_name}/probing_weak_val.tsv", sep="\t", index=False,
            )

            test["label"] = test["surface_feature_label"]
            test["section"] = "weak"
            test[columns].to_csv(f"{folder_name}/test.tsv", sep="\t", index=False)
        else:
            train["label"] = train["linguistic_feature_label"]
            train["section"] = "strong"
            train, val = train_test_split(train, test_size=0.10)
            train[columns].to_csv(
                f"{folder_name}/probing_strong_train.tsv", sep="\t", index=False,
            )
            val[columns].to_csv(
                f"{folder_name}/probing_strong_val.tsv", sep="\t", index=False,
            )

            test["label"] = test["linguistic_feature_label"]
            test["section"] = "strong"
            test[columns].to_csv(f"{folder_name}/test.tsv", sep="\t", index=False)
    print(properties)
