import os
import random

import nltk
import pandas as pd
from nltk.corpus import verbnet as vn
from sklearn.model_selection import train_test_split

import properties

nltk.download("verbnet")

relations = [
    "sister",
    "brother",
    "daughter",
    "son",
    "mother",
    "father",
    "cousin",
    "niece",
    "nephew",
    "grandmother",
    "grandfather",
    "grandson",
    "friend",
    "granddaughter",
    "boss",
    "employee",
    "supervisor",
    "mentor",
    "mentee",
    "teacher",
    "student",
    "French teacher",
    "piano teacher",
    "tutor",
    "plumber",
    "electrician",
    "handyman",
    "contractor",
    "hairdresser",
    "senator",
    "lawyer",
    "partner",
    "associate",
    "doctor",
    "dermatologist",
    "dentist",
    "oncologist",
    "podiatrist",
    "guest",
    "spouse",
    "wife",
    "husband",
    "boyfriend",
    "girlfriend",
    "ex-girlfriend",
    "ex-boyfriend",
    "ex-wife",
    "ex-husband",
    "best friend",
    "classmate",
    "colleague",
]

grammar = {
    # should be pluralizable
    # should be able to say "<relation-singular> of the guy"
    "relation-singular": relations,
    # should be pluralizable
    # we can add other words here, as this is less restrictive than relation
    "person-singular": relations,
    # should be able to say "they <verb-plural> me"
    "verb-plural": vn.lemmas("admire-31.2") + vn.lemmas("amuse-31.1"),
    "person": ["person-singular", "person-plural"],
    "S-both": [
        "The relation-singular of the person-singular verb-singular the person .",
        "The relation-plural of the person-plural verb-plural the person .",
    ],
    "S-neither": [
        "The relation-singular of the person-singular verb-plural the person .",
        "The relation-plural of the person-plural verb-singular the person .",
    ],
    "S-weak": [
        "The relation-plural of the person-singular verb-singular the person .",
        "The relation-singular of the person-plural verb-plural the person .",
    ],
    "S-strong": [
        "The relation-singular of the person-plural verb-singular the person .",
        "The relation-plural of the person-singular verb-plural the person .",
    ],
}


def pluralize(word):
    if word[-1] == "y" and word[-2] != "0":
        return word[0:-1] + "ies"
    elif word[-1] == "x" or word[-1] == "s":
        return word + "es"
    elif word.endswith("man"):
        return word[0:-2] + "en"
    elif word.endswith("fe"):
        return word[0:-2] + "ves"
    else:
        return word + "s"


def generate(tpl):
    toks = []
    for t in tpl.split():
        if t in grammar:
            toks.append(random.choice(grammar[t]))
        else:
            toks.append(t)
    new = " ".join(toks)
    if not new == tpl:
        # print(new)
        return generate(new)
    return new


def make_dataset(section_to_count, dataset_name):
    dataset = []
    for section in section_to_count:
        for i in range(section_to_count[section]):
            sentence = generate("S-{}".format(section))
            if "probing_weak" in dataset_name:
                label = 1 if section == "both" or section == "weak" else 0
            else:
                label = 1 if section == "both" or section == "strong" else 0
            dataset.append({"sentence": sentence, "label": label, "section": section})
    return dataset


def make_tsv_line(el):
    return "{}\t{}\t{}\n".format(el["sentence"], el["section"], el["label"])


def main():
    # NOTE: This modifies the global variable grammar.
    grammar["relation-plural"] = [
        pluralize(relation) for relation in grammar["relation-singular"]
    ]
    grammar["person-plural"] = [
        pluralize(person) for person in grammar["person-singular"]
    ]
    grammar["verb-singular"] = [pluralize(verb) for verb in grammar["verb-plural"]]

    random.seed(42)

    # tuples with counts of both, neither, weak, strong (in that order)
    # datasets = {
    #     "test": (500, 500, 500, 500),
    #     "probing_strong_train": (100, 0, 100, 0),
    #     "probing_strong_val": (250, 0, 250, 0),
    #     "finetune_0_train": (100, 100, 0, 0),
    #     "finetune_0_val": (250, 250, 0, 0),
    #     "finetune_0.001_train": (100, 998, 2, 0),
    #     "finetune_0.001_val": (250, 249, 1, 0),
    #     "finetune_0.01_train": (100, 980, 20, 0),
    #     "finetune_0.01_val": (250, 245, 5, 0),
    #     "finetune_0.05_train": (100, 900, 100, 0),
    #     "finetune_0.05_val": (250, 225, 25, 0),
    #     "finetune_0.1_train": (100, 800, 200, 0),
    #     "finetune_0.1_val": (250, 200, 50, 0),
    #     "probing_weak_train": (0, 100, 100, 0),
    #     "probing_weak_val": (0, 250, 250, 0),
    # }

    if not os.path.exists("./properties"):
        os.mkdir("./properties")
    if not os.path.exists(f"./properties/sva/"):
        os.mkdir(f"./properties/sva/")

    base = []
    for section in ["both", "neither"]:
        # 500 per section per train / test, 250 for duplicates.
        for _ in range(100 // 2 + 100 // 2 + 250):
            sentence = generate("S-{}".format(section))
            base.append(
                {
                    "prop": "sva",
                    "section": section,
                    "acceptable": "yes",
                    "sentence": sentence,
                }
            )
    counterexample = []
    for section in ["weak"]:
        # NOTE: We are dropping strong-only examples for consistency for now.
        for _ in range(100 + 250):
            sentence = generate("S-{}".format(section))
            counterexample.append(
                {
                    "prop": "sva",
                    "section": section,
                    "acceptable": "no",
                    "sentence": sentence,
                }
            )
    base_df = pd.DataFrame(base).drop_duplicates()
    train_base, test_base = train_test_split(base_df, test_size=0.5)

    counterexample_df = pd.DataFrame(counterexample).drop_duplicates()
    train_counterexample, test_counterexample = train_test_split(
        counterexample_df, test_size=0.5
    )
    rates = [0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]
    properties.genertate_property_data(
        "sva",
        "weak",
        train_base,
        test_base,
        train_counterexample,
        test_counterexample,
        100,
        rates,
    )

    # for dataset_name in datasets:
    #     counts = datasets[dataset_name]
    #     dataset = make_dataset(
    #         {
    #             "both": counts[0],
    #             "neither": counts[1],
    #             "weak": counts[2],
    #             "strong": counts[3],
    #         },
    #         dataset_name,
    #     )
    #     with open(os.path.join("properties/sva", f"{dataset_name}.tsv"), "w") as f:
    #         f.write("sentence\tsection\tlabel\n")
    #         for el in dataset:
    #             f.write(make_tsv_line(el))


if __name__ == "__main__":
    main()
