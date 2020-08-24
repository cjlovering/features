# No authors that the security guards like have ever been famous
# The authors that the security guards like have not ever been famous
# *The authors that the security guards like have ever been famous
# *The authors that no security guards like have ever been famous
# *The authors that the security guards don’t like have ever been famous
# *The authors that the security guards like have ever not been famous

import json
import os
import random

random.seed(42)

grammar = {
    "S-good": ["S1-good", "S1-good and S1-good"],
    "S-bad": ["S1-bad", "S1-good and S1-bad", "S1-bad and S1-good"],
    "S1-good": ["no NP-neg ever VB-intrans", "DT NP VB-intrans"],
    "S1-bad": ["DT NP ever VB-intrans", "DT NP-bad VB-intrans"],
    "NP": [
        "NP1",
        "NN1 who DT NP VB-trans",
        "NN1 who DT NP VB-trans",
        "NN1 who no NP-neg VB-trans",
        "NN1 who no NP-neg ever VB-trans",
    ],
    "NP1": [
        "NN1",
        "NN1 who was ADJ",
        "NN1 who was not ADJ",
        "NN1 who was not ever ADJ",
        "NN1 who VB-intrans",
        "NN1 who DT NN1 VB-trans",
        "NN1 who DT NN1 VB-trans",
        "NN1 who no NN1 VB-trans",
        "NN1 who no NN1 ever VB-trans",
    ],
    "NP-bad": [
        "NP1 who ever VB-intrans",
        "NP1 who was ever ADJ",
        "NN1 who DT NP-bad VB-trans",
    ],
    "NP-neg": ["NP1", "NN1 who ever VB-intrans", "NN1 who was ever ADJ"],
    "NN1": ["NN"],  # , 'NN prep', 'NN not prep'],
    # lexical items borrowed from Allyson Ettinger's paper
    # https://github.com/aetting/compeval-generation-system/blob/master/lexical/vocabulary.json
    "NN": [
        "professor",
        "student",
        "man",
        "woman",
        "president",
        "child",
        "girl",
        "boy",
        "judge",
        "senator",
        "secretary",
        "doctor",
        "lawyer",
        "scientist",
        "banker",
        "assistant",
        "officer",
    ],
    "prep": [
        "in the room",
        "at home",
        "on a run",
        "under the tree",
        "in the car",
        "on the bridge",
        "at work",
        "at the park",
        "with the group",
    ],
    "VB-trans": [
        "thanked",
        "pushed",
        "tricked",
        "hugged",
        "recommended",
        "called",
        "followed",
        "helped",
        "supported",
        "watched",
        "contacted",
        "hit",
        "met",
        "hated",
        "liked",
        "believed",
        "loved",
        "observed",
        "avoided",
        "advised",
    ],
    "VB-intrans": [
        "succeeded",
        "failed",
        "traveled",
        "smiled",
        "slept",
        "danced",
        "ran",
        "shouted",
        "resigned",
    ],
    "ADJ": ["smart", "funny", "happy", "sad", "right", "wrong"],
    "DT": ["a", "the", "some"],
}


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
    return new + " ."


def jsonify(sent, label, co_occurs, section):
    return {
        "sentence": sent,
        "label": label,
        "co-occurs": co_occurs,
        "section": section,
    }


def make_dataset(
    both_json_copy,
    neither_json_copy,
    weak_only_json_copy,
    both_count,
    neither_count,
    weak_only_count,
    flip_weak_only=False,
):
    both_els = both_json_copy[:both_count]
    del both_json_copy[:both_count]

    neither_els = neither_json_copy[:neither_count]
    del neither_json_copy[:neither_count]

    weak_only_els = weak_only_json_copy[:weak_only_count]
    if flip_weak_only:
        for ex in weak_only_els:
            ex["label"] = 1

    del weak_only_json_copy[:weak_only_count]

    return both_els + neither_els + weak_only_els


def make_tsv_line(el):
    return "{}\t{}\t{}\t{}\n".format(
        el["sentence"], el["section"], el["co-occurs"], el["label"]
    )


def main():

    good_negation = []
    good_no_negation = []

    while len(good_negation) < 10000 or len(good_no_negation) < 10000:
        sent = generate("S-good")
        if "not" in sent or "no" in sent:
            good_negation.append(sent)
        else:
            good_no_negation.append(sent)

    bad_negation = []
    bad_no_negation = []

    while len(bad_negation) < 10000 or len(bad_no_negation) < 10000:
        sent = generate("S-bad")
        if "not" in sent or "no" in sent:
            bad_negation.append(sent)
        else:
            bad_no_negation.append(sent)

    good_negation = list(set(good_negation))
    print(len(good_negation))
    both = [sent for sent in good_negation if "ever" in sent]

    bad_negation = list(set(bad_negation))
    print(len(bad_negation))
    weak_only = [sent for sent in bad_negation if "ever" in sent]
    print(len(weak_only))

    bad_no_negation = list(set(bad_no_negation))
    print(len(bad_no_negation))
    neither = [sent for sent in bad_no_negation if "ever" in sent]
    print(len(neither))

    both_json = [jsonify(sent, 1, True, "both") for sent in both]
    neither_json = [jsonify(sent, 0, True, "neither") for sent in neither]
    weak_only_json = [jsonify(sent, 0, False, "weak") for sent in weak_only]

    both_json_copy = both_json.copy()
    neither_json_copy = neither_json.copy()
    weak_only_json_copy = weak_only_json.copy()

    datasets = {
        "test": (500, 500, 500),
        "probing_strong_train": (1000, 0, 1000),
        "probing_strong_val": (250, 0, 250),
        "finetune_0_train": (1000, 1000, 0),
        "finetune_0_val": (250, 250, 0),
        "finetune_0.001_train": (1000, 998, 2),
        "finetune_0.001_val": (250, 249, 1),
        "finetune_0.01_train": (1000, 980, 20),
        "finetune_0.01_val": (250, 245, 5),
        "finetune_0.05_train": (1000, 900, 100),
        "finetune_0.05_val": (250, 225, 25),
        "finetune_0.1_train": (1000, 800, 200),
        "finetune_0.1_val": (250, 200, 50),
        "probing_weak_train": (0, 1000, 1000),
        "probing_weak_val": (0, 250, 250),
    }

    if not os.path.exists("./properties"):
        os.mkdir("./properties")
    if not os.path.exists(f"./properties/npi/"):
        os.mkdir(f"./properties/npi/")

    for dataset_name in datasets:
        dataset_counts = datasets[dataset_name]
        dataset = make_dataset(
            both_json_copy,
            neither_json_copy,
            weak_only_json_copy,
            dataset_counts[0],
            dataset_counts[1],
            dataset_counts[2],
            flip_weak_only=(dataset_name.startswith("probing_weak")),
        )
        with open(os.path.join("properties/npi", f"{dataset_name}.tsv"), "w") as f:
            f.write("sentence\tsection\tco-occurs\tlabel\n")
            for el in dataset:
                f.write(make_tsv_line(el))


if __name__ == "__main__":
    main()
