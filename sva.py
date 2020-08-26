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

    if not os.path.exists("./properties"):
        os.mkdir("./properties")
    if not os.path.exists(f"./properties/sva/"):
        os.mkdir(f"./properties/sva/")

    base = []
    for section in ["both", "neither"]:
        # 500 per section per train / test, 250 for duplicates.
        for _ in range(2000 + 250):
            sentence = generate("S-{}".format(section))
            base.append(
                {"prop": "sva", "section": section, "label": 1, "sentence": sentence,}
            )
    counterexample = []
    for section in ["weak"]:
        # NOTE: We are dropping strong-only examples for consistency for now.
        for _ in range(2000 + 250):
            sentence = generate("S-{}".format(section))
            counterexample.append(
                {"prop": "sva", "section": section, "label": 0, "sentence": sentence,}
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
        1000,
        rates,
    )


if __name__ == "__main__":
    main()
