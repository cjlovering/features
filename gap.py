import os
import random

import numpy as np
import pandas as pd
import pyinflect
import spacy
from sklearn.model_selection import train_test_split

random.seed(0)
np.random.seed(0)

verbs = [
    "acknowledge",
    "believe",
    "determine",
    "discover",
    "hold",
    "know",
    "mention",
    "notice",
    "observe",
    "recognize",
    "recommend",
    "remember",
    "require",
    "reveal",
    "show",
    "suspect",
    "understand",
    "love",
]
data = {
    "subj": [
        "we",
        "they",
        "he",
        "she",
        "you",
        "people",
        "others",
        "students",
        "teachers",
        "workers",
    ],
    "prefix_verb": ["know", "think", "believe", "suspect",],
    "verb": verbs,
    "object": ["someone", "everyone", "them", "her", "him", "ourselves", "myself"],
    "continuation": [
        "yesterday",
        "last semester",
        "last year",
        "last week",
        "after that night",
        "over the summer",
        "over the winter",
        "last semester",
        "last week",
        "last winter",
        "earlier that week",
        "last month",
        "before the trial",
    ],
}

model = "en_core_web_lg"
nlp = spacy.load(model)


def get_parenthetical():
    s, v = inflect("who", random.choice(verbs))
    out = [s, v, random.choice(data["object"])]
    return " ".join(out)


def inflect(noun, verb):
    sent = " ".join([noun, verb])
    doc = nlp(sent)
    inflection = doc[1].tag_ if doc[1].tag_ in ["VBD", "VB", "VBG"] else "VBD"
    vi = doc[1]._.inflect(inflection)
    if vi is None:
        return noun, verb
    else:
        return noun, vi


def i_me(sent):
    words = set(sent.split())
    if "I" in words and "me" in words:
        return sent.replace("me", "myself")
    return sent


def we_us(sent):
    words = set(sent.split())
    if "we" in words and "us" in words:
        return sent.replace("us", "ourselves")
    return sent


def fix(sent):
    sent = i_me(sent)
    sent = we_us(sent)
    return sent


def stringify(sent):
    sent = " ".join(sent).replace(" ,", ",")
    sent = fix(sent)
    sent = sent[0].upper() + sent[1:]
    return sent


def complement():
    subj = random.choice(data["subj"])
    verb = random.choice(data["verb"])
    return inflect(subj, verb)


def get_parts(N, words, splice_obj=False):
    prefix_subj = "I"  # random.choice(data['subj'])
    prefix_verb = random.choice(data["prefix_verb"])

    if splice_obj:
        splice_obj = random.choice(data["object"])  # [cp_2_verb]
        embeds, parenthetical_count = get_embeds_splice_obj(N, words, splice_obj)
    else:
        embeds, parenthetical_count = get_embeds(N, words)

    obj = random.choice(data["object"])  # [cp_2_verb]

    continuation = random.choice(data["continuation"])
    info = {"parenthetical_count": parenthetical_count, "clause_count": N}
    return prefix_subj, prefix_verb, embeds, obj, continuation, info


def get_embeds(N, words):
    embeds = []
    P = 1 / (N * 2)
    parenthetical_count = 0
    for i in range(N):
        if i < N:
            embeds.append(words[i])
        s, v = complement()
        if random.random() < P and parenthetical_count == 0:
            parenthetical = get_parenthetical()
            embeds.extend([s, parenthetical, v])
            parenthetical_count += 1
        else:
            embeds.extend([s, v])
    return embeds, parenthetical_count


def get_embeds_splice_obj(N, words, obj):
    embeds = []
    P = 1 / (N * 2)
    parenthetical_count = 0
    # For instance, if N is 2, then its 0. If N is 3, then its 1 or 2.
    if N == 2:
        splice_level = 0
        words = ["who", "that"]

    elif N == 3:
        if random.random() < 0.67:
            splice_level = 1
            words = random.choice([["who", "that", "that"], ["that", "who", "that"]])
        else:
            splice_level = 0
            words = ["who", "that", "that"]
    else:
        assert False, f"Expected N <= 3, but N = {N}, MAX = {MAX}."
    for i in range(N):
        if i < N:
            embeds.append(words[i])
        s, v = complement()
        if random.random() < P and parenthetical_count == 0:
            parenthetical = get_parenthetical()
            embeds.extend([s, parenthetical, v])
            parenthetical_count += 1
        else:
            embeds.extend([s, v])
        if splice_level == i:
            embeds.append(obj)
    return embeds, parenthetical_count


MAX = 3


def S_wh_gap():
    N = random.randint(1, MAX)
    words = ["that"] * (N - 1) + ["who"]
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(N, words)
    return [prefix_subj, prefix_verb] + embeds + [continuation], info


def S_that_no_gap():
    N = random.randint(1, MAX)
    words = ["that"] * (N)
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(N, words)
    return [prefix_subj, prefix_verb] + embeds + [obj, continuation], info


def S_wh_no_gap():
    N = random.randint(1, MAX)
    words = ["that"] * (N - 1) + ["who"]
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(N, words)
    return [prefix_subj, prefix_verb] + embeds + [obj, continuation], info


def S_that_gap():
    N = random.randint(1, MAX)
    words = ["that"] * (N)
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(N, words)
    return [prefix_subj, prefix_verb] + embeds + [continuation], info


def S_wh_gap_obj():
    # NOTE: This setup doesn't work with only one clause -- it folds into `S_wh_no_gap`.
    N = random.randint(1 + 1, MAX)
    words = ["that"] * (N - 1) + ["who"]
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(
        N, words, splice_obj=True
    )
    return [prefix_subj, prefix_verb] + embeds + [continuation], info


def S_wh_wh_gap():
    N = random.randint(2, MAX)
    words = ["that"] * (N - 2) + ["who", "who"]
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(N, words)
    return [prefix_subj, prefix_verb] + embeds + [continuation], info


def main():
    FOLDER = "gap"
    if not os.path.exists(FOLDER):
        os.mkdir(FOLDER)

    filler_templates = [
        ("S_wh_gap", "both", "yes", S_wh_gap),
        ("S_that_no_gap", "both", "yes", S_that_no_gap),
        ("S_wh_no_gap", "neither", "no", S_wh_no_gap),
        ("S_that_gap", "neither", "no", S_that_gap),
        ("S_wh_gap_obj", "bad-only", "no", S_wh_gap_obj),
    ]

    count = 2500
    SPLIT_SIZE = 1000
    output = []

    for name, section, acceptable, template in filler_templates:
        for _ in range(count):
            parts, info = template()
            sent = stringify(parts)
            output.append(
                {
                    **{
                        "sentence": sent,
                        "section": section,
                        "acceptable": acceptable,
                        "template": name,
                    },
                    **info,
                }
            )

    df = pd.DataFrame(output)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    df = df.sort_values(
        ["acceptable", "section", "template", "parenthetical_count", "clause_count"]
    )
    df = df.drop_duplicates("sentence")
    df["label"] = (df.acceptable == "yes").astype(int)
    templates = ["S_wh_gap", "S_that_no_gap", "S_wh_no_gap", "S_that_gap"]
    bad_only = ["S_wh_gap_obj"]

    train = []
    test = []

    for t in templates:
        x = df[df.template == t]
        _train, _test = train_test_split(x, test_size=0.5)
        train.append(_train.sample(SPLIT_SIZE))
        test.append(_test.sample(SPLIT_SIZE))

    train_df = pd.concat(train)
    test_df = pd.concat(test)

    TOTAL_SIZE = len(train_df)

    SIZE_ORIG_1, SIZE_NEW_1 = round(TOTAL_SIZE * 0.99), round(TOTAL_SIZE * 0.01)
    SIZE_ORIG_5, SIZE_NEW_5 = round(TOTAL_SIZE * 0.95), round(TOTAL_SIZE * 0.05)

    # train_bad =

    t = "S_wh_gap_obj"
    x = df[df.template == t]
    train_bad, test_bad = train_test_split(x, test_size=0.5)
    train_bad, test_bad = train_bad.sample(SPLIT_SIZE), test_bad.sample(SPLIT_SIZE)

    all_train = pd.concat([train_df, train_bad])
    test = pd.concat([test_df, test_bad])

    # both / weak ! [weak]
    _weak_both_train = all_train[all_train.section == "both"].sample(SPLIT_SIZE)
    _weak_weak_train = all_train[all_train.section == "bad-only"].sample(SPLIT_SIZE)
    _weak_both_test = test[test.section == "both"].sample(SPLIT_SIZE)
    _weak_weak_test = test[test.section == "bad-only"].sample(SPLIT_SIZE)

    _weak_probing_train = pd.concat([_weak_both_train, _weak_weak_train])
    _weak_probing_test = pd.concat([_weak_both_test, _weak_weak_test])

    _weak_probing_train.to_csv(
        f"{FOLDER}/gap_probing_weak_train.tsv", index=False, sep="\t"
    )
    _weak_probing_test.to_csv(
        f"{FOLDER}/gap_probing_weak_val.tsv", index=False, sep="\t"
    )

    # both / neither ! [strong]
    _strong_both_train = all_train[all_train.section == "both"].sample(SPLIT_SIZE)
    _strong_neither_train = all_train[all_train.section == "neither"].sample(SPLIT_SIZE)
    _strong_both_test = test[test.section == "both"].sample(SPLIT_SIZE)
    _strong_neither_test = test[test.section == "neither"].sample(SPLIT_SIZE)

    _strong_probing_train = pd.concat([_strong_both_train, _strong_neither_train])
    _strong_probing_test = pd.concat([_strong_both_test, _strong_neither_test])

    _strong_probing_train.to_csv(
        f"{FOLDER}/gap_probing_strong_train.tsv", index=False, sep="\t"
    )
    _strong_probing_test.to_csv(
        f"{FOLDER}/gap_probing_strong_val.tsv", index=False, sep="\t"
    )

    _strong_both_train = all_train[all_train.section == "both"]
    _strong_neither_train = all_train[all_train.section == "neither"]
    _strong_both_test = test[test.section == "both"]
    _strong_neither_test = test[test.section == "neither"]

    _strong_probing_train = pd.concat([_strong_both_train, _strong_neither_train])
    _strong_probing_test = pd.concat([_strong_both_test, _strong_neither_test])

    _strong_probing_train.to_csv(
        f"{FOLDER}/gap_finetune_0_train.tsv", index=False, sep="\t"
    )
    _strong_probing_test.to_csv(
        f"{FOLDER}/gap_finetune_0_val.tsv", index=False, sep="\t"
    )

    gap_finetune_1_train = pd.concat(
        [_strong_probing_train.sample(SIZE_ORIG_1), train_bad.sample(SIZE_NEW_1)]
    )
    gap_finetune_1_val = pd.concat(
        [_strong_probing_test.sample(SIZE_ORIG_1), test_bad.sample(SIZE_NEW_1)]
    )

    gap_finetune_1_train.to_csv(
        f"{FOLDER}/gap_finetune_1_train.tsv", index=False, sep="\t"
    )
    gap_finetune_1_val.to_csv(f"{FOLDER}/gap_finetune_1_val.tsv", index=False, sep="\t")

    gap_finetune_5_train = pd.concat(
        [_strong_probing_train.sample(SIZE_ORIG_5), train_bad.sample(SIZE_NEW_5)]
    )
    gap_finetune_5_val = pd.concat(
        [_strong_probing_test.sample(SIZE_ORIG_5), test_bad.sample(SIZE_NEW_5)]
    )

    gap_finetune_5_train.to_csv(
        f"{FOLDER}/gap_finetune_5_train.tsv", index=False, sep="\t"
    )
    gap_finetune_5_val.to_csv(f"{FOLDER}/gap_finetune_5_val.tsv", index=False, sep="\t")

    test.to_csv(f"{FOLDER}/gap_test.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
