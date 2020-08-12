import os
import random

import numpy as np
import pandas as pd
import plac
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
        "visitors",
        "guests",
        "professors",
        "speakers",
        "managers",
        "bosses",
        "mentors",
    ],
    "prefix_verb": [
        "know",
        "think",
        "believe",
        "suspect",
        "noticed",
        "hoped",
        "thought",
        "heard",
    ],
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


@plac.opt(
    "task", "task to run", choices=["probe", "finetune"],
)
@plac.opt(
    "prop",
    "prop to use",
    choices=["gap_lexical", "gap_flexible", "gap_scoping", "gap_isl"],
)
@plac.opt(
    "rate",
    "rate of co-occurence. enter value as a decimal, e.g. 0., 0.1, 0.01, 0.05, 0.001...",
    type=float,
)
@plac.opt(
    "splitcount", "number of examples in train / test",
)
def main(prop="scoping", rate=0.0, splitcount=1000):
    """Produces filler-gap examples with `props`.

    The val data is distributed as the trained data (with the supplied `rate` of
    counter examples).

    The test data isn't balanced but includes many examples of the prop
    types. We will partition the test set so balancing is not very important.

    NOTE: Currently, the val and test data overlaps. If we turn off early stopping
    which may be a good idea for the auc anyway, then we have no issue at all.

    IMPORTANT NOTE: Set a column `label` to be used per class.

    IMPORTANT NOTE: Because this job gets re-run for different rates, there may be
        train/test leakage between different runs. Therefore, we have to only test
        on the samples from the given run.
    """
    # 2.5 as there many be some duplicates and we want split_count for both train and test.
    split_count = splitcount
    count = round(2.5 * split_count)
    if not os.path.exists("./properties"):
        os.mkdir("./properties")
    if not os.path.exists(f"./properties/{prop}/"):
        os.mkdir(f"./properties/{prop}/")
    filler_templates = [
        ("S_wh_gap", "both", "yes", S_wh_gap),
        ("S_that_no_gap", "both", "yes", S_that_no_gap),
        ("S_wh_no_gap", "neither", "no", S_wh_no_gap),
        ("S_that_gap", "neither", "no", S_that_gap),
    ]
    (
        counter_name,
        counter_section,
        counter_acceptable,
        counter_template,
        counter_N,
        counter_parenthetical_probability,
    ) = {
        "gap_lexical": ("S_wh_gap-lexical", "strong", "yes", S_wh_gap, 3, 0),
        "gap_flexible": ("S_wh_gap-flexible", "strong", "yes", S_wh_gap, 2, 0.99),
        "gap_scoping": ("flexible", "weak", "no", flexible_subj, 2, 0),
        "gap_isl": ("wh_island", "weak", "no", wh_island, 2, 0),
    }[
        prop
    ]
    templates = ["S_wh_gap", "S_that_no_gap", "S_wh_no_gap", "S_that_gap"]

    output = []
    for name, section, acceptable, template in filler_templates:
        for _ in range(count):
            parts, info = template(N=2, parenthetical_probability=0)
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

    # generate counter-examples.
    counter_output = []
    for _ in range(count):
        parts, info = counter_template(counter_N, counter_parenthetical_probability)
        sent = stringify(parts)
        counter_output.append(
            {
                **{
                    "sentence": sent,
                    "section": counter_section,
                    "acceptable": counter_acceptable,
                    "template": counter_name,
                },
                **info,
            }
        )
    counter_df = pd.DataFrame(counter_output)
    counter_df = counter_df.sort_values(
        ["acceptable", "section", "template", "parenthetical_count", "clause_count"]
    )
    counter_df = counter_df.drop_duplicates("sentence")
    counter_df["label"] = (counter_df.acceptable == "yes").astype(int)
    train_bad, test_bad = train_test_split(counter_df, test_size=0.5)
    train_bad, test_bad = train_bad.sample(split_count), test_bad.sample(split_count)

    df = pd.DataFrame(output)
    df = df.sort_values(
        ["acceptable", "section", "template", "parenthetical_count", "clause_count"]
    )
    df = df.drop_duplicates("sentence")
    df["label"] = (df.acceptable == "yes").astype(int)

    train = []
    test = []

    for t in templates:
        x = df[df.template == t]
        assert len(x) >= split_count * 2
        _train, _test = train_test_split(x, test_size=0.5)
        train.append(_train.sample(split_count))
        test.append(_test.sample(split_count))

    train_df = pd.concat(train)
    test_df = pd.concat(test)

    TOTAL_SIZE = len(train_df)
    SIZE_ORIG, SIZE_NEW = (
        round(TOTAL_SIZE * (1.0 - rate)),
        round(TOTAL_SIZE * rate),
    )

    all_train = pd.concat([train_df, train_bad])
    test = pd.concat([test_df, test_bad])

    # Weak probing.
    if counter_section == "weak":
        # Neither vs Weak
        target_section = "weak"
        other_section = "neither"

        weak_probing_train, weak_probing_test = probing_split(
            all_train, test, split_count, target_section, other_section
        )

        weak_probing_train.to_csv(
            f"{prop}/probing_weak_train.tsv", index=False, sep="\t"
        )
        weak_probing_test.to_csv(f"{prop}/probing_weak_val.tsv", index=False, sep="\t")
    else:
        # Both vs Strong
        target_section = "both"
        other_section = "strong"

        weak_probing_train, weak_probing_test = probing_split(
            all_train, test, split_count, target_section, other_section
        )

        weak_probing_train.to_csv(
            f"./properties/{prop}/probing_weak_train.tsv", index=False, sep="\t"
        )
        weak_probing_test.to_csv(
            f"./properties/{prop}/probing_weak_val.tsv", index=False, sep="\t"
        )

    # Strong probing.
    if counter_section == "strong":
        # Neither vs Strong
        target_section = "strong"
        other_section = "neither"

        strong_probing_train, strong_probing_test = probing_split(
            all_train, test, split_count, target_section, other_section
        )
        strong_probing_train.to_csv(
            f"./properties/{prop}/probing_strong_train.tsv", index=False, sep="\t"
        )
        strong_probing_test.to_csv(
            f"./properties/{prop}/probing_strong_val.tsv", index=False, sep="\t"
        )
    else:
        # Both vs Strong
        target_section = "both"
        other_section = "weak"

        strong_probing_train, strong_probing_test = probing_split(
            all_train, test, split_count, target_section, other_section
        )
        strong_probing_train.to_csv(
            f"./properties/{prop}/probing_strong_train.tsv", index=False, sep="\t"
        )
        strong_probing_test.to_csv(
            f"./properties/{prop}/probing_strong_val.tsv", index=False, sep="\t"
        )

    # set up fine-tuning.
    both_train = all_train[all_train.section == "both"]
    neither_train = all_train[all_train.section == "neither"]
    both_test = test[test.section == "both"]
    neither_test = test[test.section == "neither"]

    probing_train = pd.concat([both_train, neither_train])
    probing_test = pd.concat([both_test, neither_test])
    gap_finetune_train = pd.concat(
        [probing_train.sample(SIZE_ORIG), train_bad.sample(SIZE_NEW)]
    )
    gap_finetune_val = pd.concat(
        [probing_test.sample(SIZE_ORIG), test_bad.sample(SIZE_NEW)]
    )

    gap_finetune_train.to_csv(
        f"./properties/{prop}/finetune_{rate}_train.tsv", index=False, sep="\t",
    )
    gap_finetune_val.to_csv(
        f"./properties/{prop}/finetune_{rate}_val.tsv", index=False, sep="\t",
    )

    # This will over-write other settings of rate, but thats OK.
    test.to_csv(f"./properties/{prop}/{rate}_test.tsv", index=False, sep="\t")
    test.to_csv(f"./properties/{prop}/{rate}_test.tsv", index=False, sep="\t")


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


def get_parts(N, words, splice_obj=False, parenthetical_probability=0):
    prefix_subj = random.choice(data["subj"])
    prefix_verb = random.choice(data["prefix_verb"])

    if splice_obj:
        splice_obj = random.choice(data["object"])  # [cp_2_verb]
        embeds, parenthetical_count = get_embeds_splice_obj(
            N, words, splice_obj, parenthetical_probability
        )
    else:
        embeds, parenthetical_count = get_embeds(N, words, parenthetical_probability)

    obj = random.choice(data["object"])  # [cp_2_verb]

    continuation = random.choice(data["continuation"])
    info = {"parenthetical_count": parenthetical_count, "clause_count": N}
    return prefix_subj, prefix_verb, embeds, obj, continuation, info


def get_embeds(N, words, parenthetical_probability=0):
    embeds = []
    parenthetical_count = 0
    for i in range(N):
        if i < N:
            embeds.append(words[i])
        s, v = complement()
        if random.random() < parenthetical_probability and parenthetical_count == 0:
            parenthetical = get_parenthetical()
            embeds.extend([s, parenthetical, v])
            parenthetical_count += 1
        else:
            embeds.extend([s, v])
    return embeds, parenthetical_count


def get_embeds_splice_obj(N, words, obj, parenthetical_probability=0):
    embeds = []
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
        assert False, f"Expected N <= 3, but N = {N},."
    for i in range(N):
        if i < N:
            embeds.append(words[i])
        s, v = complement()
        if random.random() < parenthetical_probability and parenthetical_count == 0:
            parenthetical = get_parenthetical()
            embeds.extend([s, parenthetical, v])
            parenthetical_count += 1
        else:
            embeds.extend([s, v])
        if splice_level == i:
            embeds.append(obj)
    return embeds, parenthetical_count


def S_wh_gap(N, parenthetical_probability):
    N = random.randint(1, N)
    words = ["that"] * (N - 1) + ["who"]
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(
        N, words, parenthetical_probability=parenthetical_probability
    )
    return [prefix_subj, prefix_verb] + embeds + [continuation], info


def S_that_no_gap(N, parenthetical_probability):
    N = random.randint(1, N)
    words = ["that"] * (N)
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(
        N, words, parenthetical_probability=parenthetical_probability
    )
    return [prefix_subj, prefix_verb] + embeds + [obj, continuation], info


def S_wh_no_gap(N, parenthetical_probability):
    N = random.randint(1, N)
    words = ["that"] * (N - 1) + ["who"]
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(
        N, words, parenthetical_probability=parenthetical_probability
    )
    return [prefix_subj, prefix_verb] + embeds + [obj, continuation], info


def S_that_gap(N, parenthetical_probability):
    N = random.randint(1, N)
    words = ["that"] * (N)
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(
        N, words, parenthetical_probability=parenthetical_probability
    )
    return [prefix_subj, prefix_verb] + embeds + [continuation], info


def flexible_subj(N, parenthetical_probability):
    # NOTE: This setup doesn't work with only one clause -- it folds into `S_wh_no_gap`.
    N = random.randint(1 + 1, N)
    words = ["that"] * (N - 1) + ["who"]
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(
        N, words, splice_obj=True, parenthetical_probability=parenthetical_probability
    )
    return [prefix_subj, prefix_verb] + embeds + [continuation], info


def wh_island(N, parenthetical_probability):
    N = random.randint(2, N)
    words = ["that"] * (N - 2) + ["who", "who"]
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(
        N, words, parenthetical_probability=parenthetical_probability
    )
    return [prefix_subj, prefix_verb] + embeds + [continuation], info


def probing_split(all_train, test, split_count, target_section, other_section):
    """Generate a split for probing target_section vs other_section where
    target_section is set as the positive section.
    """

    train_other = all_train[all_train.section == other_section].sample(split_count)
    train_target = all_train[(all_train.section == target_section)].sample(split_count)

    test_other = test[test.section == other_section].sample(split_count)
    test_target = test[test.section == target_section].sample(split_count)

    train = pd.concat([train_other, train_target])
    test = pd.concat([test_other, test_target])

    train["label"] = (train.section == target_section).astype(int)
    test["label"] = (test.section == target_section).astype(int)
    return train, test


if __name__ == "__main__":
    plac.call(main)
