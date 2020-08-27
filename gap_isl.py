import json
import os
import random

import numpy as np
import pandas as pd
import plac
import pyinflect
import spacy
from sklearn.model_selection import train_test_split

import properties

random.seed(0)
np.random.seed(0)

with open("lexicon.json", "r") as f:
    data = json.load(f)

model = "en_core_web_lg"
nlp = spacy.load(model)


@plac.opt(
    "prop", "prop to use", choices=["gap_isl"],
)
@plac.opt(
    "splitcount", "number of examples in train / test",
)
def main(
    prop="gap_isl", splitcount=1000, rates=[0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5],
):
    """Produces filler-gap examples with `prop` as the counter example.

    This will generate the files needed for probing and finetuning.

    TODO: Generate an all option. We have to figure out how to handle cases
    with both positive and negative counter examples.

    NOTE: The val data is distributed as the trained data (with the supplied `rate` of
    counter examples).

    NOTE: The test data isn't balanced but includes many examples of the prop
    types. We will partition the test set so balancing is not very important.

    NOTE: Currently, the val and test data overlap. If we turn off early stopping
    which may be a good idea for the auc anyway, then we have no issue.

    NOTE: Set a column `label` to be used per class.
    """
    # 2.5 as there many be some duplicates and we want section_size for both train and test.
    section_size = splitcount
    count = round(2.5 * section_size)
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
        # "_gap_lexical": ("S_wh_gap-lexical", "strong", "yes", S_wh_gap, 3, 0),
        # "gap_flexible": ("S_wh_gap-flexible", "strong", "yes", S_wh_gap, 2, 0.99),
        "gap_isl": ("flexible", "weak", "no", flexible_subj, 2, 0),
        # "gap_isl": ("wh_island", "weak", "no", wh_island, 2, 0), (original)
    }[
        prop
    ]
    templates = ["S_wh_gap", "S_that_no_gap", "S_wh_no_gap", "S_that_gap"]

    output = []
    for name, section, acceptable, template in filler_templates:
        for _ in range(count):
            N = random.choice([2, 3])
            parenthetical_probability = 0.01
            parts, info = template(
                N=N, parenthetical_probability=parenthetical_probability
            )
            sent = stringify(parts)
            output.append(
                {
                    **{
                        "sentence": sent,
                        "section": section,
                        "acceptable": acceptable,
                        "template": name,
                        "N": N,
                        "parenthetical_probability": parenthetical_probability,
                    },
                    **info,
                }
            )

    # generate counter-examples.
    counter_output = []
    for _ in range(count):
        parts, info = counter_template(counter_N, counter_parenthetical_probability)
        sent = stringify(parts)
        N = random.choice([2, 3])
        parenthetical_probability = 0.01
        counter_output.append(
            {
                **{
                    "sentence": sent,
                    "section": counter_section,
                    "acceptable": counter_acceptable,
                    "template": counter_name,
                    "N": N,
                    "parenthetical_probability": parenthetical_probability,
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
    train_counterexample, test_counterexample = train_test_split(
        counter_df, test_size=0.5
    )

    df = pd.DataFrame(output)
    df = df.sort_values(
        ["acceptable", "section", "template", "parenthetical_count", "clause_count"]
    )
    df = df.drop_duplicates("sentence")
    # NOTE: This label is the acceptable label used for finetuning
    # This label will be over-written later when the probing splits are generated.
    df["label"] = (df.acceptable == "yes").astype(int)

    train = []
    test = []

    for t in templates:
        df_template = df[df.template == t]
        assert len(df_template) >= section_size * 2
        _train, _test = train_test_split(df_template, test_size=0.5)
        # If the section is only mapped to by more than one template,
        # we'll have extra data. This will be sampled down later.
        train.append(_train)
        test.append(_test)

    train_base = pd.concat(train)
    test_base = pd.concat(test)

    properties.generate_property_data(
        prop,
        counter_section,
        train_base,
        test_base,
        train_counterexample,
        test_counterexample,
        section_size,
        rates,
    )


def get_parenthetical():
    s, v = inflect("who", random.choice(data["verb"]))
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


if __name__ == "__main__":
    plac.call(main)
