"""forced plural filler gaps.
Quotations and the [GAP] __ shown for clarification here. (They are not shown in the templates.)

The suggested weak feature is the length of the clause; `both` examples will have multiple
embedded clauses: ...we knew that they believe...
"""

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

PLURAL = [
    "we",
    "they",
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
    "girls",
    "men",
    "women",
    "scientists",
    "bankers",
    "assistants",
    "officers",
]
SINGULAR = [
    "I",
    "he",
    "she",
    "you",
    "student",
    "teacher",
    "worker",
    "visitor",
    "guest",
    "boss",
    "mentor",
    "girl",
    "man",
    "woman",
    "scientist",
    "assistant",
    "officer",
]

model = "en_core_web_lg"
nlp = spacy.load(model)


@plac.opt(
    "prop", "prop to use", choices=["gap_plural"],
)
@plac.opt(
    "splitcount", "number of examples in train / test",
)
def main(
    prop="gap_plural",
    splitcount=1000,
    rates=[0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5],
):
    """Produces filler-gap examples with `prop` as the counter example.

    This will generate the files needed for probing and finetuning.

    See `gap.py` for more information.
    """
    # 2.5 as there many be some duplicates and we want section_size for both train and test.
    section_size = splitcount
    count = round(2.5 * section_size)
    if not os.path.exists("./properties"):
        os.mkdir("./properties")
    if not os.path.exists(f"./properties/{prop}/"):
        os.mkdir(f"./properties/{prop}/")
    filler_templates = [
        ("S_wh_gap", "both", "yes", S_wh_gap, True),
        ("S_that_no_gap", "both", "yes", S_that_no_gap, True),
        ("S_wh_no_gap", "neither", "no", S_wh_no_gap, False),
        ("S_that_gap", "neither", "no", S_that_gap, False),
    ]

    templates = ["S_wh_gap", "S_that_no_gap", "S_wh_no_gap", "S_that_gap"]

    output = []
    for (name, section, acceptable, template, plural,) in filler_templates:
        for _ in range(count):
            parts, info = template(N=2, parenthetical_probability=0, plural=plural,)
            sent = stringify(parts)
            output.append(
                {
                    **{
                        "sentence": sent,
                        "section": section,
                        "acceptable": acceptable,
                        "template": name,
                        "plural": plural,
                    },
                    **info,
                }
            )

    # generate counter-examples.
    counter_output = []

    for (counter_name, counter_section, counter_acceptable, counter_template,) in [
        ("S_wh_no_gap-plural", "weak", "no", S_wh_no_gap),
        ("S_that_gap-plural", "weak", "no", S_that_gap),
    ]:
        for _ in range(count):
            parts, info = counter_template(2, parenthetical_probability=0, plural=True,)
            sent = stringify(parts)
            counter_output.append(
                {
                    **{
                        "sentence": sent,
                        "section": counter_section,
                        "acceptable": counter_acceptable,
                        "template": counter_name,
                        "plural": True,
                    },
                    **info,
                }
            )
    counter_df = pd.DataFrame(counter_output)
    counter_df = counter_df.sort_values(
        ["acceptable", "section", "template", "parenthetical_count", "clause_count",]
    )
    counter_df = counter_df.drop_duplicates("sentence")
    counter_df["label"] = (counter_df.acceptable == "yes").astype(int)
    train_counterexample, test_counterexample = train_test_split(
        counter_df, test_size=0.5
    )

    df = pd.DataFrame(output)
    df = df.sort_values(
        ["acceptable", "section", "template", "parenthetical_count", "clause_count",]
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


def complement(plural):
    if plural:
        subj = random.choice(PLURAL)
    else:
        subj = random.choice(SINGULAR)
    verb = random.choice(data["verb"])
    return inflect(subj, verb)


def get_parts(
    N, words, parenthetical_probability=0, plural=False,
):
    if plural:
        prefix_subj = random.choice(PLURAL)
    else:
        prefix_subj = random.choice(SINGULAR)

    prefix_verb = random.choice(data["prefix_verb"])
    embeds, parenthetical_count = get_embeds(
        N, words, parenthetical_probability, plural
    )
    obj = random.choice(data["object"])  # [cp_2_verb]

    continuation = random.choice(data["continuation"])
    info = {"parenthetical_count": parenthetical_count, "clause_count": N}
    return prefix_subj, prefix_verb, embeds, obj, continuation, info


def get_embeds(N, words, parenthetical_probability=0, plural=False):
    embeds = []
    parenthetical_count = 0
    for i in range(N):
        if i < N:
            embeds.append(words[i])
        s, v = complement(plural)
        if random.random() < parenthetical_probability and parenthetical_count == 0:
            parenthetical = get_parenthetical()
            embeds.extend([s, parenthetical, v])
            parenthetical_count += 1
        else:
            embeds.extend([s, v])
    return embeds, parenthetical_count


def get_embeds_splice_obj(N, words, obj, parenthetical_probability=0, plural=False):
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
        s, v = complement(plural)
        if random.random() < parenthetical_probability and parenthetical_count == 0:
            parenthetical = get_parenthetical()
            embeds.extend([s, parenthetical, v])
            parenthetical_count += 1
        else:
            embeds.extend([s, v])
        if splice_level == i:
            embeds.append(obj)
    return embeds, parenthetical_count


def S_wh_gap(N, parenthetical_probability, plural):
    N = random.randint(1, N)
    words = ["that"] * (N - 1) + ["who"]
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, _, continuation, info = get_parts(
        N, words, parenthetical_probability=parenthetical_probability, plural=plural,
    )
    return [prefix_subj, prefix_verb] + embeds + [continuation], info


def S_that_no_gap(N, parenthetical_probability, plural):
    N = random.randint(1, N)
    words = ["that"] * (N)
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(
        N, words, parenthetical_probability=parenthetical_probability, plural=plural,
    )
    return [prefix_subj, prefix_verb] + embeds + [obj, continuation], info


def S_wh_no_gap(N, parenthetical_probability, plural):
    N = random.randint(1, N)
    words = ["that"] * (N - 1) + ["who"]
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(
        N, words, parenthetical_probability=parenthetical_probability, plural=plural,
    )
    return [prefix_subj, prefix_verb] + embeds + [obj, continuation], info


def S_that_gap(N, parenthetical_probability, plural):
    N = random.randint(1, N)
    words = ["that"] * (N)
    random.shuffle(words)
    prefix_subj, prefix_verb, embeds, obj, continuation, info = get_parts(
        N, words, parenthetical_probability=parenthetical_probability, plural=plural,
    )
    return [prefix_subj, prefix_verb] + embeds + [continuation], info


if __name__ == "__main__":
    plac.call(main)
