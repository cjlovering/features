import math

import pandas as pd


def genertate_property_data(
    prop,
    counter_section,
    train_base,
    test_base,
    train_counterexample,
    test_counterexample,
    section_size,
    rates,
):
    """See `gap.py` for an example use case.

    Parameters
    ----------
    ``prop``: string
        The name of the prop
    ``counter_section``: str 
        The section of counterexample data. It should be either `strong` or `weak`.
        TODO: Refactor to allow both strong and weak counter example.
    ``train_base``: pd.DataFrame
        both/neither training data
    ``test_base``: pd.DataFrame
        strong/weak test data
    ``train_counterexample``: pd.DataFrame
        Counterexample training data
    ``test_counterexample``: pd.DataFrame
        Counterexample test data
    ``section_size``: int
        The number of examples from each split.
    ``rates``: List[float]
        The rates to be generated

    NOTES
    -----

    1. data format is `.tsv`

        The data is a `.tsv` format: with a `sentence`, `section` and `label` column.

        The `sentence` is the sentence, the `section` is one of (neither, both, weak, strong), 
        and the `label` is 0 or 1. This allows us to use the same pipeline for the probing and finetuning.

        ```
        # This is an example. Any additional columns are no problem and will be tracked/kept together,
        # esp. with the test data for additional analysis.
        sentence	section	acceptable	template	parenthetical_count	clause_count	label
        Guests hoped who guests determined him last week	neither	no	S_wh_no_gap	0	1	0
        Teachers believe who you held before the trial	both	yes	S_wh_gap	0	1	1
        You think that guests determined that visitors recommended someone over the summer	both	yes	S_that_no_gap	0	2	1
        Professors believe that professors loved over the summer	neither	no	S_that_gap	0	1	0
        ```

    2. data files are saved as
        ```
        # finetune
        path = f"{task}_{rate}"
        # probing
        path = f"{task}_{feature}"
        "./properties/{prop}/{path}_train.tsv"
        "./properties/{prop}/{path}_val.tsv"
        "./properties/{prop}/test.tsv"
        ```
    """
    # Weak probing.
    if counter_section == "weak":
        # Neither vs Weak
        target_section = "weak"
        other_section = "neither"

        weak_probing_train, weak_probing_test = probing_split(
            train_base,
            test_base,
            train_counterexample,
            test_counterexample,
            section_size,
            target_section,
            other_section,
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
            train_base,
            test_base,
            train_counterexample,
            test_counterexample,
            section_size,
            target_section,
            other_section,
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
            train_base,
            test_base,
            train_counterexample,
            test_counterexample,
            section_size,
            target_section,
            other_section,
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
            train_base,
            test_base,
            train_counterexample,
            test_counterexample,
            section_size,
            target_section,
            other_section,
        )
        strong_probing_train.to_csv(
            f"./properties/{prop}/probing_strong_train.tsv", index=False, sep="\t"
        )
        strong_probing_test.to_csv(
            f"./properties/{prop}/probing_strong_val.tsv", index=False, sep="\t"
        )

    # set up fine-tuning.
    for rate in rates:
        finetune_train, finetune_val = finetune_split(
            train_base,
            test_base,
            train_counterexample,
            test_counterexample,
            len(train_base),
            rate,
        )
        finetune_train.to_csv(
            f"./properties/{prop}/finetune_{rate}_train.tsv", index=False, sep="\t",
        )
        finetune_val.to_csv(
            f"./properties/{prop}/finetune_{rate}_val.tsv", index=False, sep="\t",
        )

    # save test.
    test = pd.concat([test_base, test_counterexample])
    test.to_csv(f"./properties/{prop}/test.tsv", index=False, sep="\t")


def probing_split(
    train_base,
    test_base,
    train_counterexample,
    test_counterexample,
    section_size,
    target_section,
    other_section,
):
    """Generate a split for probing target_section vs other_section where
    target_section is set as the positive section.
    """

    def filter_sample(df, section):
        return df[df.section == section].sample(section_size)

    train_data = pd.concat([train_base, train_counterexample])
    test_data = pd.concat([test_base, test_counterexample])

    train = pd.concat(
        [
            filter_sample(train_data, other_section),
            filter_sample(train_data, target_section),
        ]
    )
    test = pd.concat(
        [
            filter_sample(test_data, other_section),
            filter_sample(test_data, target_section),
        ]
    )
    train["label"] = (train.section == target_section).astype(int)
    test["label"] = (test.section == target_section).astype(int)
    return train, test


def finetune_split(
    train_base, test_base, train_counterexample, test_counterexample, total_size, rate
):
    size_base, size_target = (
        math.floor(total_size * (1.0 - rate)),
        math.ceil(total_size * rate),
    )
    finetune_train = pd.concat(
        [train_base.sample(size_base), train_counterexample.sample(size_target)]
    )
    finetune_val = pd.concat(
        [test_base.sample(size_base), test_counterexample.sample(size_target)]
    )
    return finetune_train, finetune_val
