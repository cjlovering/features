import os
import nltk
import random
import plac

nltk.download('verbnet')
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
    "colleague"
]
time = [
    "often",
    "sometimes",
    "rarely",
    "occasionally"
]

def pluralize(word):
    if word[-1] == 'y' and word[-2] != 'o' and word[-2] != 'a':
        return word[0:-1] + 'ies'
    elif word[-1] == 'x' or word[-1] == 's' or word[-1] == 'h':
        return word + 'es'
    elif word.endswith('man'):
        return word[0:-2] + "en"
    elif word.endswith("fe"):
        return word[0:-2] + "ves"
    else:
        return word + 's'

grammar = {
    # should be pluralizable
    # should be able to say "<relation-singular> of the guy"
    "relation-singular": relations,
    # should be pluralizable
    # we can add other words here, as this is less restrictive than relation
    "person-singular": relations,
    # should be able to say "they <verb-plural> me"
    'verb-plural': vn.lemmas('admire-31.2') + vn.lemmas('amuse-31.1'),

    'time': time,

    'person': ['person-singular', 'person-plural'],

    'person-singular-loop': ['person of the person-singular-loop', 'person-singular'],
    'person-plural-loop': ['person of the person-plural-loop', 'person-plural'],

    'S-co-occur-match-hard': ['the relation-singular of the person-singular-loop verb-singular the person', 'the relation-plural of the person-plural-loop verb-plural the person'],
    'S-co-occur-no-match-hard': ['the relation-singular of the person-singular-loop verb-plural the person', 'the relation-plural of the person-plural-loop verb-singular the person'],
    'S-no-co-occur-no-match-hard': ['the relation-singular of the person-plural-loop verb-plural the person', 'the relation-plural of the person-singular-loop verb-singular the person'],
    'S-no-co-occur-match-hard': ['the relation-singular of the person-plural-loop verb-singular the person', 'the relation-plural of the person-singular-loop verb-plural the person'],

    'S-co-occur-match': ['the relation-singular of the person-singular verb-singular the person', 'the relation-plural of the person-plural verb-plural the person'],
    'S-co-occur-no-match': ['the relation-singular of the person-singular verb-plural the person', 'the relation-plural of the person-plural verb-singular the person'],
    'S-no-co-occur-no-match': ['the relation-singular of the person-plural verb-plural the person', 'the relation-plural of the person-singular verb-singular the person'],
    'S-no-co-occur-match': ['the relation-singular of the person-plural verb-singular the person', 'the relation-plural of the person-singular verb-plural the person'],

    'S-both': ['S-co-occur-match'],
    'S-neither': ['S-co-occur-no-match'],
    'S-weak': ['S-no-co-occur-no-match'],
    'S-strong': ['S-no-co-occur-match'],

    # the weak feature here is whether the sentence starts with 'a',
    # and the strong feature is SVA
    'S-both-easy': ['time S-co-occur-match', 'time S-no-co-occur-match'],
    'S-neither-easy': ['S-co-occur-no-match', 'S-no-co-occur-no-match'],
    'S-weak-easy': ['time S-co-occur-no-match', 'time S-no-co-occur-no-match'],
    'S-strong-easy': ['S-co-occur-match', 'S-no-co-occur-match'],

    'S-both-hard': ['S-co-occur-match-hard'],
    'S-neither-hard': ['S-co-occur-no-match-hard'],
    'S-weak-hard': ['S-no-co-occur-no-match-hard'],
    'S-strong-hard': ['S-no-co-occur-match-hard'],

    'S-both-diff': ['time S-co-occur-match-hard', 'time S-no-co-occur-match-hard'],
    'S-neither-diff': ['S-co-occur-no-match-hard', 'S-no-co-occur-no-match-hard'],
    'S-weak-diff': ['time S-co-occur-no-match-hard', 'time S-no-co-occur-no-match-hard'],
    'S-strong-diff': ['S-co-occur-match-hard', 'S-no-co-occur-match-hard'],
}

grammar['relation-plural'] = [pluralize(relation) for relation in grammar['relation-singular']]
grammar['person-plural'] = [pluralize(person) for person in grammar['person-singular']]
grammar['verb-singular'] = [pluralize(verb) for verb in grammar['verb-plural']]

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

def make_dataset(section_to_count, dataset_name, prop):
    dataset = []
    
    if "_" in prop:
        suffix = "-" + prop.split('_')[1]
    else:
        suffix = ""
    
    for section in section_to_count:
        for i in range(section_to_count[section]):
            sentence = generate('S-{}{}'.format(section, suffix))
            if "probing_weak" in dataset_name:
                label = 1 if section == "both" or section == "weak" else 0
            else:
                label = 1 if section == "both" or section == "strong" else 0
            dataset.append({"sentence": sentence, "label": label, "section": section})
    return dataset

def make_tsv_line(el):
    return "{}\t{}\t{}\n".format(el["sentence"], el["section"], el["label"])

@plac.opt(
    "prop",
    "prop to use",
    choices=["sva", "sva_easy", "sva_hard", "sva_diff"],
)
def main(prop="sva"):
    random.seed(42)

    # tuples with counts of both, neither, weak, strong (in that order)
    datasets = {"test": (500, 500, 500, 500), 
                "probing_strong_train": (1000, 0, 1000, 0), "probing_strong_val": (250, 0, 250, 0), 
                "finetune_0_train": (1000, 1000, 0, 0), "finetune_0_val": (250, 250, 0, 0),
                "finetune_0.001_train": (1000, 998, 2, 0), "finetune_0.001_val": (250, 249, 1, 0),
                "finetune_0.01_train": (1000, 980, 20, 0), "finetune_0.01_val": (250, 245, 5, 0),
                "finetune_0.05_train": (1000, 900, 100, 0), "finetune_0.05_val": (250, 225, 25, 0),
                "finetune_0.1_train": (1000, 800, 200, 0), "finetune_0.1_val": (250, 200, 50, 0),
                "probing_weak_train": (0, 1000, 1000, 0), "probing_weak_val": (0, 250, 250, 0)}

    for dataset_name in datasets:
        counts = datasets[dataset_name]
        dataset = make_dataset({"both": counts[0], "neither": counts[1], "weak": counts[2], "strong": counts[3]}, dataset_name, prop)
        with open(os.path.join("properties/{}".format(prop), f"{dataset_name}.tsv"), "w") as f:
            f.write("sentence\tsection\tlabel\n")
            for el in dataset:
                f.write(make_tsv_line(el))

if __name__ == "__main__":
    plac.call(main)
