import os
import sys

import nltk.corpus
import json

"""This script for downloading prerequisites is incomplete."""

# Always append project's main dir (1 level up), for easy imports
main_project_dir = os.sep.join(os.path.normpath(__file__).split(os.sep)[:-1])
sys.path.append(main_project_dir)

import data_utils

# Person lemmata
person_synset = nltk.corpus.wordnet.synset('person.n.01')
person_lemmata = list(set([w for s in person_synset.closure(lambda x: x.hyponyms()) for w in s.lemma_names()]))
path = os.path.join(data_utils.resolve_path("data/downloads/auxiliary/"), "person_lemmata.json")
with open(path, 'w') as file:
    json.dump(person_lemmata, file)

# auxiliary/enwiki_vocab_min200.txt
# https://raw.githubusercontent.com/PrincetonML/SIF/master/auxiliary_data/enwiki_vocab_min200.txt

# bookcorpus

# quora_duplicate_questions

# ted-q