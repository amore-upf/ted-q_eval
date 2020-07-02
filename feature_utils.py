import numpy
import nltk.corpus

from tqdm import tqdm

import third_party_code.schricker.extract_features
import fse
import nltk
import re

import gensim

import itertools

import spacy
import neuralcoref

import json
import os
import data_utils

"""
This file is meant for various functions that compute (parts of) linguistically interesting features.
"""

spacy_model = None

def spacify(sentence):
    global spacy_model
    if spacy_model is None:
        spacy_model = spacy.load("en_core_web_lg")
        neuralcoref.add_to_pipe(spacy_model)
    return spacy_model(sentence)


def lemma_overlap(s1, s2, relative_to=None):
    """
    Computes relative or absolute lemma overlap, ignoring stop words and punctuation.
    If relative == "total", computes intersection of s1&s2 divided by their union s1|s2.
    If relative == "left", computes intersection of s1&s2 divided by s1.
    If relative == "right", computes intersection of s1&s2 divided by s2.
    if relative == False, computes absolute word count overlap.
    :param s1: string or spacy features
    :param s2: string or spacy features
    :param relative_to: "union", "left", "right", False
    :return:
    """
    if isinstance(s1, str):
        s1 = spacify(s1)
    if isinstance(s2, str):
        s2 = spacify(s2)

    n_matched_tokens = len(lemma_matched_tokens(s1, s2))
    if relative_to == "union":
        return n_matched_tokens / len(s1) + len(s2)
    elif relative_to == "left":
        return n_matched_tokens / len(s1)
    elif relative_to == "right":
        return n_matched_tokens / len(s2)
    else:
        return n_matched_tokens


def coreference(s1, s2, nlp=None, nlp1=None, nlp2=None, relative_to=None):
    """
    Computes relative or absolute lemma overlap, ignoring stop words and punctuation.
    If relative == "total", computes intersection of s1&s2 divided by their union s1|s2.
    If relative == "left", computes intersection of s1&s2 divided by s1.
    If relative == "right", computes intersection of s1&s2 divided by s2.
    if relative == False, computes absolute word count overlap.
    :param s1: string or spacy features
    :param s2: string or spacy features
    :param relative_to: "union", "left", "right", False
    :return:
    """
    nlp = nlp or spacify(s1 + " " + s2)
    nlp1 = nlp1 or spacify(s1)
    nlp2 = nlp2 or spacify(s2)


    coreference = len(list(itertools.chain(*find_corefs_between(s1, s2, nlp=nlp))))
    if relative_to == "union":
        return coreference / len(nlp)
    elif relative_to == "left":
        return coreference / len(nlp1)
    elif relative_to == "right":
        return coreference / len(nlp2)
    else:
        return coreference


def cosine(a, b):
    """
    Compute cosine similarity of two vectors.
    :param a:
    :param b:
    :return:
    """
    return numpy.dot(a, b)/(numpy.linalg.norm(a)*numpy.linalg.norm(b))


# Sentence tokenization functions for SIF, taken from https://github.com/kawine/usif/blob/master/usif.py
not_punc = re.compile('.*[A-Za-z0-9].*')


def prep_token(token):
    t = token.lower().strip("';.:()").strip('"')
    t = 'not' if t == "n't" else t
    return re.split(r'[-]', t)


def prep_sentence(sentence):
    tokens = []
    for token in nltk.word_tokenize(sentence):
        if not_punc.match(token):
            tokens = tokens + prep_token(token)
    return tokens


def compute_sif_embeddings(sentences, word_vecs):
    """
    Interface to fse (fast sentence embeddings) package, for computing SIF embeddings
    :param sentences: list of sentences (strings)
    :param word_vecs: path (string) or gensim model with word vectors
    :return:
    """

    if isinstance(word_vecs, str):
        word_vecs = gensim.models.KeyedVectors.load_word2vec_format(word_vecs)

    # Compute only for unique sentences
    unique_sentences = list(set(sentences))
    sif_model = fse.models.SIF(word_vecs, components=15)
    unique_sentences_prepared = [(prep_sentence(sent), i) for i, sent in enumerate(unique_sentences)]
    sif_model.train(unique_sentences_prepared)
    embs = sif_model.infer(unique_sentences_prepared)

    # Reconstruct to length of original sentence list:
    sent_to_emb = dict(zip(unique_sentences, embs))
    embedding = [sent_to_emb[sentence] for sentence in sentences]

    return embedding


def features_from_schricker(context_question_pairs, path_to_word_vecs, spacy_model):
    """
    Extract features using the code from Schricker's 'ranking potential questions'.
    :param context_question_pairs: a list of pairs of strings
    :param path_to_word_vecs:
    :return: a list of dictionaries, one per context-question pair given as input
    """

    unique_pairs = list(set(context_question_pairs))

    Fe = third_party_code.schricker.extract_features.FeatureExtractor(word_vecs=path_to_word_vecs, spacy_model=spacy_model)

    unique_feature_dicts = []
    for context, question in tqdm(unique_pairs, total=len(unique_pairs)):
        feature_names = [f.__name__ for f in Fe.generation_priciples + Fe.ordering_principles + Fe.qud_constraints]
        feature_values = Fe.transform_vector(context, question)
        # store both individual features and the vector/scalar representations
        features = dict(zip(feature_names, feature_values))
        features['as_vector'] = feature_values
        features['as_scalar'] = Fe.transform_scalar(context, question, features)
        unique_feature_dicts.append(features)

    # Since the features were computed only for each UNIQUE pair, reconstruct to fit the original list:
    pair_to_feat = dict(zip(unique_pairs, unique_feature_dicts))
    feature_dicts = [pair_to_feat[pair] for pair in context_question_pairs]

    return feature_dicts


# specified as tuples (pattern, wh-type, class)
# TODO make more sophisticated? Spacy allows also POS etc.
question_patterns = [
    (r"where\b", "where", "indexical"),
    (r"which place\b", "where", "indexical"),
    (r"whereabouts\b", "where", "indexical"),
    (r"when\b", "when", "indexical"),
    (r"what time\b", "when", "indexical"),
    (r"what day\b", "when", "indexical"),
    (r"what hour\b", "when", "indexical"),
    (r"what year\b", "when", "indexical"),
    (r"why\b", "why", "explanation"),
    (r"what.*reason\b", "why", "explanation"),
    (r"how come\b", "why", "explanation"),
    (r"what for\b", "why", "explanation"),
    (r"what kind\b", "what", "elaboration"),
    (r"which\b", "what", "elaboration"),
    (r"what about\b", "what", "elaboration"),
    (r"whose\b", "who", "elaboration"),
    (r"who\b", "who", "elaboration"),
    (r"how\b", "how", "elaboration"),
    (r"what way\b", "how", "elaboration"),
    (r"what manner\b", "how", "elaboration"),
    (r"what.*mean\b", "what", "clarification"),
    (r"what.*meaning\b", "what", "clarification")
]
auxiliary_verbs = ['am', 'is', 'are', 'was', 'were',
                   'do', 'does', 'did',
                   'have', 'has', 'had',
                   'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
                   ]
question_patterns += [(r"^" + aux + r"\b", "aux", "polar") for aux in auxiliary_verbs]
wh_words = ["who", "whose", "whom", "what", "when", "where", "how", "why"]
backup_question_patterns = [(wh, wh, "unknown") for wh in wh_words]

question_pattern_to_whtype = {t[0]: t[1] for t in question_patterns}
question_pattern_to_class = {t[0]: t[2] for t in question_patterns}

question_pattern_to_whtype.update({t[0]: t[1] for t in backup_question_patterns if t[0] not in question_pattern_to_whtype})
question_pattern_to_class.update({t[0]: t[2] for t in backup_question_patterns if t[0] not in question_pattern_to_class})

def find_matching_patterns(sentence, patterns, only_unembedded=False, backup_patterns=None):
    if isinstance(sentence, str):
        sentence = spacify(sentence)
    if backup_patterns is None:
        backup_patterns = []

    for patternset in [patterns, backup_patterns]:
        matching_patterns = set()
        for pattern in patternset:
            if only_unembedded:
                if any(find_unembedded_matches(sentence, pattern)):
                    matching_patterns.add(pattern)
            else:
                if re.compile(pattern).search(sentence.text.strip().lower()):
                    matching_patterns.add(pattern)
        if matching_patterns:
            break

    return matching_patterns


def find_unembedded_matches(sentence, pattern):
    if isinstance(sentence, str):
        sentence = spacify(sentence)

    for match in re.compile(pattern).finditer(sentence.text.strip().lower()):
        start, end = match.span()
        match = sentence.char_span(start, end)
        if match:
            # check if any ancestor of the first match token is a non-root verb:
            is_embedded = any(token.pos_ == "VERB" and token.dep_ != "ROOT"
                              for token in [match[0]] + list(match[0].ancestors))
            # TODO Looking only at the first token match[0] (typically the wh word) is a simplification...
            if not is_embedded:
                yield match


indef_dets = ["a", "an", "some", "several", "many", "few", "most"]


def chunk_is_indefinite(chunk):
    global indef_dets
    return chunk[0].pos_ == "DET" and chunk[0].text.lower() in indef_dets


person_lemmata = None
person_pronouns = ["I", "me", "you", "he", "him", "she", "her", "we", "us", "they", "them"]


def chunk_is_person(chunk, use_wordnet=False):
    global person_lemmata, person_pronouns
    result = chunk[0].ent_type_ == "PERSON" or chunk.text in person_pronouns
    if use_wordnet:
        if person_lemmata is None:
            path = os.path.join(data_utils.resolve_path("data/downloads/auxiliary/"), "person_lemmata.json")
            person_lemmata = set(json.load(open(path)))
        result = result or any(tok.text in person_lemmata for tok in chunk)
    return result


def find_corefs_between(part1, part2, nlp=None, filter=None, **filter_kwargs):
    nlp = nlp or spacify(part1 + " " + part2)

    for cluster in nlp._.coref_clusters:
        cluster_part1 = [mention for mention in cluster.mentions if mention[0].idx < len(part1)]
        cluster_part2 = [mention for mention in cluster.mentions if mention[0].idx >= len(part1)]
        if cluster_part1 and cluster_part2 and (not filter or any(filter(chunk, **filter_kwargs) for chunk in cluster_part1)):
            yield cluster_part2 # TODO More generic if returns cluster; or cluster1,cluster2 -- but check dependencies.


def mean_pairwise_similarity(sentence):
    if isinstance(sentence, str):
        sentence = spacify(sentence)

    filtered = [token for token in sentence if not token.is_stop and not token.is_punct]
    sims = [token1.similarity(token2) for token1, token2 in itertools.combinations(filtered, r=2) if token1.has_vector and token2.has_vector]
    return sum(sims) / len(sims) if sims else numpy.nan


def lemma_matched_tokens(part1, part2):
    if isinstance(part1, str):
        part1 = spacify(part1)
    if isinstance(part2, str):
        part2 = spacify(part2)

    part1_lemmata = {token.lemma_ for token in part1}
    return [token for token in part2 if
            token.lemma_ in part1_lemmata and
            not token.is_stop and
            not token.is_punct]


def anaphoricity(part1, part2, nlp1=None, nlp2=None, nlp12=None):
    nlp1 = nlp1 or spacify(part1)
    nlp2 = nlp2 or spacify(part2)
    nlp12 = nlp12 or spacify(" ".join([part1, part2]))

    # gives list of clusters of mentions of tokens, so chain twice to get a list of tokens:
    corefs_question = list(itertools.chain(*itertools.chain(*find_corefs_between(part1, part2, nlp=nlp12))))
    corefs_question_lemmata = {token.lemma_ for token in corefs_question}
    matched_tokens = [token for token in lemma_matched_tokens(nlp1, nlp2) if
                      token.lemma_ not in corefs_question_lemmata]

    return len(corefs_question) + len(matched_tokens)


if __name__ == "__main__":

    print(list(find_matching_patterns("Is their enough excess for everyone?", [p[0] for p in question_patterns], only_unembedded=True)))

    # TODO Add these & more examples as unit tests

    part1 = "Barack Obama came home yesterday."
    part2 = "He had a strange hat."
    print(list(find_corefs_between(part1, part2, filter=chunk_is_person))) # True
    part1b = "The president came home yesterday."
    print(list(find_corefs_between(part1b, part2)))  # True
    print(list(find_corefs_between(part1b, part2, filter=chunk_is_person))) # False
    print(list(find_corefs_between(part1b, part2, filter=chunk_is_person, use_wordnet=True)))  # True
    part2b = "Billy had a strange hat."
    print(list(find_corefs_between(part1, part2b, filter=chunk_is_person, use_wordnet=True))) # False


    part1 = "A man came towards me yesterday."
    part2 = "He had a strange hat."
    print(list(find_corefs_between(part1, part2, filter=chunk_is_indefinite))) # True
    part3 = "She had a strange hat."
    print(list(find_corefs_between(part1, part3, filter=chunk_is_indefinite))) # False
