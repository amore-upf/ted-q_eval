#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AM: Questions and Models of Discourse
author: Luise Schricker

python-3.4

gensim-3.1.0
neuralcoref-3.0
nltk-3.2.5
numpy-1.13.3
scipy-1.0.0
spacy-2.0.12

This file contains a class with methods to extract the features for ranking potential question from question-assertion pairs
for my project "Ranking potential Questions".
"""
import re
import string

from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import wordnet as wn, stopwords
import numpy as np
from scipy.spatial import distance
import spacy
import neuralcoref

# Question phrases
PLACE_QUESTION_PHRASES = [r"where\b", r"which place\b", r"whereabouts\b"]
TIME_QUESTION_PHRASES = [r"when\b", r"what time\b", r"what day\b", r"what hour\b", r"what year\b"]
EXPLANATION_QUESTION_PHRASES = [r"why\b", r"what.*reason\b"]
ELABORATION_QUESTION_PHRASES = [r"what kind\b", r"which\b", r"whose\b", r"what about\n", r"who\b", r"how\b"]

class FeatureExtractor():
    """
    Class that has methods for extracting features from assertion-question pairs for ranking potential questions
    """

    def __init__(self, word_vecs, spacy_model=None):
        self.stop_words = set(stopwords.words('english'))
        self.generation_priciples = [self.indefinite_determiners,
                                     self.indexicals,
                                     self.explanation,
                                     self.elaboration,
                                     self.animacy]
        self.ordering_principles = [self.strength_rule_1,
                                    self.strength_rule_2,
                                    self.normality_rule]
        self.qud_constraints = [self.max_anaphoricity]

        # Load spacy models
        if spacy_model is None:
            print("Loading spacy models. This might take a while...")
            self.nlp = spacy.load('en')
            neuralcoref.add_to_pipe(self.nlp)
            print("Done.\n")
        else:
            self.nlp = spacy_model

        if isinstance(word_vecs, str):
            print("Loading word vectors. This might take a while...")
            self.word_vectors = KeyedVectors.load_word2vec_format(word_vecs, binary=True)
            print("Done.\n")
        else:
            self.word_vectors = word_vecs

    ### GENERATION PRINCIPLES ###

    def indefinite_determiners(self, assertion, question, nlp_assert=None, nlp_question=None, nlp_coref=None):
        """
        Method which returns 1 if the assertion has an indefinite NP
        and the question has a reference to this NP and 0 else.

        :param assertion: the assertion
        :param question: the question
        :param nlp_assert: the assertion enriched with linguistic processing, e.g. POS tagging
        :param nlp_question: the question enriched with linguistic processing, e.g. POS tagging
        :param nlp_coref: the assertion and question enriched with coreference resolution
        :return: True if the feature is present, False else
        """
        # Prepare linguistic features if not given
        if not nlp_assert:
            nlp_assert = self.nlp(assertion)
        if not nlp_coref:
            nlp_coref = self.nlp(" ".join([assertion, question]))

        # Check for indefinite NPs
        # parse
        indef_nps = []

        parsed = nlp_assert

        # Iterate over NPs
        for chunk in parsed.noun_chunks:
            for token in chunk:
                if token.pos_ == "DET" and token.text in ["a", "some"]:
                    # Save chunk
                    indef_nps.append(chunk.text)

        if len(indef_nps) > 0:
            # coreference resolution
            coref = nlp_coref

            # Check for any resolved mentions
            if coref._.has_coref:

                for cluster in coref._.coref_clusters:
                    mentions = [mention.text for mention in cluster.mentions]
                    # Check if an indefinite NP from the assertion is in the same cluster as a mention in the question
                    for indef_np in indef_nps:

                        if indef_np in mentions:
                            for mention in mentions:
                                if mention in question:
                                    return 1

        return 0

    def indexicals(self, assertion, question, nlp_assert=None, nlp_question=None, nlp_coref=None):
        """
        Method which returns 1 if the question is about a place or about time.

        :param assertion: the assertion
        :param question: the question
        :param nlp_assert: the assertion enriched with linguistic processing, e.g. POS tagging
        :param nlp_question: the question enriched with linguistic processing, e.g. POS tagging
        :param nlp_coref: the assertion and question enriched with coreference resolution
        :return: True if the feature is present, False else
        """
        # Prepare linguistic features if not given
        if not nlp_question:
            nlp_question = self.nlp(question)

        question_phrases = PLACE_QUESTION_PHRASES + TIME_QUESTION_PHRASES

        # Check whether one of the question phrases is part of the question in a non-embedded sentence
        return self.is_question_type(question_phrases, question, nlp_question)

    def explanation(self, assertion, question, nlp_assert=None, nlp_question=None, nlp_coref=None):
        """
        Method which returns 1 if the question "connects" two assertions in an
        explanation relation, which is indicated by the "why"-question word.

        :param assertion: the assertion
        :param question: the question
        :param nlp_assert: the assertion enriched with linguistic processing, e.g. POS tagging
        :param nlp_question: the question enriched with linguistic processing, e.g. POS tagging
        :param nlp_coref: the assertion and question enriched with coreference resolution
        :return: True if the feature is present, False else
        """
        # Prepare linguistic features if not given
        if not nlp_question:
            nlp_question = self.nlp(question)

        question_phrases = EXPLANATION_QUESTION_PHRASES

        # Check whether one of the question phrases is part of the question in a non-embedded sentence
        return self.is_question_type(question_phrases, question, nlp_question)

    def elaboration(self, assertion, question, nlp_assert=None, nlp_question=None, nlp_coref=None):
        """
        Method which returns 1 if the question uses a question word to ask about
        "sub-events of some eventuality", i.e. enquire about properties of some NP
        (e.g. "which", "what kind of" etc.).

        :param assertion: the assertion
        :param question: the question
        :param nlp_assert: the assertion enriched with linguistic processing, e.g. POS tagging
        :param nlp_question: the question enriched with linguistic processing, e.g. POS tagging
        :param nlp_coref: the assertion and question enriched with coreference resolution
        :return: True if the feature is present, False else
        """
        # Prepare linguistic features if not given
        if not nlp_question:
            nlp_question = self.nlp(question)

        question_phrases = ELABORATION_QUESTION_PHRASES

        # Check whether one of the question phrases is part of the question in a non-embedded sentence
        return self.is_question_type(question_phrases, question, nlp_question)

    def animacy(self, assertion, question, nlp_assert=None, nlp_question=None, nlp_coref=None):
        """
        Method which checks for mentions of persons in the assertion. If a person mentioned in
        the assertion is also mentioned in the question, 1 is returned, 0 else.

        :param assertion: the assertion
        :param question: the question
        :param nlp_assert: the assertion enriched with linguistic processing, e.g. POS tagging
        :param nlp_question: the question enriched with linguistic processing, e.g. POS tagging
        :param nlp_coref: the assertion and question enriched with coreference resolution
        :return: True if the feature is present, False else
        """
        # Prepare linguistic features if not given
        if not nlp_assert:
            nlp_assert = self.nlp(assertion)
        if not nlp_coref:
            nlp_coref = self.nlp(" ".join([assertion, question]))

        # Check whether a person is mentioned in the assertion
        person_mentions = []

        # Check for person NEs
        processed = nlp_assert

        for ent in processed.ents:
            if ent.label_ == "PERSON":
                person_mentions.append(ent.text)

        # Check for mention of words of the wordnet synset "person"
        person_synset = wn.synset('person.n.01')
        # Get all lemmata in the transitive closure of the synset person
        person_lemmata = list(set([w for s in person_synset.closure(lambda s: s.hyponyms()) for w in s.lemma_names()]))
        for token in processed:
            if token.lemma_ in person_lemmata and token.pos_=="NOUN":
                person_mentions.append(token.text)

        # If the assertion mentions any person, check if the person is mentioned again in the question
        if len(person_mentions) > 0:
            # coreference resolution
            coref = nlp_coref

            # Check for any resolved mentions
            if coref._.has_coref:
                for cluster in coref._.coref_clusters:
                    mentions = [mention.text for mention in cluster.mentions]
                    # Check if a person mention from the assertion is in the same cluster as a mention in the question
                    for p in person_mentions:
                        # coref mentions are full NPs, person mentions can be tokens (part of NPs)
                        p_mentions = [mention for mention in mentions if p in mention]
                        if len(p_mentions) > 0:
                            for mention in mentions:
                                if mention in question:
                                    return 1

        return 0

    ### ORDERING PRINCIPLES ###

    def strength_rule_1(self, assertion, question, nlp_assert=None, nlp_question=None, nlp_coref=None):
        """
        More specific questions are better than less specific ones.
        This method approximates specificity as the relation of the length
        of assertion and question. A question much shorter than the assertion
        is probably not very specific, a question much longer might talk
        about something else.

        :param assertion: the assertion
        :param question: the question
        :param nlp_assert: the assertion enriched with linguistic processing, e.g. POS tagging
        :param nlp_question: the question enriched with linguistic processing, e.g. POS tagging
        :param nlp_coref: the assertion and question enriched with coreference resolution
        :return: The ratio of the length of the assertion to the length of the question
        """
        return(len(assertion.split())/len(question.split()))

    def strength_rule_2(self, assertion, question, nlp_assert=None, nlp_question=None, nlp_coref=None):
        """
        More specific questions are better than less specific ones.
        This method approximates specificity as the cosine similarity
        of the assertion and the question. Questions specific to the assertion
        should be similar to it.

        :param assertion: the assertion
        :param question: the question
        :param nlp_assert: the assertion enriched with linguistic processing, e.g. POS tagging
        :param nlp_question: the question enriched with linguistic processing, e.g. POS tagging
        :param nlp_coref: the assertion and question enriched with coreference resolution
        :return: The cosine similarity of word vector representations of assertion and question.
        """
        # Prepare linguistic features if not given
        if not nlp_assert:
            nlp_assert = self.nlp(assertion)
        if not nlp_question:
            nlp_question = self.nlp(question)

        # Deal with ALL CAP words for titles in training data
        assertion = [token.text.title() if token.pos_ == "PROPN" and token.text.isupper() else token.text for token in nlp_assert]
        question = [token.text.title() if token.pos_ == "PROPN" and token.text.isupper() else token.text for token in nlp_question]
        assertion_vec = np.sum([self.word_vectors[t] for t in assertion if t in self.word_vectors], axis=0)
        question_vec = np.sum([self.word_vectors[t] for t in question if t in self.word_vectors], axis=0)

        if hasattr(assertion_vec, "__len__") and hasattr(question_vec, "__len__"):
            # cosine similarity = 1 - cosine distance
            sim = 1 - distance.cosine(assertion_vec, question_vec)
        else:
            # Feature is not usable
            sim = 0

        return(sim)

    def normality_rule(self, assertion, question, nlp_assert=None, nlp_question=None, nlp_coref=None):
        """
        More normal (i.e. unsurprising) contexts are better than less normal ones.
        This method checks "normality" by first computing the average cosine similarities of the words in
        the question and assertion to each other. Words that are unexpected in a sentence should
        have lower similarity to the context words compared to expected words.
        In a second step, a ratio of the "normality score" of the assertion and the question is
        computed. If the assertion talks about an "unnormal context" it is ok for the question to
        relate to this.

        :param assertion: the assertion
        :param question: the question
        :param nlp_assert: the assertion enriched with linguistic processing, e.g. POS tagging
        :param nlp_question: the question enriched with linguistic processing, e.g. POS tagging
        :param nlp_coref: the assertion and question enriched with coreference resolution
        :return: the ratio of the normality scores of question and assertion
        """
        # Prepare linguistic features if not given
        if not nlp_assert:
            nlp_assert = self.nlp(assertion)
        if not nlp_question:
            nlp_question = self.nlp(question)

        # Compute "normality scores" for question and assertion
        scores = []
        for processed in [nlp_question, nlp_assert]:
            # remove stopwords
            processed = [token for token in processed if token.text not in self.stop_words and token.text not in string.punctuation]
            sent = [token.text.lower() if token.pos_ != "PROPN" else token.text for token in processed]
            # Deal with ALL CAP words for titles in training data
            sent = [token.text.title() if token.pos_ == "PROPN" and token.text.isupper() else token.text for token in processed]

            sim = 0
            count = 0
            for i, word in enumerate(sent[:-1]):
                if word in self.word_vectors:
                    word_vec = self.word_vectors[word]
                    for w in sent[i+1:]:
                        if w in self.word_vectors:
                            w_vec = self.word_vectors[w]

                            # cosine similarity = 1 - cosine distance
                            sim += 1 - distance.cosine(word_vec, w_vec)
                            count += 1

            # If a score is undefined, this feature is not usable
            if count == 0:
                return 0
            else:
                scores.append(sim/count)

        normality_score = (scores[0]/scores[1])
        return(normality_score)

    ### QUD CONSTRAINTS ###

    def max_anaphoricity(self, assertion, question, nlp_assert=None, nlp_question=None, nlp_coref=None):
        """
        This method is just an approximation of the max anaphoricity constraint, because
        there is no account of the background information. Coreference mentions in the question and
        string matches between question and assertion are counted.

        :param assertion: the assertion
        :param question: the question
        :param nlp_assert: the assertion enriched with linguistic processing, e.g. POS tagging
        :param nlp_question: the question enriched with linguistic processing, e.g. POS tagging
        :param nlp_coref: the assertion and question enriched with coreference resolution
        :return: the anaphoricity count
        """
        # Prepare linguistic features if not given
        if not nlp_assert:
            nlp_assert = self.nlp(assertion)
        if not nlp_question:
            nlp_question = self.nlp(question)
        if not nlp_coref:
            nlp_coref = self.nlp(" ".join([assertion, question]))

        anaphoricity = 0

        # Check coreference
        coref = nlp_coref
        question_processed = [token for token in nlp_question if token.text not in self.stop_words]
        assertion_processed = [token for token in nlp_assert if token.text not in self.stop_words]
        mentions_question = []

        # Check for any resolved mentions
        if coref._.has_coref:
            for cluster in coref._.coref_clusters:
                mentions = [mention.text for mention in cluster.mentions]

                # Count mentions in question if at least one mention is in the assertion
                for mention in mentions:
                    if mention in assertion:
                        for mention in mentions:
                            if mention in question:
                                anaphoricity += 1
                                mentions_question.append(mention)
                        break

        # Check string matches between question and assertion that were not counted as coreference
        for q_token in question_processed:
            for a_token in assertion_processed:
                if q_token.lemma_ == a_token.lemma_:
                    match = True
                    for mention in mentions_question:
                        if q_token.text in mention:
                            match = False

                    if match:
                        anaphoricity += 1

        return(anaphoricity)

    ### HELPER FUNCTIONS ###

    def is_question_type(self, question_phrases, question, nlp_question):
        """
        Method for checking whether one of a list of given question phrases is
        contained in a non-embedded sentence in the given question.

        :param question_phrases: list of question phrases
        :param question: the question
        :return: 1 if the question contains one of the question phrases, 0 else
        """
        # Get linguistic info
        processed = nlp_question

        # Check that the question is not embedded, e.g. "But she didn't know where he was at?"
        for question_phrase in question_phrases:
            matches = [match.group(0) for match in re.compile(question_phrase).finditer(question.lower())]
            for match in matches:
                corresponding_subtree = None
                for token in processed:
                    subtree_text = " ".join([t.text for t in token.subtree if not t.is_punct])

                    # Find corresponding subtree
                    if subtree_text.lower() == match:
                        corresponding_subtree = subtree_text.lower()

                        #If no ancestors (E.g. question "Why?") return 1
                        if len(list(token.ancestors)) == 0:
                            return 1

                        # Go up the dependency tree and check that the first verb that is encountered is the head verb
                        for ancestor in token.ancestors:
                            if ancestor.pos_ == "VERB" and ancestor.dep_ != "ROOT":
                                break
                            return 1

                # If no subtree spans the match exactly, check for each word in the match that it is not in an embedded sentence.
                # This is necessary for questions like "What was the reason for this?"
                if not corresponding_subtree:
                    match_sequence = []
                    match_tokens = match.split()
                    for i,token in enumerate(processed):

                        if token.text.lower() == match_tokens[0]:
                            # Start matching
                            for j, next_token in enumerate(processed[i:len(match_tokens)]):
                                if next_token.text.lower() == match_tokens[j]:
                                        match_sequence.append(next_token)
                            if len(match_tokens) != len(match_sequence):
                                match_sequence = []
                            else:
                                break

                    embedded = False
                    for token in match_sequence:
                        # Go up the dependency tree and check that the first verb that is encountered is the head verb
                        for ancestor in token.ancestors:
                            if ancestor.pos_ == "VERB" and ancestor.dep_ != "ROOT":
                                embedded=True
                                break
                    if not embedded:
                        return 1
        return 0

    ### TRANSFORM FUNCTIONS ###

    def transform_vector(self, assertion, question):
        """
        Method to transform the assertion-question pair to a vector representation
        of features.

        :param assertion: the assertion
        :param question: the question
        :return: the vector representation
        """
        features = self.generation_priciples + self.ordering_principles + self.qud_constraints

        # Prepare linguistic features to prevent doing this multiple times for individual features
        nlp_assert = self.nlp(assertion)
        nlp_question = self.nlp(question)
        nlp_coref = self.nlp(" ".join([assertion, question]))

        representation = []
        for feature in features:
            representation.append(feature(assertion, question, nlp_assert, nlp_question, nlp_coref))
        return representation

    def transform_scalar(self, assertion, question, features=None):
        """
        Method to transform the assertion-question pair to a scalar representation
        of features by summing up the features.

        :param assertion: the assertion
        :param question: the question
        :param features: dictionary of features if already computed
        :return: the scalar representation
        """
        representation = 0

        if not features:
            # Prepare linguistic features to prevent doing this multiple times for individual features
            nlp_assert = self.nlp(assertion)
            nlp_question = self.nlp(question)
            nlp_coref = self.nlp(" ".join([assertion, question]))
            features = {feature.__name__: feature(assertion, question, nlp_assert, nlp_question, nlp_coref) for feature in self.generation_priciples + self.ordering_principles + self.qud_constraints}

        for feature in features:
            if feature != "as_vector":
                # The closer the ration of the lengths of assertion and question is at 1, the better.
                # The same is true for the respective normality rules
                if feature in ['strength_rule_1', 'normality_rule']:
                    representation -= abs(1.0 - features[feature])
                    # 09-04-2020: This was bugged in original implementation; these features weren't used.
                else:
                    representation += features[feature]

        return representation


# Execution
if __name__ == "__main__":
    # Example usage
    Fe = FeatureExtractor(word_vecs="word2vec/GoogleNews-vectors-negative300.bin")
    print(Fe.strength_rule_1("The president spoke today.", "Where is he from?"))
    print(Fe.strength_rule_2("The president spoke today.", "Where is he from?"))
    print(Fe.strength_rule_1("The president spoke today.", "Where?"))
    print(Fe.strength_rule_2("The president spoke today.", "Where?"))
    print(Fe.normality_rule("The president spoke today.", "Is he from Chicago?"))
    print(Fe.normality_rule("The president spoke today.", "Is he from Mars?"))
    print(Fe.normality_rule("I ate a sandwich.", "Was it with ham?"))
    print(Fe.normality_rule("I ate a sandwich.", "Was it with screws?"))
    print(Fe.max_anaphoricity("My dog is the best!", "Why is he the best?"))
    print(Fe.transform_vector("The president spoke today.", "Where is he from?"))
    print(Fe.transform_scalar("The president spoke today.", "Where is he from?"))