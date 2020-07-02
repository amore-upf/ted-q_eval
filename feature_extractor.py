import feature_utils
import os
import pprint
import pandas as pd
import pickle
import functools
import logging
import time
import numpy as np
import gensim
import inspect
import spacy
import neuralcoref
import nltk.corpus
import nltk
import gleu_score   # separate import because missing from nltk
import bert_score
import transformers

logging.basicConfig(level=logging.INFO)
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)


registered_features = []  # will contain all methods decorated by @feature


def feature(func):
    """
    Feature decorator, ensures (i) features are registered, and (ii) computed values are stored under the right key for later use.
    :param func:
    :return:
    """

    @functools.wraps(func)  # makes the decorated function behave as the original (name etc)
    def decorated_func(self, items, *args, **kwargs):

        parameters = list(inspect.signature(func).parameters.items())

        # treat all defaultless arguments, including var_positional (*cols), as column names:
        columns = []
        arg_index = 0
        for name, param in parameters[2:]:
            if arg_index >= len(args):
                break
            if param.kind == param.VAR_POSITIONAL:
                # if func takes any list of args, assume they all signify columns:
                columns.extend(args[arg_index:])
                break
            elif param.default is inspect.Parameter.empty:
                # the argument signifies a column:
                columns.append(args[arg_index])
            else:
                # the argument is in fact a keyword argument:
                kwargs[name] = args[arg_index]
            arg_index += 1

        # Add all defaults to the kwargs and sort them in order of signature:
        kwargs_complete = {**{k: v.default for k, v in parameters if v.default is not inspect.Parameter.empty}, **kwargs}
        kwargs_complete_sorted = sorted([(k, v) for k,v in kwargs_complete.items()], key=lambda x: [p[0] for p in parameters].index(x[0]))
        function_str = f"{func.__name__}({','.join(columns)},{','.join(f'{str(k)}={str(v)}' for k,v in kwargs_complete_sorted)})"

        if not columns:  # no saving/loading of pre-computed results
            logging.info(f" Extracting {function_str}: computing {len(items)} anew (no column arguments; results will not be saved).")
            return func(self, items, *columns, **kwargs)

        stored_values = self.computed_features.setdefault(function_str, {})
        keys = items[columns].apply(lambda x: tuple(x), axis=1)

        # compute & store new items
        compute_indices = ~keys.isin(stored_values)
        num_new = sum(compute_indices)
        logging.info(f" Extracting {function_str}: {len(compute_indices) - num_new} precomputed items available, computing {num_new} anew.")

        if num_new > 0:
            new_values = dict(zip(keys[compute_indices],
                                  func(self, items[compute_indices], *columns, **kwargs)))
            stored_values.update(new_values)

        # return list of values
        all_values = [stored_values[key] for key in keys]
        return all_values

    registered_features.append((func.__name__, decorated_func))

    return decorated_func


def parse_string(s):
    try:
        val = int(s)
    except ValueError:
        try:
            val = float(s)
        except ValueError:
            if s in ["True", "False"]:
                val = s == "True"
            else:
                val = s
    return val


class FeatureExtractor():
    """
    Each feature is implemented as a method decorated by @feature or @feature_of(*args).
    A feature function should take a pandas dataframe of items and return a list of feature values,
    one value for each item.

    When decorated by @feature_of(*args), the arguments *args should be the labels of columns in the
    items dataframe on which the feature depends. Doing this is optional, but greatly improves time
    and space efficiency, as it ensures that computed features will be maximally concisely saved to harddisk.

    The crucial method for extracting features for items is extract().
    """

    def __init__(self, path_to_computed_features="data/extracted_features/temp.pkl", mode='a', model_paths=None, recompute_features=None):
        """
        :param path_to_computed_features: auxiliary file intended to be accessed only through this FeatureExtractor.
        :param mode:  "w", "r", "a", None: whether to write, append, or read only (or none)
        :param model_paths: a dictionary of paths to existing models, e.g., word vectors.
        """
        recompute_features = recompute_features or []

        # Bookkeeping features:
        self.feature_name_to_function = dict(registered_features)
        self.feature_order = [f[0] for f in registered_features]

        # Keep track of loaded models and pre-computed features
        self.model_map = model_paths or {}
        self.path_to_computed_features = path_to_computed_features
        self.models = {}

        # Whether to write, append or read-only
        self.mode = mode

        # Hello world:
        logging.info(" Initializing FeatureExtractor with {} registered features: \n - {}".format(
            len(self.feature_order),
            '\n - '.join(self.feature_order)))

        # Setup for appropriate writing/reading mode:
        if os.path.exists(self.path_to_computed_features) and self.mode in "w":
            logging.warning(f" {self.path_to_computed_features} exists and will be overwritten (to avoid, use mode 'a' or 'r')")
            time.sleep(.1)
            input("    ENTER to continue.")
            self.computed_features = {}
            self.mode = 'a'  # after first write, start appending
        elif not os.path.exists(self.path_to_computed_features) and self.mode in "aw":
            self.computed_features = {}
        elif self.mode in "ar":
            with open(self.path_to_computed_features, 'rb') as inpath:
                self.computed_features = pickle.load(inpath)
                logging.info(f" {self.size()} pre-computed feature values loaded from {inpath.name}.")

        for feature in recompute_features:
            for key in list(self.computed_features.keys()):
                if key.startswith(feature.replace(' ', '')):
                    logging.info(f"Will recompute feature {key} (WARNING: features on which it depends may not automatically recomputed!).")
                    del self.computed_features[key]


    def _dump(self, new=None):
        """
        If permitted by self.mode ("w" or "a"), saves the computed features dictionary to a pickle file
        (path specified when initializing class).
        :param new: (optional, used for logging message only) number of new items or None.
        :return:
        """
        if self.mode in "wa":
            os.makedirs(os.path.dirname(self.path_to_computed_features), exist_ok=True)
            with open(self.path_to_computed_features, 'w+b') as outpath:
                # dump everything except spacy (coref spans are not serializable)
                to_dump = {k: v  for k, v in self.computed_features.items() if not k.startswith('spacy')}
                pickle.dump(to_dump, outpath)
                logging.info(f" {self.size()} feature values {f'({new} new) ' if new is not None else ''}dumped to {outpath.name}.")

    def size(self):
        """
        Return the number of computed feature values, as an indication of size (mainly for logging messages).
        :return:
        """
        return sum([len(x) for x in self.computed_features.values()])

    def _parse_feature_string(self, feature_string):
        """
        Turns a feature specifying string into a tuple (function, args, kwargs) ready to be called.
        :param feature_string: e.g., "spacy(highlight,question)" or "overlap(context,question,relative_to=right)"
        :return: feature function, args, kwargs
        """
        if '(' in feature_string:  # if feature is to be called with arguments
            feature_name, arg_str = feature_string.strip(')').split('(')
            args = [arg.strip() for arg in arg_str.split(',') if '=' not in arg]
            kwargs = [arg.strip().split('=') for arg in arg_str.split(',') if '=' in arg]
            kwargs = {k.strip(): parse_string(v.strip()) for k, v in kwargs}
        else:
            feature_name = feature_string
            args = []
            kwargs = {}
        feature_function = self.feature_name_to_function[feature_name]
        return feature_function, args, kwargs

    def extract(self, items, features_to_extract):
        """
        Extracts features for items, and returns result as a list of dictionaries.
        :param items: pandas dataframe with the required columns for the selected features (e.g., "context", "question")
        :param features_to_extract: set of features to extract, as strings, e.g., spacy(context,question) will call spacy(items, context, question)
        :return: pandas dataframe with the features of one item per row
        """
        size_before = self.size()

        # Features are computed in order of definition in the code
        features_to_extract = sorted(features_to_extract, key=lambda x: self.feature_order.index(x.split('(')[0]))

        # Call all specified feature functions with the right arguments
        requested_features = [{} for _ in range(len(items))]
        for feature_string in features_to_extract:
            feature_function, args, kwargs = self._parse_feature_string(feature_string)
            values = feature_function(self, items, *args, **kwargs)
            for i, value in enumerate(values):
                if isinstance(value, dict): # in case of dict, return as multiindex columns
                    for subfeature in value:
                        requested_features[i][(feature_string, subfeature)] = value[subfeature]
                else:
                    requested_features[i][feature_string] = value

        # Only (re)save to disk if something new was actually computed
        size_after = self.size()
        if size_before != size_after:  # sufficient because elements can only be added, not removed
            self._dump(new=size_after-size_before)

        # Return as pandas dataframe
        features_df = pd.DataFrame(requested_features)
        features_df.index = items.index

        return features_df

    def extract_single(self, feature, *item, **kwargs):
        df = pd.DataFrame([item], columns=["a", "b", "c", "d", "e", "f", "g", "h"][:len(item)])
        return feature(df, *df.columns, **kwargs)[0]

    def load_word_vecs(self, key):
        """
        Load word vectors, and store in self.models for future use.
        :param key: 
        :return: gensim word vectors
        """
        model_path = self.model_map[key]
        model = self.models.get(model_path, None)
        if model is None:
            model = gensim.models.KeyedVectors.load_word2vec_format(model_path)
            self.models[model_path] = model
        return model

    def load_spacy(self):
        """
        Load Spacy ('en') with Neuralcoref, and store in self.models for future use.
        :return: Spacy+Neuralcoref model
        """
        model = self.models.get('spacy', None)
        if model is None:
            model = spacy.load('en_core_web_lg')
            neuralcoref.add_to_pipe(model)
            self.models['spacy'] = model
        return model

    # model features:

    @feature
    def spacy(self, items, *cols):
        model = self.load_spacy()
        sentences = [' '.join(t) for t in items[list(cols)].itertuples(index=False, name=None)]
        values = list(model.pipe(sentences))
        return values

    @feature
    def sif_embedding(self, items, col):
        word_vecs = self.load_word_vecs("word_vecs_for_sif")
        sentences = items[col].tolist()
        return feature_utils.compute_sif_embeddings(sentences, word_vecs)

    # length:

    @feature
    def length_ratio(self, items, col1, col2):
        nlp1 = self.spacy(items, col1)
        nlp2 = self.spacy(items, col2)
        return [len(s1) / len(s2)
                for s1, s2 in zip(nlp1, nlp2)]

    # similarity scores:

    @feature
    def lemma_overlap(self, items, col1, col2, relative_to=None):
        spacy1 = self.spacy(items, col1)
        spacy2 = self.spacy(items, col2)
        values = [feature_utils.lemma_overlap(c1, c2, relative_to=relative_to)
                  for c1, c2 in zip(spacy1, spacy2)]
        return values

    @feature
    def sif_score(self, items, col1, col2):
        sent_embs1 = self.sif_embedding(items, col1)
        sent_embs2 = self.sif_embedding(items, col2)
        return [feature_utils.cosine(emb1, emb2)
                for emb1, emb2 in zip(sent_embs1, sent_embs2)]

    @feature
    def mean_word_cosine(self, items, col1, col2):
        nlp1 = self.spacy(items, col1)
        nlp2 = self.spacy(items, col2)
        return [max([s1.similarity(s2) for s1 in c1.sents for s2 in c2.sents])
                for c1, c2 in zip(nlp1, nlp2)]

    @feature
    def bleu_score(self, items, col1, col2):
        nlp1 = self.spacy(items, col1)
        nlp2 = self.spacy(items, col2)
        return [nltk.translate.bleu_score.sentence_bleu([[t.lemma for t in sent] for sent in c1.sents], [t.lemma for t in c2])
                for c1, c2 in zip(nlp1, nlp2)]

    @feature
    def gleu_score(self, items, col1, col2):
        nlp1 = self.spacy(items, col1)
        nlp2 = self.spacy(items, col2)
        return [gleu_score.sentence_gleu([[t.lemma for t in sent] for sent in c1.sents], [t.lemma for t in c2])
                for c1, c2 in zip(nlp1, nlp2)]

    @feature
    def bert_score(self, items, col1, col2, score="f1"):
        cases = {
            "precision": 0,
            "recall": 1,
            "f1": 2,
        }
        nlp1 = self.spacy(items, col1)
        bert_scores = bert_score.score(items[col2].tolist(),
                                       [[s.text for s in c.sents] for c in nlp1],
                                       model_type="bert-base-uncased",
                                       lang="en",
                                       rescale_with_baseline=True,
                                       verbose=False)
        return bert_scores[cases[score]].numpy().tolist()

    # question patterns:

    @feature
    def question_patterns(self, items, col, only_unembedded=False):
        nlp = self.spacy(items, col)
        question_patterns = [t[0] for t in feature_utils.question_patterns]
        backup_question_patterns = [t[0] for t in feature_utils.backup_question_patterns]
        return [feature_utils.find_matching_patterns(s, question_patterns, only_unembedded=only_unembedded, backup_patterns=backup_question_patterns)
                for s in nlp]

    @feature
    def same_question_pattern(self, items, col1, col2, only_unembedded1=False, only_unembedded2=False):
        matching_question_patterns1 = self.question_patterns(items, col1, only_unembedded=only_unembedded1)
        matching_question_patterns2 = self.question_patterns(items, col2, only_unembedded=only_unembedded2)
        return [bool(set1 & set2)
                for set1, set2 in zip(matching_question_patterns1, matching_question_patterns2)]

    # whtypes:

    @feature
    def whtypes(self, items, col, only_unembedded=False):
        matching_question_patterns = self.question_patterns(items, col, only_unembedded=only_unembedded)
        return [set(feature_utils.question_pattern_to_whtype[p] for p in patterns)
                for patterns in matching_question_patterns]

    @feature
    def has_whtype(self, items, col, whtype=None, only_unembedded=False):
        whtypes = self.whtypes(items, col, only_unembedded=only_unembedded)
        return [whtype in types for types in whtypes]

    @feature
    def same_whtype(self, items, col1, col2, only_unembedded1=False, only_unembedded2=False):
        whtypes1 = self.whtypes(items, col1, only_unembedded=only_unembedded1)
        whtypes2 = self.whtypes(items, col2, only_unembedded=only_unembedded2)
        return [bool(set1 & set2)
                for set1, set2 in zip(whtypes1, whtypes2)]

    # question classes:

    @feature
    def question_classes(self, items, col, only_unembedded=False):
        matching_question_patterns = self.question_patterns(items, col, only_unembedded=only_unembedded)
        return [[feature_utils.question_pattern_to_class[p] for p in patterns]
                for patterns in matching_question_patterns]

    @feature
    def has_question_class(self, items, col, question_class=None, only_unembedded=False):
        question_classes = self.question_classes(items, col, only_unembedded=only_unembedded)
        return [question_class in classes for classes in question_classes]

    # coreference:

    @feature
    def num_coreferring_mentions(self, items, col1, col2, relative_to=None):
        nlp = self.spacy(items, col1, col2)
        return [feature_utils.coreference(c1, c2, s, relative_to=relative_to)
                for c1, c2, s in zip(items[col1], items[col2], nlp)]

    @feature
    def has_coref_to_indefinite(self, items, col1, col2):
        nlp = self.spacy(items, col1, col2)
        return [any(feature_utils.find_corefs_between(c1, c2, s, filter=feature_utils.chunk_is_indefinite))
                for c1, c2, s in zip(items[col1], items[col2], nlp)]

    @feature
    def has_coref_to_person(self, items, col1, col2, use_wordnet=True):
        nlp = self.spacy(items, col1, col2)
        return [any(feature_utils.find_corefs_between(c1, c2, s, filter=feature_utils.chunk_is_person, use_wordnet=use_wordnet))
                for c1, c2, s in zip(items[col1], items[col2], nlp)]

    # schricker old:

    @feature
    def schricker(self, items, col1, col2):
        """
        Computes Schricker's features, calling their original code (with minor tweaks/bug fixes):
        'indefinite_determiners', 'indexicals', 'explanation', 'elaboration', 'animacy',
        'strength_rule_1', 'strength_rule_2', 'normality_rule', 'max_anaphoricity'.
        :param items: pandas dataframe with "context" and "question" column
        :return: list of dictionaries with Schricker's features
        """
        word_vecs = self.load_word_vecs("word_vecs_for_schricker")
        spacy_model = self.load_spacy()
        context_question_pairs = list(items[[col1, col2]].itertuples(index=False, name=None))
        schricker_features = feature_utils.features_from_schricker(context_question_pairs, word_vecs, spacy_model)
        return schricker_features

    # schricker new:

    @feature
    def indexicals(self, items, col):
        return self.has_question_class(items, col, question_class="indexical", only_unembedded=True)

    @feature
    def elaboration(self, items, col):
        return self.has_question_class(items, col, question_class="elaboration", only_unembedded=True)

    @feature
    def explanation(self, items, col):
        return self.has_question_class(items, col, question_class="explanation", only_unembedded=True)

    @feature
    def indefinite_determiners(self, items, col1, col2):
        return self.has_coref_to_indefinite(items, col1, col2)

    @feature
    def animacy(self, items, col1, col2):
        return self.has_coref_to_person(items, col1, col2, use_wordnet=True)

    @feature
    def strength_rule_1(self, items, col1, col2):
        return self.length_ratio(items, col1, col2)

    @feature
    def strength_rule_2(self, items, col1, col2, sim_function="cos"):
        cases = {
            "bleu": self.bleu_score,
            "gleu": self.gleu_score,
            "cos": self.mean_word_cosine,
            "sif": self.sif_score,
            "bert": self.bert_score,
        }
        return cases[sim_function](items, col1, col2)

    @feature
    def normality_rule(self, items, col1, col2):
        nlp1 = self.spacy(items, col1)
        nlp2 = self.spacy(items, col2)
        return [feature_utils.mean_pairwise_similarity(c2) / feature_utils.mean_pairwise_similarity(c1)
                for c1, c2 in zip(nlp1, nlp2)]

    @feature
    def max_anaphoricity(self, items, col1, col2):
        """
        Counting coreference mentions in col2 and string matches between col1 and col2.
        """
        # TODO To be optimized: remove need for nlp1, nlp2
        nlp1 = self.spacy(items, col1)
        nlp2 = self.spacy(items, col2)
        nlp12 = self.spacy(items, col1, col2)
        return [feature_utils.anaphoricity(c1, c2, s1, s2, s12)
                for c1, c2, s1, s2, s12 in zip(items[col1], items[col2], nlp1, nlp2, nlp12)]

    # BERT logits as a feature:

    @feature
    def bert_logits(self, items, col1, col2, path_to_predictions=None):
        if path_to_predictions:     # Backwards compatibility:
            logging.warning(f"path_to_predictions is deprecated; using config preprocessing/bert_logits instead.")
        path_to_predictions = self.model_map['bert_logits']
        predictions = pd.read_csv(path_to_predictions, index_col=0)
        if any(t[0] != t[1] for t in zip(items['evoked?'], predictions['target'])):
            logging.warning(f"Bert predictions do not align with items ({path_to_predictions})")
        return [predictions.loc[i, ["logit_0", "logit_1"]].tolist() for i in items.index]

    @feature
    def bert_logit0(self, items, col1, col2, path_to_predictions=None):
        return [l[0] for l in self.bert_logits(items, col1, col2, path_to_predictions=path_to_predictions)]

    @feature
    def bert_logit1(self, items, col1, col2, path_to_predictions=None):
        return [l[1] for l in self.bert_logits(items, col1, col2, path_to_predictions=path_to_predictions)]

    @feature
    def bert_logit_difference(self, items, col1, col2, path_to_predictions=None):
        logits = self.bert_logits(items, col1, col2, path_to_predictions=path_to_predictions)
        return [l[1]-l[0] for l in logits]

    # meta info convenient for analysis
    @feature
    def tedq_answered(self, items):
        tedq = pd.read_csv("data/downloads/ted-q/TED-Q_elicitation.csv", index_col='id')['answered']
        return tedq[items['question_id']].tolist()

    # testing:

    @feature
    def foo(self, items, *cols, multiplier=1):
        """
        Fake feature just for testing something...
        :param items:
        :param cols: column identifiers (not really used)
        :param multiplier: a useless keyword argument to test if they work
        :return: A random float for each item
        """
        return np.random.random(len(items[list(cols)])) * multiplier

    # TODO Consider implementing specificity and informativeness from the stay hungry stay focused paper


if __name__ == "__main__":
    """
    Some code for testing the feature extractor.
    """

    ###################################
    ###### Parameters to change: ######
    ###################################
    FEATURES_TO_COMPUTE = [
        # "overlap(context, question, relative_to=right)",
        # "overlap(context, question, right)", # This also works, but is less robust to change
        # "question_type(question)",
        # "foo(question, multiplier=8)",
        # "foo(question, 8)", # Not allowed: since foo takes variable number of *cols, other arguments need keywords
        # 'gleu_score(context, question)',
        # 'bleu_score(context, question)',
        # 'bert_score(context, question, score=f1)',
        # # Schricker:
        # "schricker(context, question)",
        # "indexicals(question)",
        # "elaboration(question)",
        # "explanation(question)",
        # "indefinite_determiners(context, question)",
        # "animacy(context, question)",
        # "strength_rule_1(context, question)",
        # "strength_rule_2(context, question)",
        # "normality_rule(context, question)",
        # "max_anaphoricity(context, question)",
        # "question_patterns(question)",
        "tedq_answered",
    ]
    # TODO Replace these by Spacy-internal models for simplicity
    MODELS_TO_USE = {
        "word_vecs_for_schricker": "/home/u148187/datasets/glove.6B/glove.6B.300d.word2vecformat.txt",
        "word_vecs_for_sif": "/home/u148187/datasets/glove.6B/glove.6B.300d.word2vecformat.txt",
        # In addition, Spacy+Neuralcoref is included automatically
    }
    ###################################
    ###################################

    # '/home/u148187/datasets/GoogleNews-vectors-negative300.bin'
    # '/home/u148187/datasets/glove.6B/glove.840B.300d.txt'  ## doesn't work
    # '/home/u148187/datasets/glove.6B/glove.6B.300d.word2vecformat.txt'

    # Apply to random sample of the tedq items, just for testing (same random sample each time):
    np.random.seed(12345)
    items_sample = pd.read_csv("data/tasks/c_q_binary/tedq_train.csv").sample(200)

    print(items_sample.to_string())
    print()
    time.sleep(.1)

    # Extract features:
    Fe = FeatureExtractor(
        model_paths=MODELS_TO_USE,
        mode="w",   # 'a' append to previously extracted features; set to 'w' to compute everything anew
    )

    # Extract features for a whole dataframe:
    extracted_features = Fe.extract(items_sample, FEATURES_TO_COMPUTE)
    # Alternative for a single item:
    print(Fe.extract_single(Fe.question_patterns, "blabla sentence 1", "Is their enough excess for everyone?"))

    # Now you would typically write the features to a csv, together with the items' index, for subsequent use in analysis/machine learning.
    # Now, for testing, instead, let's just print the result:
    time.sleep(.1)
    print("\nExtracted features:")
    np.set_printoptions(threshold=10)
    for (i, item), (_, features) in zip(items_sample.iterrows(), extracted_features.iterrows()):
        print(f"\n Item {i}:\n - Context: {item['context']}\n - Question: {item['question']}\n - Highlight: {item['highlight']}")
        pprint.pprint(features)
