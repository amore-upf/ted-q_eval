import pandas as pd
import csv
from tqdm import tqdm
import os
import numpy as np
import itertools
import logging
import nltk


def prepare_c_q_binary_for_bert(df):
    df.rename(columns={'context': 'text_a', 'question': 'text_b', 'evoked?': 'labels'}, inplace=True)
    df = df[['text_a', 'text_b', 'labels']]
    df['labels'] = df['labels'].astype(int)  # Seems necessary after al.
    return df


def get_highlight_containing_sentence(df):
    logging.info("Reducing context to only the sentence containing the highlight, or the last sentence.")
    # Use a loop instead of pd.apply to save memory (I think?):
    for i, row in df.iterrows():
        sents = nltk.sent_tokenize(row['context'])
        if 'highlight' in row:
            for sent in sents:
                if row['highlight'] in sent:
                    break
        else:
            sent = sents[-1]
        df.at[i, 'context'] = sent
    return df


def read_feature_list_from_config(string):
    # parse feature strings:
    split_string = string.strip('[]').split(',' if '\n' not in string else ',\n')
    feature_strings = [s.strip(' \n') for s in split_string]
    feature_strings = [s for s in feature_strings if s]
    if any(('(' in s) != (')' in s) for s in feature_strings):  # TODO make this check more robust:
        logging.error(f"Feature list was incorrectly parsed.")
    return feature_strings


def resolve_path(path, additional_dirs=None):
    dirs = ["", os.getcwd(), os.path.split(__file__)[0]] # main project dir
    if additional_dirs:
        dirs += additional_dirs
    for dir in dirs:
        newpath = os.path.relpath(os.path.join(dir, path))
        if os.path.exists(newpath):
            if path != newpath:
                logging.warning(f"Resolved {path} as {newpath}.")
            return newpath
    open(path)


def can_write(path, warning=None):
    if os.path.exists(path):
        if os.path.isfile(path) or len(os.listdir(path)) > 0:
            return input("Overwrite existing {}? N/y{}".format(path, '' if warning is None else '  [WARNING: '+ warning.upper()+'!]')).lower().startswith("y")
    return True


def _string_to_list(l):
    return [n for n in l.strip('[]').split(', ')]


def sort_contextids(ids):
    if '.txt-' in str(ids[0]):  # it's TEDQ
        ids = sorted(ids, key=lambda x: (x.split('.txt-')[0], float(x.split('.txt-')[1])))
    elif 'A6' in str(ids):  # it's QUDs
        ids = sorted(ids, key=lambda x: int(x[1:]))
    elif len(str(ids[0]).split('-')) > 2:  # it's bookcorpus
        ids = sorted(ids, key=lambda x: tuple(int(s.strip()) for s in x.split('-')[-2:]))
    else:
        logging.warning("From sort_contextids(): Didn't know how to sort the context_id indices.")
        ids = sorted(ids)
    return ids


def remove_overlap(indices1, indices2, window):
    all_indices = sort_contextids(indices1 + indices2)
    filtered_indices2 = []
    for i in indices2:
        idx = all_indices.index(i)
        window_start = max(0, idx - window)
        window_end = min(len(all_indices), idx + window)
        window_ids = all_indices[window_start:idx] + (all_indices[idx+1:window_end] if idx+1 < window_end else [])
        if not any([j in indices1 for j in window_ids]):
            filtered_indices2.append(i)
    return indices1, filtered_indices2


def n_fold_split(df, folds, by=None, random_state=None, return_train_test_splits=False, contiguity=None, remove_overlap_window=None):
    """
    Split a dataframe in folds (given number or proportion), optionally splitting 'by' a column, meaning
    overlap in this column is avoided.
    :param df: pandas dataframe
    :param folds: integer number of folds, or list of integers/floats representing relative sizes.
    :param by: optional column label to split by, meaning overlap in this column is avoided
    :param random_state:
    :param return_train_test_splits: if false, returns a list of all the folds; if true, returns a list of train/test splits.
    :return:
    """
    print(len(df), folds, by, random_state, return_train_test_splits, contiguity, remove_overlap_window)
    if random_state:
        np.random.seed(random_state)
    if isinstance(folds, list) or isinstance(folds, tuple):
        folds = [f/sum(folds) for f in folds]
    else:
        folds = [1/folds for _ in range(folds)]
    if not by:
        indices = df.index.tolist()
    else:
        indices = list(df[by].unique())
        indices = sort_contextids(indices)
    if contiguity:
        indices = [indices[i:i + contiguity] for i in range(0, len(indices), contiguity)]
    np.random.shuffle(indices)
    folds = np.cumsum([f * len(indices) for f in folds])
    folds = [0] + [int(round(x)) for x in folds]
    folds[-1] = len(indices)    # in case of rounding error
    fold_indices = [indices[i:j] for i,j in zip(folds, folds[1:])]
    if contiguity:
        fold_indices = [list(itertools.chain(*split_inds)) for split_inds in fold_indices]
    if return_train_test_splits:
        fold_indices = [(list(itertools.chain(*[fold_indices[i] for i in range(len(fold_indices)) if i != j])), fold_indices[j]) for j in range(len(fold_indices))]
        if remove_overlap_window:
            fold_indices = [remove_overlap(f[0], f[1], remove_overlap_window) for f in fold_indices]
        if by:
            fold_indices = [tuple(df.loc[df[by].isin(inds)].index.tolist() for inds in split) for split in fold_indices]
    else:
        if remove_overlap_window and len(fold_indices) == 2:  # TODO This doesn't work yet with multiple folds
            fold_indices = remove_overlap(fold_indices[0], fold_indices[1], remove_overlap_window)
        if by:
            fold_indices = [df.loc[df[by].isin(inds)].index.tolist() for inds in fold_indices]
    return fold_indices


def read_data(path, mode=None):
    """
    Takes a dataset name and returns a pandas dataframe, with some minimal preprocessing where necessary.
    :param dataset: identifier of the dataset (see keys in data_paths).
    :return: pandas Dataframe
    """

    # TODO make iterable?

    print(f"-------\nLoading dataset {path}")

    if mode is None:
        # TODO automatically determine mode from path if possible
        pass

    # Bookcorpus requires some special treatment (csv lines have varying number of columns)
    if mode.startswith('bookcorpus'):
        lines = []
        with open(path) as infile:
            reader = csv.reader(infile)
            for i, line in tqdm(enumerate(reader), desc='Reading bookcorpus'):
                lines.append([line[0], line[1:]])
        df = pd.DataFrame(lines, columns=['source', 'dialogue'])

    # For other corpora it's more straightforward:
    else:
        converters = {}
        sep = ','
        index_col = None
        names = None
        usecols = None
        keep_rows = lambda df: [True] * len(df)

        if mode.startswith('quora'):
            sep = '\t'
            index_col = 'id'

        elif mode == 'ted-q':
            converters = {'potential_answers': _string_to_list,
                          'potential_question': _string_to_list,
                          'neighbors': _string_to_list}
            index_col = 'id'
            keep_rows = lambda df: df['worker'] != 'Lucio'  # Remove this idiot annotator

        elif mode == 'ted-q_comparison_aggregated':
            index_col = ["question1_id", "question2_id"]
            converters = {'relatedness_list': lambda x: [float(x) for x in _string_to_list(x)]}

        elif mode == 'ted-q_comparison':
            index_col = ["target_id", "comparison_id", "assignmentid"]
            usecols = lambda col: col not in ['worktimeinseconds', 'reward']

        df = pd.read_csv(path, converters=converters, sep=sep, index_col=index_col, names=names, usecols=usecols).sort_index()

        df = df.loc[keep_rows(df)]

    print(f"Loaded dataset {path}; {len(df)} rows; {len(df.columns)} columns; indexed by {df.index.names}. First 5 rows:\n{df[:5].to_string()}\n------")

    return df


def get_dict_from_source_to_contexts(annotations, n_sentences_per_context, path_to_sources='data/downloads/ted-q/sources/'):
    """
    Takes the evoked questions dataset and generates a dictionary containing,
    for each source text, a dictionary of chunk_start/chunk_end pairs to strings
    representing discourse context up to (and including) the given chunk.
    :param evoked_questions: Pandas dataframe as read from annotations.csv
    :param max_chunks_per_snippet: How many chunks (~2 sentences each) should be maximally considered for the context.
    :param snippet_target_num_words: How many words the context should minimally have (within the limit set by max_chunks_per_snippet).
    :return: dictionary of sources to dictionaries of chunk_start/chunk_end to strings.
    """
    sources = annotations['source'].unique()

    questions_per_chunk = annotations.reset_index().groupby(['source', 'chunk_start', 'chunk_end']).agg({'id': list}).rename(columns={'id': 'annotation_ids'})
    chunks = sorted(questions_per_chunk.index.tolist(), key=lambda x: x[1])
    source_to_list_of_chunks = {source: [(chunk[1], chunk[2]) for chunk in chunks if chunk[0] == source] for source in sources}

    source_to_dict_of_snippets = {source: {} for source in sources}
    for (source, chunk_start, chunk_end), row in tqdm(questions_per_chunk.iterrows(), total=len(questions_per_chunk), desc='Retrieving discourse contexts'):
        chunk_idx = source_to_list_of_chunks[source].index((chunk_start, chunk_end))
        earlier_chunk_start = source_to_list_of_chunks[source][max(0, chunk_idx - 2*n_sentences_per_context)][0] # 2* is arbitrarily liberal
        snippet = get_fragment_raw(source, int(earlier_chunk_start), int(chunk_end), path_to_sources=path_to_sources)
        # old: leave the snippet as such for n chunks, regardless of n sentences
        # new:
        sentences = nltk.sent_tokenize(snippet)
        snippet = ' '.join(sentences[-min(len(sentences), n_sentences_per_context):])
        source_to_dict_of_snippets[source][(chunk_start, chunk_end)] = snippet

    return source_to_dict_of_snippets


def get_fragment_raw(source, start_char=None, end_char=None, path_to_sources='data/downloads/ted-q/sources/'):
    """
    Retrieves a fragment from a raw source text by character (between start_char and end_char).
    :param path: path to source text
    :param start_char: can be None
    :param end_char: can be None
    :return: string (any newlines replaced by spaces)
    """
    path = os.path.join(path_to_sources, source)
    source_text = open(path).read()
    start_char = start_char or 0
    end_char = end_char or len(source_text)

    return source_text[start_char:end_char].replace('\n', ' ')


def render_evoked_questions_inline(by='chunk_end'):
    """
    Prints a string representation of a text with evoked questions in-line. Currently only handles TED talks.
    :param by: whether to insert questions at chunk_end, or at highlight_end.
    """
    df = read_data('ted-q')
    evoked_questions = df.loc[df['type'] == 'question']
    sources =  evoked_questions['source'].unique()
    # evoked_questions.reset_index(inplace=True)
    # evoked_questions.set_index(['source', by, 'id'], inplace=True)

    for source in sources:
        lines = []
        fragment = get_fragment_raw(source)
        questions_for_fragment  = evoked_questions.loc[evoked_questions['source'] == source]
        questions_for_fragment.sort_values(by=by, inplace=True)
        last_i = 0
        for _, question in questions_for_fragment.iterrows():
            if last_i != question[by]:
                lines.append(fragment[last_i:int(question[by])])
            last_i = int(question[by])
            lines.append('Q: ' + question['content'])
        print('\n'.join(lines))



def downsample_csv(in_path, n, out_filename=None, random_state=None):
    df = pd.read_csv(in_path)
    if n <= 1.0:
        n = round(n * len(df))
    if out_filename:
        out_filepath = os.path.join(os.path.split(in_path)[0], out_filename)
    else:
        out_filepath = f'{in_path[:-4]}_{n}.csv'
    df_sample = df.sample(n, random_state=random_state)
    df_sample.to_csv(out_filepath, index=False)
    print(f"Downsampled to {len(df_sample)}, written to {out_filepath}")


def add_max_contrib_diff_column():

    model_dirs = ['schricker2.0_bookcorpus_context',
    'schricker2.0_bookcorpus_sentence',
    'schricker2.0_tedq_context',
    'schricker2.0_tedq_highlight',
    'schricker2.0_tedq_sentence',]
    pred_dirs = [f'outputs/c_q_binary/tree/{model_dir}/predictions' for model_dir in model_dirs]

    pred_paths = [f'{pred_dir}/{pred_file}' for pred_dir in pred_dirs for pred_file in os.listdir(pred_dir)]

    for path_to_preds in pred_paths:
        if not 'bookcorpus_train' in path_to_preds:

            print(path_to_preds)
            preds = pd.read_csv(path_to_preds, index_col=0)
            print(preds[:5].to_string())
            print(preds.columns)
            features = [col[len('contrib_0_'):] for col in preds.columns if col.startswith('contrib_0_')]
            print(features)
            for feature in features:
                preds[f'contrib_{feature}'] = preds[f'contrib_1_{feature}'] - preds[f'contrib_0_{feature}']

            preds['most_pos_feature'] = preds[[f'contrib_{feature}' for feature in features]].idxmax(axis=1).apply(lambda x: x.replace('contrib_', ''))
            preds['most_neg_feature'] = preds[[f'contrib_{feature}' for feature in features]].idxmin(axis=1).apply(lambda x: x.replace('contrib_', ''))
            preds['dominant_feature'] = preds[[f'contrib_{feature}' for feature in features]].abs().idxmax(axis=1).apply(lambda x: x.replace('contrib_', ''))

            preds.to_csv(path_to_preds)


def turn_tedq_into_json(df):
    # df = pd.read_csv(inpath)
    list_of_dicts = []
    for i, row in df.iterrows():
        list_of_dicts.append({'context': row['context'],
                              'qas': [{'id': i,
                                       'question': row['question'],
                                       'is_impossible': not row['evoked?'],
                                       'answers': [] if not row['evoked?'] else [
                                           {'text': row['highlight'],
                                            'answer_start': row['context'].index(row['highlight'])}
                                           ]
                                       }]
                              })
    return list_of_dicts


if __name__ == '__main__':

    # turn_tedq_into_json()


    quit()


    # downsample_csv('data/tasks/c_q_binary/bookcorpus_test.csv', 4000, random_state=13579)

    ##
    #  Commands for reading the various datasets:
    # read_data('bookcorpus_sample')
    # read_data('bookcorpus')    # takes some time  # TODO add option to return iterator

    ##
    #  Code for reading the evoked questions data and then, for each question annotation,
    #  extracting (from the source texts) a preceding discourse context of a desired size:
    annotations = read_data('data/downloads/ted-q/TED-Q_elicitation.csv', mode='ted-q')
    contexts = get_dict_from_source_to_contexts(annotations, n_sentences_per_context=4)

    ## So now we can easily take an arbitray annotation and print it with its context:
    annotation = annotations.loc['1565638102.f97d405b607c9aca212d884d340cb014.10.0']
    source = annotation['source']
    print('CONTEXT:', contexts[source][(annotation['chunk_start'], annotation['chunk_end'])])
    print('QUESTION:', annotation['content'])

    ## To print the text with questions in-line (TED only)
    # render_evoked_questions_inline()

