import pandas as pd
import os
import random
from tqdm import tqdm
import csv
import nltk
import numpy as np
import sys
import click
from localconfig import config
import itertools
import shutil

# Always append project's main dir (1 level up), for easy imports
main_project_dir = os.sep.join(os.path.normpath(__file__).split(os.sep)[:-1])
sys.path.append(main_project_dir)

import data_utils

# TODO Also include the QUD annotations task for eval.

@click.command()
@click.argument('config_file', type=click.Path(exists=True), required=True)
@click.option('--out_dir', type=click.Path(), required=False, default=None)
def main(config_file, out_dir):
    """
    :param config_file:
    :param out_dir: default out_dir is ../data/tasks/<format>
    :return:
    """
    config.read(config_file)
    config.meta.out_dir = out_dir or os.path.relpath(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2] + ["data", "tasks", config.task.format]))
    config.meta.name = os.path.split(config_file)[1][:-4]   # take name from config file
    config.meta.dataset = data_utils.resolve_path(config.meta.dataset, additional_dirs=[config_file])
    os.makedirs(config.meta.out_dir, exist_ok=True)
    shutil.copy(config_file, config.meta.out_dir)

    random.seed(config.meta.random_seed)
    np.random.seed(random.randint(0, 99999))

    if config.task.format == "c_q_highlight":
        c_q_highlight(config)
    elif config.task.format == "c_q_binary":
        c_q_binary(config)


def c_q_highlight(config):
    print("c_q_highlight not yet implemented.")


def c_q_binary(config):

    if config.meta.dataset_format == "dialogues":
        df = c_q_binary_from_bookcorpus(config.meta.dataset,
                                        config.task.context_size,
                                        (config.task.confounder_min_distance, config.task.confounder_max_distance),
                                        config.task.num_confounders_per_item,
                                        )
    elif config.meta.dataset_format == "tedq":
        df = c_q_binary_from_tedq(config.meta.dataset,
                                  config.task.context_size,
                                  (config.task.confounder_min_distance, config.task.confounder_max_distance),
                                  config.task.num_confounders_per_item,
                                  )
    elif config.meta.dataset_format == "quds":
        df = c_q_binary_from_quds(config.meta.dataset,
                                  config.task.context_size,
                                  (config.task.confounder_min_distance, config.task.confounder_max_distance),
                                  config.task.num_confounders_per_item,
                                  )
    else:
        print("Data format unknown.")
        quit()

    split = data_utils.n_fold_split(df, folds=(1-config.split.test_proportion, config.split.test_proportion), by="context_id" if config.split.contiguity > 1 else None, contiguity=config.split.contiguity)
    for part, label in zip(split, ["train", "test"]):
        path_for_csv = f'{config.meta.out_dir}/{config.meta.name}_{label}.csv'
        part = df.loc[part, :]
        part.to_csv(path_for_csv, index=False)
        print(f'{config.meta.name} {label} partition ({len(part)} context-question pairs; {part["evoked?"].mean():.2f} positive) written to {path_for_csv}.')


def c_q_binary_from_tedq(in_path, context_size, confounder_range, n_confounders_per_item):
    """
    :param in_path:
    :param context_size: in number of tedq 'chunks'.
    :param confounder_range:
    :param confounders_per_item:
    :param random_state:
    :return:
    """
    annotations = data_utils.read_data(in_path, mode='ted-q')
    questions = annotations.loc[annotations['type'] == 'question']
    path_to_sources = data_utils.resolve_path('data/downloads/ted-q/sources/')
    contexts = data_utils.get_dict_from_source_to_contexts(questions, n_sentences_per_context=context_size, path_to_sources=path_to_sources)

    # Compute list of lists of question ids, ordered by chunk_end, for computing alternative questions later on
    questions = questions.sort_values(by=['source', 'chunk_end'])
    ordered_question_ids = []
    last_chunk_end = 0
    for i, arbitrary_question in questions.iterrows():
        if arbitrary_question['chunk_end'] == last_chunk_end:
            ordered_question_ids[-1].append(i)
        else:
            ordered_question_ids.append([i])
            last_chunk_end = arbitrary_question['chunk_end']

    # Assemble a dataframe of items / a dictionary of items
    columns = ['evoked?', 'context', 'question', 'source', 'context_id', 'question_id', 'highlight']
    items_for_df = []
    for i, question_ids in enumerate(ordered_question_ids):

        # neighbors are within distance chunks ago or in the future
        min_dist = confounder_range[0]
        max_dist = confounder_range[1]
        neighbor_ids = []
        for dist in range(min_dist, max_dist+1):
            neighbor_ids += ordered_question_ids[i-dist] if i >= dist else []
            neighbor_ids += ordered_question_ids[i+dist] if i < len(ordered_question_ids)-dist else []
            neighbor_ids = random.sample(neighbor_ids, min(len(neighbor_ids), n_confounders_per_item*len(question_ids)))

        arbitrary_question = questions.loc[question_ids[0]]
        source = arbitrary_question['source']
        chunk_start = arbitrary_question['chunk_start']
        chunk_end = arbitrary_question['chunk_end']
        context = contexts[source][(chunk_start, chunk_end)]

        original_highlights = questions.loc[question_ids]['highlight'].tolist()

        # Create items
        for question_id, evoked in [(q, True) for q in question_ids] + [(q, False) for q in neighbor_ids]:
            question = questions.loc[question_id]
            if evoked:
                highlight = question["highlight"]
            else:
                highlight = random.choice(original_highlights)
            item = [evoked, context, question['content'], source, f"{source}-{chunk_end}", question_id, highlight]
            items_for_df.append(item)

    df = pd.DataFrame(items_for_df, columns=columns)

    # Only keep those items where highlight is in context (with context size >= 2 this can happen only for
    # excerpt-initial chunks, due to them being larger than 2 sentences.)
    df = df.loc[df.apply(lambda x: x['highlight'] in x['context'], axis=1)].reset_index(drop=True)

    return df


def c_q_binary_from_bookcorpus(in_path, context_size, confounder_range, n_confounders_per_item):
    """

    :param in_path:
    :param context_size: in number of turns
    :param confounder_range:
    :param n_confounders_per_item:
    :param context_size_by_sentences: whether to count sentences; otherwise turns
    :return:
    """
    print("Extracting training data from bookcorpus dialogues...")
    def is_question(utterance):
        return utterance.endswith('?') or utterance.endswith('?!')

    # First collect all one-utterance questions, to be used as confounders
    questions_per_source = {}
    n_questions_total = 0

    with open(in_path) as infile:
        numlines = sum(1 for _ in infile)

    with open(in_path) as infile:
        for d, dialog in tqdm(enumerate(csv.reader(infile)), total=numlines, desc='Finding questions'):
            # each dialogue is a comma-separated sequence of turns (each potentially multiple sentences)
            if not dialog[0] in questions_per_source:
                questions_per_source[dialog[0]] = []
            for t, turn in enumerate(dialog[1:]):
                for u, utterance in enumerate(nltk.sent_tokenize(turn)):
                    if is_question(utterance):
                        questions_per_source[dialog[0]].append((f"{dialog[0]}-{d}-{t}-{u}", utterance))
                        n_questions_total += 1
    print(f'Found {n_questions_total} questions from {len(questions_per_source)} sources.')

    columns_for_df = ['evoked?', 'context', 'question', 'source', 'context_id', 'question_id']
    items_for_df = []
    with open(in_path) as infile:
        for d, dialog in tqdm(enumerate(csv.reader(infile)), total=numlines, desc='Composing items'):
            if len(dialog[1:]) > 1:
                source = dialog[0]
                questions_of_source = questions_per_source[source]
                dialog_tokenized = [nltk.sent_tokenize(turn) for turn in dialog[1:]]
                # Search all pairs of subsequent, nonempty, sent_tokenized turns:
                for t, (turn1, turn2) in enumerate(zip(dialog_tokenized, dialog_tokenized[1:])):
                    if turn1 and turn2:
                        utterance = turn2[0]
                        if is_question(utterance) and not is_question(turn1[-1]):
                            context_sentences = list(itertools.chain(*dialog_tokenized[:t+1]))
                            context = ' '.join(context_sentences[-min(len(context_sentences), context_size):])
                            # context = ' '.join(dialog[max(1, t+1-(context_size-1)):t+2])   # old: context_size by chunk:
                            context_id = f"{source}-{d}-{t}"
                            question_id = f"{source}-{d}-{t+1}-0"
                            # Add positive item:
                            items_for_df.append([True, context, utterance, source, context_id, question_id])
                            # Find candidate confounders:
                            question_index = [q[0] for q in questions_of_source].index(question_id)
                            start_left, end_left = question_index - confounder_range[1], question_index - confounder_range[0]
                            start_right, end_right = question_index + confounder_range[0], question_index + confounder_range[1]
                            candidate_confounders = questions_of_source[start_left:end_left] + \
                                                    questions_of_source[start_right:end_right]
                            # Add confounder(s):
                            n_confounders = min(n_confounders_per_item, len(candidate_confounders))
                            for question_id, alternative_question in random.sample(candidate_confounders, n_confounders):
                                items_for_df.append([False, context, alternative_question, source,
                                                     f"{source}-{d}-{t}", question_id])

    df = pd.DataFrame(items_for_df, columns=columns_for_df)

    return df


def c_q_binary_from_quds(in_path, context_size, confounder_range, n_confounders_per_item):

    columns_for_df = ['evoked?', 'context', 'question', 'source', 'context_id', 'question_id']
    items = []

    # Iterate over files in directory
    for filename in os.listdir(in_path):

        cleaned_lines = []
        with open(os.path.join(in_path, filename), "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("Interviewer") or line.startswith("Snowden"):
                    continue
                if line.startswith(">"):
                    cleaned_lines.append(line)
                elif cleaned_lines:
                    cleaned_lines[-1] = cleaned_lines[-1] + " " + line

        turns = []
        n_chars = 0
        n_questions = 0
        for line in cleaned_lines:
            line = line.strip(">").strip()
            # TODO not all assertions have the A label...
            label, line = line[0], line.split(maxsplit=1)[1].strip('- ')
            if label == "Q":
                id = f"Q_{filename}.A{n_chars}.Q{n_questions}"
                n_questions += 1
            else:
                id = f"A{n_chars}"
                n_chars += len(line) + 1
                n_questions = 0
            turns.append((label, line, id))

        # Construct items
        for i, (a, q) in enumerate(zip(turns, turns[1:])):
            if a[0] == "A" and q[0] == "Q":
                context = [t[1] for t in turns[:i+1] if t[0] == "A"]
                context = nltk.sent_tokenize(' '.join(context))
                if context_size < len(context):
                    context = context[-context_size:]
                context = ' '.join(context)

                items.append([True, context, q[1], filename, a[2], q[2]])
                left_questions = [t for t in turns[:i] if t[0] == "Q"]
                right_questions = [t for t in turns[i+1:] if t[0] == "Q"]

                left_start = max(0, len(left_questions) - confounder_range[1])
                left_end = max(len(left_questions), len(left_questions) - confounder_range[0])
                right_start = max(0, len(right_questions) - confounder_range[0])
                right_end = min(len(right_questions), confounder_range[1])

                candidate_confounders = left_questions[left_start:left_end] + \
                                        right_questions[right_start:right_end]

                for confounder in random.sample(candidate_confounders, min(n_confounders_per_item, len(candidate_confounders))):
                    items.append([False, context, confounder[1], filename, a[2], confounder[2]])

    df = pd.DataFrame(items, columns=columns_for_df)

    # TODO I can do more to clean up the contexts, but is there really a point?
    #      No train/test split, no cross-validation etc. -- that's the only reason why it would matter.

    # for i, row in df.groupby('context_id').agg({'context': list}).iterrows():
    #     print("\n\n", i)
    #     for x in row['context']:
    #         print(x)

    return df


if __name__ == "__main__":
    main()