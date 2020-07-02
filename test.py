import feature_extractor
import torch
import data_utils
import simpletransformers.classification
import pandas as pd
import logging
import os
from treeinterpreter import treeinterpreter
import joblib
import numpy as np

import click

import sklearn
from train import bert_metrics, global_bert_settings

from localconfig import config


@click.command()
@click.argument('config_file', type=click.Path(exists=True), required=True)
@click.option('--model_dir', type=click.Path(), required=False, default=None)
@click.option('--test_data', type=click.Path(), required=False, default=None)
@click.option('--context', type=str, required=False, default=None)
@click.option('--out_name', type=str, required=False, default=None)
@click.option('--overwrite', is_flag=True)
def main(config_file, model_dir, test_data, context, out_name, overwrite):
    """
    :param config: file or dir
    :param model_dir:
    :return:
    """

    if os.path.isdir(config_file):
        ini_files = [f for f in os.listdir(config_file) if f.endswith('.ini')]
        if not ini_files or len(ini_files) > 1:
            logging.warning(f"No unique config file found in {config_file}.")
            quit()
        config_file = os.path.join(config_file, ini_files[0])
    if not model_dir:
        # Will take the model(s) in the config file's (fold sub)dir.
        model_dir = os.path.split(config_file)[0]

    config.read(config_file)

    # Override config if command line args are given
    config.testing.predictions_filename = out_name or config.testing.predictions_filename
    config.testing.data = test_data or config.testing.data
    config.testing.context = context or config.testing.context

    config.testing.predictions_filename = config.testing.predictions_filename or os.path.split(config.testing.data)[1]
    if not config.testing.predictions_filename.endswith(".csv"):
        config.testing.predictions_filename += ".csv"
    os.makedirs(os.path.join(model_dir, 'predictions'), exist_ok=True)
    out_path = os.path.join(model_dir, 'predictions', config.testing.predictions_filename)

    if not overwrite and not data_utils.can_write(out_path):
        quit()

    config.testing.data = data_utils.resolve_path(config.testing.data, additional_dirs=[config_file])
    test_data = pd.read_csv(config.testing.data)

    config.testing.context = config.testing.context or config.training.context
    if config.testing.context == "sentence":
        test_data = data_utils.get_highlight_containing_sentence(test_data)
    elif config.testing.context == "highlight":
        test_data['context'] = test_data['highlight']
    elif config.testing.context == "context-highlight" and config.meta.model == 'bert':
        test_data['context'] = test_data['highlight'] + test_data['highlight']
        # for tree, this is done via the features themselves, not the 'context' column.

    if config.meta.model == "bert":

        if 'pytorch_model.bin' in model_dir:
            model_paths = [model_dir]
        else:
            model_paths = []
            for i in range(config.training.n_splits):  # Collect models for each fold
                model_paths.append(os.path.join(model_dir, f'fold_{i}', 'best_model'))

        if config.testing.only_first_fold:
            model_paths = model_paths[:1]

        test_data = data_utils.prepare_c_q_binary_for_bert(test_data)

        predict_bert(model_paths, test_data, out_path)

    elif config.meta.model == "tree":


        model_path = os.path.join(model_dir, 'best_model.joblib')

        feature_strings = data_utils.read_feature_list_from_config(config.testing.features or config.training.features)
        recompute_feature_strings = data_utils.read_feature_list_from_config(config.testing.recompute_features or config.training.recompute_features)

        models_to_use = {
            "word_vecs_for_schricker": config.testing.word_vecs_for_schricker or config.training.word_vecs_for_schricker,
            "word_vecs_for_SIF": config.testing.word_vecs_for_SIF or config.training.word_vecs_for_SIF,
            "bert_logits": config.testing.bert_logits or config.training.bert_logits,
        }

        fe = feature_extractor.FeatureExtractor(
            model_paths=models_to_use,
            mode="a" if config.testing.cached_features or config.training.cached_features else "w",
            recompute_features=recompute_feature_strings,
            path_to_computed_features=f"data/extracted_features/{config.testing.cache_name or config.training.cache_name}",
        )
        X = fe.extract(test_data, features_to_extract=feature_strings)
        X = X.fillna(X.mean())

        y = test_data['evoked?']

        predict_tree(model_path, X, y, out_path)


def predict_bert(model_paths, test_data, out_path):

    n_labels = len(test_data['labels'].unique())

    aggregated_model_outputs = None
    for model_path in model_paths:
        # TODO avoid the 0/eval_results.txt output.
        model = simpletransformers.classification.ClassificationModel('bert', model_path, num_labels=n_labels, use_cuda=torch.cuda.is_available(), cuda_device=0, args=global_bert_settings)
        _, model_outputs, _ = model.eval_model(test_data, **bert_metrics)
        if not aggregated_model_outputs:
            aggregated_model_outputs = model_outputs
        else:
            aggregated_model_outputs += model_outputs

    df = pd.DataFrame(aggregated_model_outputs, columns=[f"logit_{i}" for i in range(n_labels)], index=test_data.index)
    df['prediction'] = aggregated_model_outputs.argmax(axis=1)
    df['target'] = test_data['labels']
    df.to_csv(os.path.join(out_path))

    print("MCC:", sklearn.metrics.matthews_corrcoef(df['target'], df['prediction']))

    print("Predictions written to", out_path)


def predict_tree(model_path, X, y, out_path):

    model = joblib.load(model_path)

    n_labels = len(y.unique())
    model_outputs, bias, contributions = treeinterpreter.predict(model, X)
    contributions = contributions.reshape(contributions.shape[0], -1)
    concatenated = np.concatenate((model_outputs, bias, contributions), axis=1)

    columns = ([f"logit_{i}" for i in range(n_labels)] +
               [f"bias_{i}" for i in range(n_labels)] +
               [f"contrib_{i}_{feature}" for feature in X.columns for i in range(n_labels)])

    df = pd.DataFrame(concatenated, columns=columns, index=list(range(len(y))))
    df['prediction'] = model_outputs.argmax(axis=1)
    df['target'] = y

    # Some columns for convenience, summarizing the feature contributions:
    features = [col[len('contrib_0_'):] for col in df.columns if col.startswith('contrib_0_')]
    for feature in features:
        df[f'contrib_{feature}'] = df[f'contrib_1_{feature}'] - df[f'contrib_0_{feature}']
    df['most_pos_feature'] = df[[f'contrib_{feature}' for feature in features]].idxmax(axis=1).apply(
        lambda x: x.replace('contrib_', ''))
    df['most_neg_feature'] = df[[f'contrib_{feature}' for feature in features]].idxmin(axis=1).apply(
        lambda x: x.replace('contrib_', ''))
    df['dominant_feature'] = df[[f'contrib_{feature}' for feature in features]].abs().idxmax(axis=1).apply(
        lambda x: x.replace('contrib_', ''))

    print("MCC:", sklearn.metrics.matthews_corrcoef(df['target'], df['prediction']))

    df.to_csv(out_path)

    print("Predictions written to", out_path)


if __name__ == "__main__":
    main()

