import feature_extractor
import torch
import data_utils

import simpletransformers.classification
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection
import sklearn
import logging
import random
import itertools
import os
import shutil

import numpy as np

import click

import seaborn as sns
from matplotlib import pyplot as plt

from localconfig import config

import joblib
import json

# Logging settings:
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


global_bert_settings = {
    "use_cached_eval_features": False,
    "reprocess_input_data": True,
    "cache_dir": "cache_dir/",
    'overwrite_output_dir': True,
    'evaluate_during_training': True,  # needs to be true also for early stopping to work
    "evaluate_during_training_steps": 1,  # if <0, only do every epoch
    "early_stopping_patience": 5,
    "use_early_stopping": True,
    "evaluate_during_training_verbose": False,
    'save_model_every_epoch': False,  # not necessary; best model will be saved regardless
    'save_eval_checkpoints': False,  # not necessary; best model will be saved regardless
    "save_steps": -1,  # no checkpoints need to be saved
    "fp16": False,  # Setting to true gives error from Apex.
    "silent": True,
}
bert_metrics = {'acc': sklearn.metrics.accuracy_score,
                'f1': sklearn.metrics.f1_score,
                }


@click.command()
@click.argument('config_file', type=click.Path(exists=True), required=True)
@click.option('--out_dir', type=click.Path(), required=False, default=None)
def main(config_file, out_dir):
    """
    :param config_file:
    :param out_dir: default out_dir is outputs/<task>/<name>
    :return:
    """

    config.read(config_file)
    config.training.data = data_utils.resolve_path(config.training.data, additional_dirs=[config_file])
    config.meta.run_out_dir = out_dir or os.path.relpath(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-1] + ["outputs", config.meta.task, config.meta.model, config.meta.run_name]))
    os.makedirs(config.meta.run_out_dir, exist_ok=True)
    try:
        shutil.copy(config_file, config.meta.run_out_dir)
    except shutil.SameFileError:
        pass

    # Process hyperparam strings and make sure they're (psosibly singleton) lists
    hyperparameters = dict(config.items("hyperparameters"))
    for key, val in hyperparameters.items():
        hyperparameters[key] = eval(val) if isinstance(val, str) else val
    for key, val in hyperparameters.items():
        hyperparameters[key] = [val] if not isinstance(val, list) else val

    # Turn off cached features for bert in certain cases
    if config.training.cached_features and config.meta.model == "bert":
        if config.training.only_one_fold:
            global_bert_settings["use_cached_eval_features"] = True
            if config.training.eval_subsample:
                global_bert_settings["use_cached_eval_features"] = False
                logging.warning("Cannot use cached eval features with subsampling.")
            global_bert_settings["reprocess_input_data"] = False
            global_bert_settings["cache_dir"] = os.path.join("cache_dir/", config.meta.run_name)
        else:
            logging.warning("Cached input features is disabled; doesn't work correctly yet with crossvalidation.")

    # Check if file exists & ask if complete/abort/overwrite:
    output_choice = "o"
    if len(os.listdir(config.meta.run_out_dir)) > 1: # > 1 because config file exists
        if os.path.exists(os.path.join(config.meta.run_out_dir, 'results.csv')):
            output_choice = (input(f"Results file {os.path.join(config.meta.run_out_dir, 'results.csv')} exists: [R]ead results.csv only, [c]omplete, [o]verwrite, [q]uit?").lower().strip() + "r")[0]
        else:
            output_choice = (input(f"Output path for {config.meta.run_out_dir} exists: [C]omplete, [o]verwrite, [q]uit?").lower().strip() + "c")[0]
    if output_choice == "q":
        return

    random.seed(config.meta.random_seed)
    np.random.seed(random.randint(0, 99999))

    # Load train data and preprocess according to config
    train_data = pd.read_csv(config.training.data)

    if config.training.subsample:
        train_data = train_data.sample(frac=config.training.subsample if config.training.subsample <= 1 else config.training.subsample/len(train_data))
        # Since index is lost when moving to X and y for tree model:
        if config.meta.model == "tree":
            train_data.reset_index(inplace=True)    # TODO needs more principled fix

    if config.training.context == 'sentence':
        train_data = data_utils.get_highlight_containing_sentence(train_data)
    elif config.training.context == "highlight":
        train_data['context'] = train_data['highlight']
    elif config.training.context == "context-highlight" and config.meta.model == 'bert':
        train_data['context'] = train_data['highlight'] + train_data['highlight']
        # for tree, this is done via the features themselves, not the 'context' column.

    # For cross-validation:
    train_test_splits = data_utils.n_fold_split(train_data, config.training.n_splits,
                                                by='context_id', return_train_test_splits=True,
                                                contiguity=config.training.fold_contiguity, remove_overlap_window=config.training.fold_remove_overlap)
    if config.training.only_one_fold:
        train_test_splits = [train_test_splits[0]]

    # Split the main program between bert and tree:
    if config.meta.model == "bert":

        if output_choice != "r":

            train_data = data_utils.prepare_c_q_binary_for_bert(train_data)

            grid_search_bert(train_data, train_test_splits, hyperparameters,
                             validation_subsample=config.training.eval_subsample,
                             out_dir=config.meta.run_out_dir, output_mode=output_choice,
                             predictions_filename=os.path.split(config.training.data)[1])

        summarize_results_grid_search_bert(config.meta.run_out_dir)

    elif config.meta.model == "tree":

        if output_choice != "r":

            feature_strings = data_utils.read_feature_list_from_config(config.training.features)
            recompute_feature_strings = data_utils.read_feature_list_from_config(config.training.recompute_features)

            models_to_use = {
                "word_vecs_for_schricker": config.training.word_vecs_for_schricker,
                "word_vecs_for_sif": config.training.word_vecs_for_sif,
                "bert_logits": config.training.bert_logits,
            }

            fe = feature_extractor.FeatureExtractor(
                model_paths=models_to_use,
                mode="a" if config.training.cached_features else "w",
                recompute_features=recompute_feature_strings,
                path_to_computed_features=f"data/extracted_features/{config.training.cache_name}",
            )
            X = fe.extract(train_data, features_to_extract=feature_strings)
            y = train_data['evoked?']

            grid_search_random_forest(X, y, train_test_splits, hyperparameters, out_dir=config.meta.run_out_dir, predictions_filename=os.path.split(config.training.data)[1])

        summarize_results_grid_search_random_forest(config.meta.run_out_dir)


def grid_search_random_forest(X, y, train_test_splits, param_grid, out_dir="outputs/temp", predictions_filename="predictions.csv"):

    n_labels = len(y.unique())

    model = RandomForestClassifier(class_weight="balanced")
    model = sklearn.model_selection.GridSearchCV(model,
                                                 param_grid=param_grid,
                                                 cv=train_test_splits,
                                                 scoring={
                                                     "mcc": sklearn.metrics.make_scorer(sklearn.metrics.matthews_corrcoef),
                                                     "acc": sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score),
                                                     "f1": sklearn.metrics.make_scorer(sklearn.metrics.f1_score),
                                                 },
                                                 refit="mcc",
                                                 return_train_score="mcc",
                                                 verbose=1,
                                                 )
    X = X.fillna(X.mean())
    model.fit(X.values, y.values)

    # Save models
    joblib.dump(model, os.path.join(out_dir, "cv_model.joblib"))
    joblib.dump(model.best_estimator_, os.path.join(out_dir, "best_model.joblib"))

    # Save results (with hyperparams as index)
    results = pd.DataFrame(model.cv_results_)
    relevant_hyperparams = list(results['params'][0])
    for param in relevant_hyperparams:
        results[param] = results['params'].apply(lambda x: x[param])
    del results['params']
    results['run_id'] = results.index
    results = results.set_index(relevant_hyperparams, drop=True)
    results.to_csv(os.path.join(out_dir, "results.csv"))

    # Save training set predictions
    model_outputs = model.best_estimator_.predict_proba(X)
    df = pd.DataFrame(model_outputs, columns=[f"logit_{i}" for i in range(n_labels)], index=list(range(len(y))))
    df['prediction'] = model_outputs.argmax(axis=1)
    df['target'] = y
    df.to_csv(os.path.join(out_dir, predictions_filename))

    # Model info:
    feature_importances_dict = {}
    for i in range(len(model.best_estimator_.feature_importances_)):
        feature_importances_dict[X.columns[i]] = model.best_estimator_.feature_importances_[i]
    feature_importances = sorted(feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join(out_dir, 'feature_importances.json'), 'w') as file:
        json.dump(feature_importances, file)


def summarize_results_grid_search_random_forest(out_dir):
    # TODO expand and commentate

    cv_model = joblib.load(os.path.join(out_dir, "cv_model.joblib"))
    best_model = joblib.load(os.path.join(out_dir, "best_model.joblib"))

    # Print some stuff:
    print("Best params:\n", cv_model.best_params_)
    print("Best score:\n", cv_model.best_score_)

    with open(os.path.join(out_dir, 'feature_importances.json')) as file:
        feature_importances = json.load(file)
    print(feature_importances)


def grid_search_bert(data, train_test_splits, param_grid, validation_subsample=None, out_dir="outputs/temp", output_mode="o", predictions_filename="predictions.csv"):
    """
    Perform grid search, finetuning a new BERT model for each hyperparameter setting, collecting results.
    :param train_data:
    :param validation_data:
    :param param_grid: dictionary of hyperparameter: value where value can be a list of multiple values to try.
    :param run_name: outputs will be written to outputs/[run_name]
    :param output_mode: whether to [o]verwrite or [c]omplete existing records.
    :return:
    """
    n_labels = len(data['labels'].unique())

    # Use cartesian product to compute all setting combinations to explore:
    param_grid_as_list = [[(k, w) for w in v] for k, v in sorted(param_grid.items())]
    settings = list(itertools.product(*param_grid_as_list))
    relevant_hyperparams = [key for key, value in param_grid.items() if isinstance(value, list) and len(value) > 1]

    # Loop through all hyperparameter settings, collecting results:
    print(f"Starting to {'run' if output_mode == 'o' else 'complete'} {len(settings)} jobs.")
    results_rows = []
    for run_id, setting in enumerate(settings):
        run_predictions = []

        # Determine run output directory:
        settings_as_string = '_'.join(f"{k}-{v}" for k, v in setting if k in relevant_hyperparams)
        run_output_dir = f"{out_dir}/{run_id}" if len(settings) > 1 else f"{out_dir}"
        if settings_as_string != '':
            run_output_dir += f"_{settings_as_string}"

        # Cross-validation:
        for fold_id, (train_ids, eval_ids) in enumerate(train_test_splits):
            print('-------', run_id, fold_id, "-------\n", setting)
            fold_output_dir = os.path.join(run_output_dir, f"fold_{fold_id}")
            best_model_dir = os.path.join(fold_output_dir, 'best_model')

            # Train BERT if necessary:
            setting_dict = dict(setting)
            if not os.path.exists(fold_output_dir) or output_mode == 'o':
                train_data = data.loc[train_ids, :]
                test_data = data.loc[eval_ids, :]
                eval_data = data.loc[eval_ids, :]
                if validation_subsample:
                    eval_data = eval_data.sample(
                        frac=validation_subsample if validation_subsample <= 1 else validation_subsample/len(eval_data)
                    )

                # Finetune bert model:
                kwargs = {'output_dir': fold_output_dir, 'best_model_dir': best_model_dir, **global_bert_settings}
                kwargs.update(setting_dict)
                model = simpletransformers.classification.ClassificationModel('bert', 'bert-base-cased',    # TODO consider roberta instead.
                                                                              num_labels=n_labels,
                                                                              use_cuda=torch.cuda.is_available(),
                                                                              cuda_device=0, args=kwargs)
                model.train_model(train_data, eval_df=eval_data, show_running_loss=False, **bert_metrics)

                # After finetuning, record best model outputs and predictions:
                model = simpletransformers.classification.ClassificationModel('bert', best_model_dir, num_labels=n_labels,
                                                                              use_cuda=torch.cuda.is_available(),
                                                                              cuda_device=0, args=kwargs)
                _, model_outputs, _ = model.eval_model(eval_data, **bert_metrics)

                df = pd.DataFrame(model_outputs, columns=[f"logit_{i}" for i in range(n_labels)], index=test_data.index)
                df['prediction'] = model_outputs.argmax(axis=1)
                df['target'] = test_data['labels']
                df.to_csv(os.path.join(best_model_dir, "fold_predictions.csv"))
            else:
                print(f"Output dir already exists. Using pre-computed results.")

            # Retrieve training scores from the generated output file:
            training_scores = pd.read_csv(os.path.join(fold_output_dir, 'training_progress_scores.csv'))
            for _, scores in training_scores.iterrows():
                results_rows.append({'run_id': run_id, 'fold_id': fold_id, 'output_dir': fold_output_dir, **setting_dict, **scores})

            # Retrieve best model eval results:
            with open(os.path.join(best_model_dir, 'eval_results.txt')) as file:
                best_results = dict([(s.strip() for s in line.split('=')) for line in file])
            results_rows.append({'run_id': run_id, 'fold_id': fold_id, 'output_dir': fold_output_dir, **setting_dict, 'global_step': 'best', **best_results})

            # And also collect all best model predictions pf this run:
            fold_predictions = pd.read_csv(os.path.join(best_model_dir, "fold_predictions.csv"), index_col=0)
            run_predictions.append(fold_predictions)

        # Concatenate and write all best model predictions of this run:
        run_predictions = pd.concat(run_predictions)
        run_predictions.sort_index(inplace=True)
        run_predictions.to_csv(os.path.join(run_output_dir, predictions_filename))

    # Turn all collected results into a dataframe:
    results = pd.DataFrame(results_rows)

    # Move the more relevant columns to the front
    columns = list(results)
    relevant_identifiers = ['run_id', 'fold_id', 'global_step']
    relevant_scores = ['train_loss', 'eval_loss', 'mcc', 'f1', 'tp', 'fp', 'tn', 'fn']
    for column in (relevant_identifiers + relevant_scores + relevant_hyperparams)[::-1]:
        columns.insert(0, columns.pop(columns.index(column)))
    results = results[columns]

    # Write to file
    results.to_csv(os.path.join(out_dir, 'results.csv'), index=False)
    print(f"Hyperparameter search results written to {os.path.join(out_dir, 'results.csv')}")


def summarize_results_grid_search_bert(out_dir, min_mcc_to_plot=0.0):
    # TODO streamline, clean up, commentate

    results = pd.read_csv(os.path.join(out_dir, 'results.csv'))
    print(f"Hyperparameter search results read from {os.path.join(out_dir, 'results.csv')}")

    # Find out which hyperparameters were varied:
    # TODO ugly manual coding:
    hyperparameters = ['gradient_accumulation_steps','learning_rate','num_train_epochs','train_batch_size', 'warmup_ratio', 'weight_decay']
    relevant_hyperparams = [p for p in hyperparameters if p in results.columns and len(results[p].unique()) > 1]

    # First: Print best model ensemble results (all folds, averaged):
    results_best = results.loc[results['global_step'] == 'best']
    results_best_ensemble = results_best.groupby(['run_id'] + relevant_hyperparams).agg({'mcc': ['max', 'mean', 'min', 'std'], 'f1': ['max', 'mean', 'min', 'std']})
    results_best_ensemble = results_best_ensemble.sort_values(by=[('mcc', 'mean')], ascending=False)
    print(results_best_ensemble.to_string())

    # Print best model ensemble results on average, given certain hyperparameter setting:
    param_combos = [[a] for a in relevant_hyperparams] + \
                   [[a,b] for a in relevant_hyperparams for b in relevant_hyperparams if a != b] + \
                    [relevant_hyperparams]
    for param_combo in param_combos:
        if param_combo:
            results_best_ensemble_aggregated = results_best_ensemble.groupby(param_combo).agg({('mcc', 'mean'): ['max', 'mean', 'min', 'std'], ('f1', 'mean'): ['max', 'mean', 'min', 'std']})
            print(results_best_ensemble_aggregated.to_string())
            print("\n\n")

    # Next: plotting training progress. Plot only sufficiently successful runs:
    results_best.reset_index(drop=False, inplace=True)
    run_ids_to_plot = results_best_ensemble.loc[results_best_ensemble[('mcc', 'mean')] > min_mcc_to_plot].index.get_level_values('run_id')
    results_to_plot = results.loc[results['run_id'].isin(run_ids_to_plot)]
    results_to_plot = results_to_plot.loc[results['global_step'] != "best"] # i.e., not just the best model

    # Aggregate runs across folds again
    results_to_plot = results_to_plot.groupby(['run_id', 'global_step'] + relevant_hyperparams).agg({key: 'mean' for key in ['train_loss', 'eval_loss', 'mcc']}).reset_index(drop=False)

    # Plot for various combination of hyperparameters
    results_to_plot['global_step'] = results_to_plot['global_step'].astype(float).astype(int)
    results_to_plot = results_to_plot.melt(id_vars=["run_id", "global_step"] + relevant_hyperparams, var_name="score", value_name="value")
    for param in relevant_hyperparams + [None]:
        sns.lineplot(data=results_to_plot, hue='score', style=param, x='global_step', y='value', units="run_id", estimator=None, sort=True)
        plt.savefig(f'{out_dir}/plot_{param or "general"}.png')
        plt.close()



if __name__ == "__main__":
    main()

