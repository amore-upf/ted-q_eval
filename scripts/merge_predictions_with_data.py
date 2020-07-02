import click
import pandas as pd
import os
import sys
import sklearn.metrics
from localconfig import config
import data_utils
import feature_extractor

# Always append project's main dir (1 level up), for easy imports
main_project_dir = os.sep.join(os.path.normpath(__file__).split(os.sep)[:-1])
sys.path.append(main_project_dir)

@click.command()
@click.argument('path_to_data', type=click.Path(exists=True), required=True)
@click.argument('paths_to_predictions', type=click.Path(exists=True), nargs=-1)
@click.option('--out_path', type=click.Path(), required=False, default=None)
@click.option('--features', is_flag=True)
def main(path_to_data, paths_to_predictions, out_path, features):
    """
    Example use (to be run from main project directory):
    > python scripts/merge_predictions_with_data.py data/tasks/c_q_binary/tedq_train.csv ../outputs/c_q_binary/bert/tedq_context_final/0/predictions.csv
    """

    if features:
        config_path = data_utils.resolve_path('config/analysis/data_merging.ini')
        config.read(config_path)
        feature_strings = data_utils.read_feature_list_from_config(config.features_to_include.features)
        print(f"Will include the following features, as specified in {config_path}:")
        print("\n - " + "\n - ".join(feature_strings))
        models_to_use = {
            "word_vecs_for_schricker": config.features_to_include.word_vecs_for_schricker,
            "word_vecs_for_SIF": config.features_to_include.word_vecs_for_SIF,
            "bert_logits": config.features_to_include.bert_logits,
        }
        models_to_use = {key: data_utils.resolve_path(value) for key,value in models_to_use.items()}

    out_path = out_path or "predictions_with_data.csv"
    data = pd.read_csv(path_to_data, index_col=False)

    if len(paths_to_predictions) == 1:
        column_headers = [os.path.split(paths_to_predictions[0])[1]]
    else:
        column_headers = [p[len(os.path.commonprefix(paths_to_predictions)):-4] for p in paths_to_predictions]

    for path, header in zip(paths_to_predictions, column_headers):

        print(f"\nMerging {path}")

        preds = pd.read_csv(path, index_col=0, comment='#')
        preds.columns = [f"{col}:{header}" for col in preds.columns]

        data = data.join(preds, how="left", rsuffix=f":{header}")

        data_nona = data[['evoked?', f'target:{header}']].dropna()    # allow partial predictions file
        if sum(data_nona['evoked?'] != data_nona[f'target:{header}']) > 0:
            print(f" ERROR: 'target:{header}' and 'evoked?' in dataset do not align. Quitting.")
            quit()
        del data[f'target:{header}']

        data[f'prediction:{header}'].replace({1.0: True, 0.0: False}, inplace=True)  # conversion maintaining NaNs

        # Compute and print score as a sanity check
        data_nona = data[['evoked?', f'prediction:{header}']].dropna().astype(bool)
        print(f"Merged as column prediction:{header}. MCC score:", sklearn.metrics.matthews_corrcoef(data_nona['evoked?'], data_nona[f'prediction:{header}']))

    # Reorder columns
    for header in (['evoked?'] + [f'prediction:{header}' for header in column_headers])[::-1]:
        columns = list(data)
        columns.insert(0, columns.pop(columns.index(header)))
        data = data[columns]

    if features:
        fe = feature_extractor.FeatureExtractor(
            model_paths=models_to_use,
            mode="a" if config.features_to_include.cached_features else "w",
        )
        X = fe.extract(data, features_to_extract=feature_strings)
        data = data.join(X, how="left")

    print('Columns:\n -', '\n - '.join(data.columns))

    data.to_csv(out_path, index=False)
    print(f"\nAll predictions merged with data. Output written to {out_path}.")


if __name__ == "__main__":
    main()