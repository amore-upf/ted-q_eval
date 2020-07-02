import sklearn.metrics
import pandas as pd

MODELS = [
    'bert/tedq_context',
    'bert/tedq_sentence',
    'bert/tedq_highlight',
    'bert/bookcorpus_context',
    'bert/bookcorpus_sentence',
]

PREDICTIONS = [
    'tedq_test_context',
    'tedq_test_sentence',
    'tedq_test_highlight',
    'bookcorpus_test_context',
    'bookcorpus_test_sentence',
    'quds_test_context',
    'quds_test_sentence',
]

def sort_multiindex(multiindex):
    sort_key = ['tedq', 'bookcorpus', 'quds', 'context', 'sentence', 'highlight'].index
    sorted_tuples = sorted(multiindex.tolist(), key=lambda t: 999 * sort_key(t[0]) + sort_key(t[1]))
    return pd.MultiIndex.from_tuples(sorted_tuples)

columns = ['model', 'tested_on', 'mcc']
rows = []
for model in MODELS:
    for prediction in PREDICTIONS:
        model_path = 'outputs/c_q_binary/' + model
        predictions_path = model_path + '/predictions/' + prediction + '.csv'
        preds = pd.read_csv(predictions_path)
        score = sklearn.metrics.matthews_corrcoef(preds['target'], preds['prediction'])
        rows.append([model, prediction, score])

results = pd.DataFrame(rows, columns=columns)

results_pivot = results.pivot(index='tested_on', columns='model')['mcc']

results_pivot.index = pd.MultiIndex.from_tuples([(id.split('_')[0], id.split('_')[2]) for id in results_pivot.index], names=['tested on', ''])
results_pivot.columns = pd.MultiIndex.from_tuples([col.split('/')[1].split('_') for col in results_pivot.columns], names=['trained on', ''])


sorted_index = sort_multiindex(results_pivot.index)
sorted_columns = sort_multiindex(results_pivot.columns)
results_pivot = results_pivot.loc[sorted_index, sorted_columns]

print(results_pivot.to_latex(float_format="{:.2f}".format, column_format='ll|rrr|rr', multirow=True, multicolumn_format='c'))
