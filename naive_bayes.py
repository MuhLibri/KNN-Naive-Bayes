try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pdf import *
import utils, implement

Dataset: TypeAlias = list[list[float]]
Summary: TypeAlias = dict[int, Statistics]
ClassSummary: TypeAlias = dict[float, Summary]

_CLEAN_DATA = True
_PDFS = [
    dist_normal,    # Battery power
    dist_nominal,   # Bluetooth
    dist_normal,       # Clock speed
    dist_nominal,   # Dual SIM
    dist_exp,       # Kamera depan
    dist_nominal,   # 4G                # deleted
    dist_normal,    # Memori internal   # deleted
    dist_normal,    # Ketebalan
    dist_normal,    # Berat
    dist_nominal,   # Jumlah core
    dist_normal,    # Kamera utama
    dist_normal,       # Px height
    dist_normal,    # Px width
    dist_normal,    # RAM
    dist_normal,    # Phone height
    dist_normal,    # Phone width
    dist_normal,    # Talk time
    dist_nominal,   # 3G
    dist_nominal,   # Touch screen
    dist_nominal,   # Wi-Fi
]

def _corr(data_1: list[float], data_2: list[float]):
    return pd.Series(data_1).corr(pd.Series(data_2))

def _group_by_class(data: Dataset) -> dict[float, Dataset]:
    group: dict[float, Dataset] = dict()

    for row in data:
        cls = row[-1]
        if cls not in group:
            group[cls] = list()
        group[cls].append(row)
    
    return group

def _statistics_col(data: Dataset, probs: list[PDF], features: list[int]) -> Summary:
    stats = [(x[-1], Statistics(list(x)[:-1], probs[x[-1]])) for x in zip(*data, range(len(probs))) if x[-1] in features]
    summary: Summary = dict()
    for feature, stat in stats:
        summary[feature] = stat
    return summary

def _statistics_class(data: Dataset, probs: list[PDF], features: list[int]) -> ClassSummary:
    group = _group_by_class(data)
    stats: ClassSummary = dict()

    for cls, rows in group.items():
        stats[cls] = _statistics_col(rows, probs, features)
    
    return stats

def _calculate_probabilities(class_statistics: ClassSummary, row: list[float]) -> dict[float, float]:
    total_rows = sum([class_statistics[cls][0].len for cls in class_statistics])
    probs: dict[float, float] = dict()

    for cls, class_stat in class_statistics.items():
        probs[cls] = (class_statistics[cls][0].len) / float(total_rows)

        for col_idx in class_stat.keys():
            probs[cls] *= class_stat[col_idx].probability(row[col_idx])
    
    return probs

def _classify(class_probabilities: dict[float, float]) -> float:
    curr_cls, curr_max_p = None, float('-inf')

    for cls, prob in class_probabilities.items():
        if prob > curr_max_p:
            curr_max_p = prob
            curr_cls = cls
    
    return curr_cls

def _remove_at(li: list, to_delete: list[int]) -> int:
    res = list()
    [res.append(li[i]) for i in range(len(li)) if i not in to_delete]
    return res

def _get_cols(li: list[list]) -> list[list]:
    return [x for x in zip(*li)]

def _remove_cols(li: list[list], to_delete: list[int]) -> list[list]:
    columns = _remove_at([x for x in zip(*li)], to_delete)
    return [list(x) for x in zip(*columns)]

def _find_correlated_cols(data: Dataset, threshold: float = .55) -> list[int]:
    columns = _get_cols(data)

    # Construct a correlation matrix between data features
    corr_mat = [[_corr(columns[i], columns[j]) for j in range(i)] for i in range(len(columns) - 1)]

    # Filter for features that correlate to another feature above a threshold
    to_delete: list[int] = list()
    [[to_delete.append(j) for j in range(i - 1) if corr_mat[i][j] >= threshold] for i in range(len(columns) - 2)]

    return to_delete

def do_train(data: Dataset, prob: list[PDF] = _PDFS) -> ClassSummary:
    features = list(range(len(data[0]) - 1))
    if _CLEAN_DATA:
        to_delete = _find_correlated_cols(data)
        features = [i for i in features if i not in to_delete]

    return _statistics_class(data, prob, features)

def do_predict(data: Dataset, summary: ClassSummary) -> list[float]:
    res: list[float] = list()
    for row in data:
        cls = _classify(_calculate_probabilities(summary, row))
        res.append(cls)
    
    return res

def calculate_accuracy(predicted: list[float], actual: list[float]) -> float:
    return float(sum(
        [int(predicted[i] == actual[i]) for i in range(len(actual))]
    )) / float(len(actual))

def predict(training_data: Dataset, validation_data: Dataset) -> list[float]:
    model = do_train(training_data)
    prediction = do_predict(validation_data, model)

    x_train, x_validation = utils.get_x(training_data), utils.get_x(validation_data)
    train_target, validation_target = utils.get_target(training_data), utils.get_target(validation_data)

    print(f'Manual accuracy: {accuracy_score(validation_target, prediction)}')
    print('Manual precision: ', precision_score(validation_target, prediction, average='micro'))
    print('Manual recall:', recall_score(validation_target, prediction, average='micro'))

    implement.nb_sklearn(x_train, x_validation, train_target, validation_target)

    return prediction

# if __name__ == '__main__':
#     import file

#     _OUT_FILE = 'naive_bayes.csv'

#     df_train, df_valid = file.read_csv('data_train.csv'), file.read_csv('../test.csv')
#     model = do_train(df_train)
#     pred = do_predict(df_train, model)
#     print(calculate_accuracy(pred, list(zip(*df_train))[-1]))

#     pred = do_predict(_remove_cols(df_valid, [0]), model)
#     df_valid_cols = _get_cols(df_valid)
#     result = [[int(df_valid_cols[0][i]), int(pred[i])] for i in range(len(df_valid))]
#     print(f'Result: {result}')
#     print(f'Writing result to output/{_OUT_FILE}')

#     result.sort(key=lambda row: row[1])
#     file.write_prediction_result(result, _OUT_FILE)
