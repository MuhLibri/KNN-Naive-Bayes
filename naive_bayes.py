import math

class _Statistics:
    def __init__(self, data: list[float]):
        self.mean = _mean(data)
        self.std = _std(data)
        self.len = len(data)
    
    def probability(self, x: float) -> float:
        pow = -.5 * ((x - self.mean) / self.std)**2
        return 1 / (self.std * math.sqrt(2 * math.pi)) * math.exp(pow)
    
    def __repr__(self) -> str:
        return f'(mean:{self.mean} std:{self.std} len:{self.len})'

def _mean(data: list[float]) -> float:
    return sum(data) / float(len(data))

def _std(data: list[float]) -> float:
    u = _mean(data)
    var = sum([(x - u)**2 for x in data]) / float(len(data) - 1)
    return math.sqrt(var)

def _group_by_class(data: list[list[float]]) -> dict[float, list[list[float]]]:
    group: dict[float, list[list[float]]] = dict()

    for row in data:
        cls = row[-1]
        if cls not in group:
            group[cls] = list()
        group[cls].append(row)
    
    return group

def _statistics_col(data: list[list[float]]) -> list[_Statistics]:
    stats = [_Statistics(x) for x in zip(*data)]
    return stats[:-1]

def _statistics_class(data: list[list[float]]) -> dict[float, list[_Statistics]]:
    group = _group_by_class(data)
    stats: dict[float, list[_Statistics]] = dict()

    for cls, rows in group.items():
        stats[cls] = _statistics_col(rows)
    
    return stats

def _calculate_probabilities(class_statistics: dict[float, list[_Statistics]], row: list[float]) -> dict[float, float]:
    total_rows = sum([class_statistics[cls][0].len for cls in class_statistics])
    probs: dict[float, float] = dict()

    for cls, stats in class_statistics.items():
        probs[cls] = class_statistics[cls][0].len / float(total_rows)

        for col_idx in range(len(stats)):
            probs[cls] *= stats[col_idx].probability(row[col_idx])
    
    return probs

def _classify(class_probabilities: dict[float, float]) -> float:
    curr_cls, curr_max_p = None, -1.

    for cls, prob in class_probabilities.items():
        if prob > curr_max_p:
            curr_max_p = prob
            curr_cls = cls
    
    return curr_cls

def _test(class_statistics: dict[float, list[_Statistics]], test_data: list[list[float]]):
    hit: dict[float, tuple[int, int]] = dict()
    for row in test_data:
        class_probs = _calculate_probabilities(class_statistics, row)
        predict = _classify(class_probs)
        correct = row[-1]

        if predict == correct:
            if predict not in hit: hit[predict] = (0, 0)
            hit[predict] = (hit[predict][0] + 1, hit[predict][1])
        
        if correct not in hit: hit[correct] = (0, 0)
        hit[correct] = (hit[correct][0], hit[correct][1] + 1)
    
    result: dict[float, float] = dict()
    for cls, (correct, total) in hit.items():
        result[cls] = float(correct) / float(total)

    return result

def naive_bayes(train: list[list[float]], test: list[list[float]]):
    class_stats = _statistics_class(train)
    print(f'Training data accuracy: {_test(class_stats, train)}')
    print(f'Validation data accuracy: {_test(class_stats, test)}')

if __name__ == '__main__':
    import file
    df_train, df_valid = file.read_csv('data_train.csv'), file.read_csv('data_validation.csv')
    naive_bayes(df_train, df_valid)
