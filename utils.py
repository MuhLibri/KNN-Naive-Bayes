import csv
import matplotlib.pyplot as plt

# def corr_train(df_train):
#     df_train_1 = df_train.corr()[['price_range']].sort_values(by='price_range', ascending=False).drop(index='price_range')
#     plt.figure(figsize=(22, 14))
#     plt.plot(df_train_1["price_range"], color='blue', linestyle='dashed',
#              marker='o', markerfacecolor='red', markersize=10)
#     plt.title('Correlations of columns to price_range')
#     plt.xlabel('Columns')
#     plt.ylabel('Correlation to price_range')
#     print(df_train_1.loc[df_train_1["price_range"] > 0.1])
#     print()

def get_target(dataset):
    # res = list()
    # for row in dataset:
    #     res.append(row[len(dataset[0]) - 1])
    # return res
    return [row[-1] for row in dataset]

def get_x(dataset):
    # res = dataset
    # for row in res:
    #     row.pop()
    # return res
    return [row[:-2] for row in dataset]

def exclude_id(dataset):
    # res = dataset
    # for row in res:
    #     row.pop(0)
    # return res
    return [row[1:] for row in dataset]

def unpop(dataset, popped):
    # res = dataset
    # for i in range(len(res)):
    #     res[i].append(popped[i])
    # return res
    return [dataset[i] + [popped[i]] for i in range(len(dataset))]

def convert_target_to_int(dataset):
    for row in dataset:
        row[-1] = int(row[-1])
    return dataset


