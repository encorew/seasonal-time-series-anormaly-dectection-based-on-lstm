import random

from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np


def parse_date(x):
    return datetime.strptime(x, '%Y %m %d %H')


def load_data(path, save_new_csv=False):
    dataset = read_csv(path, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0,
                       date_parser=parse_date)
    # 把No那一列给删了
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names，axis=1表示删除的是列,inplace=True表示在原数据中把该列删除
    dataset.drop('cbwd', axis=1, inplace=True)
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'

    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    for column in dataset.columns:
        dataset[column] = dataset[column].astype(float)
    if save_new_csv:
        if path[-4:] == ".csv":
            dataset.to_csv(path[:-4] + "_new" + ".csv")
        else:
            dataset.to_csv(path + "_new")
    return dataset


def normalization(dataset):
    normalizer = MinMaxScaler(feature_range=(0, 2))
    for column in dataset.columns:
        # dataset[column] = np.array(dataset[column]).reshape(43824, 1)
        dataset[column] = normalizer.fit_transform(np.array(dataset['pollution']).reshape(-1, 1))
        # print("column {} shape {}".format(column, dataset[column].shape))
    return dataset


def preprocess_data(csv_path):
    data = load_data(csv_path, True)
    data = normalization(data)
    manually_adjust(data['pollution'].values)
    return data


# 这个函数用来调整训练集，消除一些异常峰值，自己添加季节性峰值
def manually_adjust(values):
    # values[410:450] = values[370:410]
    # values[445:460] = values[430:445]
    # values[1872:1885] = values[1860:1873]
    # values[1925:1935] = values[1915:1925]
    for i in range(6000, len(values) - 5):
        if values[i] > 0.84:
            rand = random.randint(-30, 30)
            values[i - 5:i + 5] = values[5140 + rand:5150 + rand]
    for i in range(0, len(values)):
        if values[i] == 0:
            values[i] = values[325 + random.randint(-10, 10)]
    values[4650:4665] = values[1050:1065]
    # 构造values[4650:4665]季节性凸起
    values[1050:1065] = values[2230:2245]
    values[15250:15265] = values[4650:4665]
    values[20250:20265] = values[4650:4665]
    values[25250:25265] = values[4650:4665]
    values[30250:30265] = values[4650:4665]
    return values


def show_data_by_plot(dataset):
    values = dataset.values
    # print(values)
    # groups = [0, 1, 2, 3, 4, 5, 6, 7]
    # for i in groups:
    #     # 把plot分成8个块，每块编号为i
    #     plt.subplot(len(groups), 1, i + 1)
    #     plt.plot(values[:, i])
    #     plt.title(dataset.columns[i])
    plt.plot(values[:, 0])
    plt.title("pollution")
    plt.show()


# csv_path = 'D:/codes/Python/LSTM/dataset/pollution.csv'
# data = preprocess_data(csv_path)
# show_data_by_plot(data)
