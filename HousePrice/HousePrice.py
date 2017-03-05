import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import feature_selection


import xgboost

def fill_data(data):
    data["LotFrontage"].fillna(data["LotFrontage"].mean(), inplace=True)
    data["MasVnrType"].fillna("Unknown", inplace=True)

    for columnName in data.columns:
        data_type = str(data[columnName].dtype)
        if data_type == "float64":
            data[columnName].fillna(data[columnName].mean(), inplace=True)
        elif data_type == "int64":
            data[columnName].fillna(data[columnName].mean(), inplace=True)
        elif data_type == "object":
            data[columnName].fillna("Unknown", inplace=True)


def transform_one_bot(train, test):
    removed_column = []
    for column_name in train.columns:
        print(column_name)
        column_index = train.columns.get_loc(column_name)
        label_transform_data = train.ix[:, column_index].append(test.ix[:, column_index])
        unique_column_value = label_transform_data.unique()
        value_count = len(unique_column_value)
        if value_count <= 2 or value_count >= 9:
            continue

        one_hot_encoder = preprocessing.OneHotEncoder()
        one_hot_encoder.fit(label_transform_data.reshape(-1, 1))
        df_train_ob = pd.DataFrame(one_hot_encoder.transform(train.ix[:, column_index].reshape(-1, 1)).toarray())
        df_test_ob = pd.DataFrame(one_hot_encoder.transform(test.ix[:, column_index].reshape(-1, 1)).toarray())
        for index in range(value_count):
            new_column_name = column_name + "_" + str(unique_column_value[index])
            print(new_column_name)
            df_train_ob.rename(columns={unique_column_value[index]: new_column_name}, inplace=True)
            df_test_ob.rename(columns={unique_column_value[index]: new_column_name}, inplace=True)
        train = train.join(df_train_ob)
        test = test.join(df_test_ob)

        removed_column.append(column_name)

    for column_name in removed_column:
        train.drop(column_name, axis=1, inplace=True)
        test.drop(column_name, axis=1, inplace=True)

    return train, test


def transform_data(train, test):
    for column_name in train.columns:
        column_data_type = str(train[column_name].dtype)
        if column_data_type != "object":
            continue

        label_encoder = preprocessing.LabelEncoder()
        column_index = train.columns.get_loc(column_name)

        label_transform_data = train.ix[:, column_index].append(test.ix[:, column_index])
        label_encoder.fit(label_transform_data)
        train.ix[:, column_index] = label_encoder.transform(train.ix[:, column_index])
        test.ix[:, column_index] = label_encoder.transform(test.ix[:, column_index])

    train, test = transform_one_bot(train, test)
    for column_name in train.columns:
        column_data_type = str(train[column_name].dtype)
        if not (column_data_type == "int64" or column_data_type == "float64"):
            continue

        column_index = train.columns.get_loc(column_name)
        label_transform_data = train.ix[:, column_index].append(test.ix[:, column_index])

        scaler = preprocessing.StandardScaler()
        scaler.fit(label_transform_data)
        train.ix[:, column_index] = scaler.transform(train.ix[:, column_index])
        test.ix[:, column_index] = scaler.transform(test.ix[:, column_index])

    return train, test


def plot_learning_curve(estimator, X, y, ylim=None, cv=None, n_jobs=8,
                        train_sizes=np.linspace(.05, 1., 10), verbose=0, plot=True):
    """
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    """
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title("learning curve")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"train")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"testing")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


def plot_feature_selection_curve(estimator, train_data, train_result, n_jobs=4):
    results = []
    std = []
    percentiles = range(5, 90, 5)
    for i in range(5, 90, 5):
        fs = feature_selection.SelectPercentile(percentile=i)
        train_data_fs = fs.fit_transform(train_data, train_result)
        scores = model_selection.cross_val_score(estimator, train_data_fs, train_result, cv=4, n_jobs=n_jobs)
        results = np.append(results, scores.mean())
        std = np.append(std, scores.std())

    optimize_percentile = np.where(results == results.max())[0]
    print("Optimize:", optimize_percentile, np.array(percentiles)[optimize_percentile], np.array(results)[optimize_percentile])

    plt.figure()
    plt.xlabel("Num")
    plt.ylabel("Accu")
    plt.fill_between(percentiles, results - std, results + std, alpha=0.1, color="b")
    plt.plot(percentiles, results)
    plt.show()

def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    fill_data(train)
    fill_data(test)

    train_result = train.SalePrice
    train = train.drop("SalePrice", axis=1)
    print(train.shape, test.shape)
    train, test = transform_data(train, test)

    estimators = [
        linear_model.LogisticRegression(),
        linear_model.Lasso(),
        linear_model.Ridge(),
        ensemble.GradientBoostingRegressor(),
        ensemble.ExtraTreesRegressor(),
        ensemble.RandomForestRegressor(),
        ensemble.AdaBoostRegressor(),
        xgboost.XGBRegressor()
    ]

    estimator_name = "GradientBoostingRegressor"
    estimator = list(filter(lambda e: str(type(e).__name__) == estimator_name, estimators))[0]

    # plot_feature_selection_curve(estimator, train, train_result)
    # plot_learning_curve(estimator, train, train_result, n_jobs=8)
    train_x, test_x, train_y, test_y =  model_selection.train_test_split(train, train_result, test_size=0.3)

    feature_selector = feature_selection.SelectPercentile(percentile=75)
    feature_selector.fit(train, train_result)
    train_x = feature_selector.transform(train_x)
    test_x = feature_selector.transform(test_x)

    estimator.fit(train_x, train_y)
    print("predict")
    predict_y = estimator.predict(test_x)

    print("score:", estimator.score(test_x, test_y))
    print("r2:", metrics.r2_score(test_y, predict_y))


if __name__ == "__main__":
    main()

