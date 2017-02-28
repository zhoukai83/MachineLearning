from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn import cross_validation

try:
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import KFold
except:
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import KFold

from sklearn.preprocessing import Imputer

import pandas as pd

import csv
import numpy as np
import matplotlib.pyplot as plt


def show(train_data, train_result, feature_x_name, feature_y_name):
    survived = train_data[train_result["Survived"] == 1]
    un_survived = train_data[train_result["Survived"] == 0]

    feature_x_index = train_data.columns.get_loc(feature_x_name)
    feature_y_index = train_data.columns.get_loc(feature_y_name)
    x_label = survived.columns[feature_x_index]
    y_label = survived.columns[feature_y_index]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.text(80, 0, "survived: red")

    plt.scatter(survived[x_label], survived[y_label], c="r")
    plt.scatter(un_survived[x_label], un_survived[y_label], c="b")
    plt.show()


def __scorer(estimator, X, y):
    Y = estimator.predict(X)
    print("accuracy:", metrics.accuracy_score(y, Y))
    print("mcc:", metrics.matthews_corrcoef(y, Y))
    print("remprt:", metrics.classification_report(y, Y))
    return metrics.accuracy_score(y, Y)


def predict():
    origin_data = pd.read_csv("train.csv")
    train_data = origin_data.drop(["Cabin"], axis=1)
    train_data = train_data.dropna()
    train_result = train_data.Survived
    train_data = train_data.drop(["Survived"], axis=1)

    test_data = pd.read_csv("test.csv")
    test_data["Age"].fillna(test_data["Age"].mean(), inplace=True)
    test_data = test_data.groupby(test_data["Pclass"]).apply(lambda d: d.fillna(d["Fare"].mean()))
    test_data = test_data.drop(["Cabin"], axis=1)

    encode_label_index = ["Sex", "Embarked", "Ticket", "Name"]
    for column in encode_label_index:
        label_encoder = preprocessing.LabelEncoder()
        column = train_data.columns.get_loc(column)

        label_transform_data = train_data.ix[:, column].append(test_data.ix[:, column])
        label_encoder.fit(label_transform_data)
        train_data.ix[:, column] = label_encoder.transform(train_data.ix[:, column])
        test_data.ix[:, column] = label_encoder.transform(test_data.ix[:, column])
        # print(label_encoder.classes_)

    train_data_last = train_data
    train_result_last = pd.DataFrame(train_result)
    # show(train_data_after_label_transform, train_result_after_label_transform, "Age", "Fare")

    train_data_last, train_result_last = remove_outliers(train_data_last, train_result_last)

    min_samples_split = 2

    pre_feature_selection(train_data_last, train_result_last, min_samples_split)
    feature_selection = SelectKBest(k=7)
    train_data_last = feature_selection.fit_transform(train_data_last, train_result_last["Survived"])
    test_data_after_transform = feature_selection.transform(test_data)

    estimator = RandomForestClassifier(min_samples_split=min_samples_split)
    # estimator = DecisionTreeClassifier()
    # estimator = AdaBoostClassifier()
    # estimator = GaussianNB()
    # estimator = SVC(kernel="rbf")
    # estimator = SVC(kernel="poly")

    tuned_parameters = [{'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9]}]
    test_estimator(estimator, train_data_last, train_result_last, tuned_parameters)

    estimator.fit(train_data_last, train_result_last["Survived"])
    train_predict = estimator.predict(train_data_last)
    print("")
    print("accuracy:", metrics.accuracy_score(train_result_last, train_predict))
    print("mcc:", metrics.matthews_corrcoef(train_result_last, train_predict))
    print("remprt:", metrics.classification_report(train_result_last, train_predict))

    test_predict = estimator.predict(test_data_after_transform)

    df_result = pd.DataFrame(test_predict, columns=["Survived"])
    df_result["PassengerId"] = test_data["PassengerId"]
    df_result.reindex(["PassengerId", "Survived"])
    df_result.to_csv("result.csv", index=False)


def test_estimator(estimator, train_data_last, train_result_last, tuned_parameters):
    results = cross_validation.cross_val_score(estimator, train_data_last, train_result_last["Survived"], cv=5)
    print(results, results.mean())

    from sklearn.model_selection import GridSearchCV
    clf = GridSearchCV(estimator, tuned_parameters, cv=5, scoring='precision_macro')
    clf.fit(train_data_last, train_result_last["Survived"])

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print("")

    print("Detailed classification report:")
    print("")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print("")
    y_true, y_pred = train_result_last, clf.predict(train_data_last)
    print(metrics.classification_report(y_true, y_pred))
    print("")


def remove_outliers(train_data, train_result):
    clf = IsolationForest()
    clf.fit(train_data)
    outlier = clf.predict(train_data)
    train_data["Outlier"] = outlier
    train_result["Outlier"] = outlier
    # show(train_data[train_data["Outlier"] == -1], train_result[train_result["Outlier"] == -1], "Age", "Fare")
    # show(train_data[train_data["Outlier"] == 1], train_result[train_result["Outlier"] == 1], "Age", "Fare")
    train_data = train_data[train_data["Outlier"] != -1]
    train_result = train_result[train_result["Outlier"] != -1]
    train_data = train_data.drop(["Outlier"], axis=1)
    train_result = train_result.drop(["Outlier"], axis=1)
    return train_data, train_result


def pre_feature_selection(train_data, train_result, min_samples_split):
    forest = ExtraTreesClassifier(random_state=0, min_samples_split=min_samples_split)
    forest.fit(train_data, train_result["Survived"])
    columns_weight = zip(train_data.columns, forest.feature_importances_)
    columns_weight = sorted(columns_weight, key=lambda x: x[1], reverse=True)
    print("\r\n".join([str(c) for c in columns_weight]))


def main():
    for num in range(1):
        predict()
        # print(test_data)
        # save_result(test_data, test_predict)


if __name__ == "__main__":
    main()