from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef

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


def save_result(test_data, predict):
    with open('result.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['PassengerId'] + ['Survived'])
        for index in xrange(len(test_data)):
            spamwriter.writerow([test_data[index][0], predict[index]])


def show(train_data, train_result, feature_x_name, feature_y_name):
    survived = train_data[train_result == 1]
    un_survived = train_data[train_result == 0]

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


def predict():
    origin_data = pd.read_csv("train.csv")

    train_data = origin_data.drop(["Cabin"], axis=1)
    # origin_data = origin_data.fillna(0)
    train_data = train_data.dropna()

    train_result = train_data.Survived
    train_data = train_data.drop(["Survived"], axis=1)

    encode_label_index = ["Sex", "Embarked", "Ticket", "Name"]
    for column in encode_label_index:
        label_encoder = preprocessing.LabelEncoder()
        column = train_data.columns.get_loc(column)
        train_data.ix[:, column] = label_encoder.fit_transform(train_data.ix[:, column])
        # print(label_encoder.classes_)

    train_data_after_label_transform = train_data
    train_result_after_label_transform = train_result

    k_fold = KFold(train_data_after_label_transform.shape[0], n_folds=5, shuffle=True)

    machine = RandomForestClassifier()
    # machine = DecisionTreeClassifier()
    # machine = AdaBoostClassifier()
    # machine = GaussianNB()
    # machine = SVC(kernel="rbf")
    for train_index, test_index in k_fold:
        train_data = train_data_after_label_transform.iloc[train_index, :]
        train_result = train_result_after_label_transform.iloc[train_index]

        test_data = train_data_after_label_transform.iloc[test_index, :]
        test_result = train_result_after_label_transform.iloc[test_index]
        machine.fit(train_data, train_result)
        train_accuracy = accuracy_score(train_result, machine.predict(train_data))
        test_predict = machine.predict(test_data)
        test_accuracy = accuracy_score(test_result, test_predict)
        print("score on train data:", train_accuracy)
        print("score on test data:", test_accuracy)

    print("mcc:", matthews_corrcoef(test_result, test_predict))
    print("report:", classification_report(test_result, test_predict))

    return train_accuracy, test_accuracy


def main():

    for num in range(1):
        predict()
    # print(test_data)
    # save_result(test_data, test_predict)

if __name__ == "__main__":
    main()