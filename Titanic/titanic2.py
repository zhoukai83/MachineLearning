from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

import pandas as pd

import csv
import numpy as np
import matplotlib.pyplot as plt


def __read_csv_file(file_name, delimiter=",", skip_header=True):
    data = []
    with open(file_name, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if skip_header:
            next(reader)
        map(lambda l: data.append(l), reader)

    data = np.array(data)
    return data


def read_train_data(encode_label_index=None):
    train = __read_csv_file("train.csv")
    if encode_label_index is None:
        encode_label_index = xrange(len(train[0]))

    label_encoder = preprocessing.LabelEncoder()

    print("before label:", train[0])
    for column in encode_label_index:
        train[:, column] = label_encoder.fit_transform(train[:, column])
    print("after label:", train[0])

    # imputer = Imputer(missing_values='',strategy='mean', axis=0)

    result = []
    map(lambda l: result.append(l[1]), train)
    train = np.delete(train, [1, 8], axis=1)
    return train, result


def read_test_data():
    test_data = __read_csv_file("test.csv")
    test_result = __read_csv_file("gender_submission.csv")
    test_result = np.delete(test_result, [0], axis=1)
    return test_data, test_result


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

    from sklearn.model_selection import KFold

    k_fold = KFold(5)

    machine = RandomForestClassifier()
    for train_index, test_index in k_fold:
        print(train_index, test_index)
        train_data = train_data_after_label_transform.iloc[train_index, :]
        train_result = train_result_after_label_transform.iloc[train_index, :]

        test_data = train_data_after_label_transform.iloc[test_index, :]
        test_result = train_result_after_label_transform.iloc[test_index, :]
        machine.fit(train_data, train_result)
        train_accuracy = accuracy_score(train_result, machine.predict(train_data))
        test_predict = machine.predict(test_data)
        test_accuracy = accuracy_score(test_result, test_predict)

    # train_data, test_data, train_result, test_result = train_test_split(train_data, train_result, test_size=0.25)
    #
    # # show(train_data, train_result, "Age", "Fare")
    #
    #
    #
    # # print(train_data.head(10))
    # # print(test_data.head(10))
    #
    # machine.fit(train_data, train_result)
    #
    # train_accuracy = accuracy_score(train_result, machine.predict(train_data))
    print("score on train data:", train_accuracy)
    print("score on test data:", test_accuracy)

    return train_accuracy, test_accuracy

def main():

    for num in xrange(10):
        predict()
    # print(test_data)
    # save_result(test_data, test_predict)

if __name__ == "__main__":
    main()