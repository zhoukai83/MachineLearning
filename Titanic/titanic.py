from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

import csv
import numpy as np

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


def show(train_data, train_result, feature_x_index, feature_y_index):
    import matplotlib.pyplot as plt

    survived = []
    un_survived = []

    train_data = np.nan_to_num(train_data)

    for index in xrange(len(train_result)):
        if train_result[index] == '0':
            survived.append(train_data[index, :])
        else:
            un_survived.append(train_data[index, :])

    survived = np.array(survived)
    feature_x_list = survived[:, feature_x_index]
    feature_y_list = survived[:, feature_y_index]
    feature_x_list[feature_x_list == ""] = "0"
    feature_y_list[feature_y_list == ""] = "0"
    feature_x_list = feature_x_list.astype(np.number)
    feature_y_list = feature_y_list.astype(np.number)
    plt.scatter(feature_x_list, feature_y_list, c="r")

    un_survived = np.array(un_survived)
    un_survived_feature_x_list = un_survived[:, feature_x_index]
    un_survived_feature_y_list = un_survived[:, feature_y_index]
    un_survived_feature_x_list[un_survived_feature_x_list == ""] = "0"
    un_survived_feature_y_list[un_survived_feature_y_list == ""] = "0"
    un_survived_feature_x_list = un_survived_feature_x_list.astype(np.number)
    un_survived_feature_y_list = un_survived_feature_y_list.astype(np.number)
    plt.scatter(un_survived_feature_x_list, un_survived_feature_y_list, c="b")
    plt.show()

def main():
    train_data, train_result = read_train_data()
    # train_data, train_result = read_train_data([3, 4, 8, 10, 11])
    train_data, test_data, train_result, test_result = train_test_split(train_data, train_result, test_size=0.3)

    # test_data, test_result  = read_test_data()

    for t in train_data:
        print(t)
    # print(train_data)
    # print train_result
    # print(test_data)

    show(train_data, train_result, 4, 1)

    machine = RandomForestClassifier()
    machine.fit(train_data, train_result)

    print("score on train data:", accuracy_score(train_result, machine.predict(train_data)))
    test_predict = machine.predict(test_data)
    print("score on test data:", accuracy_score(test_result, test_predict))
    print(test_data)
    save_result(test_data, test_predict)

if __name__ == "__main__":
    main()