from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest

from sklearn import metrics
from sklearn import cross_validation
from sklearn import ensemble


from sklearn.linear_model import SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier


try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split


import pandas as pd
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


import re
def get_number(x):
    match = re.search(r"\d\d+", x, re.IGNORECASE)
    if match:
        result = match.group()
    else:
        result = "0"
    return int(result)


def pre_process_data(origin_data):
    data = origin_data.groupby(origin_data["Pclass"]).apply(lambda d: d.fillna(d["Fare"].mean()))

    data["FirstName"] = data.Name.apply(lambda n: n.split(",")[0].strip())
    data["Temp"] = data.Name.apply(lambda n: n.split(",")[1])
    data["Title"] = data.Temp.apply(lambda n: n.split(".")[0].strip())

    data['Title'] = data['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'the Countess'],
        'Other')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    data["FarmilyName"] = data.Temp.apply(lambda n: n.split(".")[1].strip())

    data["FarmilySize"] = data["Parch"] + data["SibSp"]

    data['CabinLetter'] = data['Cabin'].apply(lambda x: str(x)[0])
    data["CabinNumber"] = data['Cabin'].apply(lambda x: get_number(str(x)))

    data["Ticket"] = data['Ticket'].apply(lambda x: get_number(str(x)))

    data["Embarked"] = data["Embarked"].factorize()[0]

    data.loc[data['FarmilySize'] >= 4, 'FarmilySize'] = 4

    #
    data = data.drop(["Cabin"], axis=1)
    data = data.drop("Temp", axis=1)
    # data = data.drop(["Name", "SibSp", "Parch", "FarmilyName", "FirstName", "CabinNumber"], axis=1)
    data = data.drop(["Name", "SibSp", "Parch", "FarmilyName", "FirstName"], axis=1)
    return data


def predict():
    origin_data = pd.read_csv("train.csv")
    print("origin_data shape:", origin_data.shape)
    train_data = pre_process_data(origin_data)
    train_data = train_data.dropna()
    train_result = train_data.Survived
    train_data = train_data.drop(["Survived", "PassengerId"], axis=1)

    test_data = pd.read_csv("test.csv")
    test_data = pre_process_data(test_data)
    test_data["Age"].fillna(test_data["Age"].mean(), inplace=True)
    test_data = test_data.drop(["PassengerId"], axis=1)

    print("shape:", train_data.shape, test_data.shape)
    print("column", train_data.columns)

    for y in train_data.columns:
        if train_data[y].dtype != "object":
            continue

        print(y, train_data[y].dtype)

        label_encoder = preprocessing.LabelEncoder()
        column = train_data.columns.get_loc(y)

        label_transform_data = train_data.ix[:, column].append(test_data.ix[:, column])
        label_encoder.fit(label_transform_data)
        train_data.ix[:, column] = label_encoder.transform(train_data.ix[:, column])
        test_data.ix[:, column] = label_encoder.transform(test_data.ix[:, column])


    train_data_last = train_data
    train_result_last = pd.DataFrame(train_result)
    print(train_data_last.head(10))
    # show(train_data_last, train_result_last, "Embarked", "Fare")

    # train_data_last, train_result_last = remove_outliers(train_data_last, train_result_last)

    min_samples_split = 2

    pre_feature_selection(train_data_last, train_result_last, min_samples_split)

    test_data_after_transform = test_data
    # feature_selection = SelectKBest(k=7)
    # train_data_last = feature_selection.fit_transform(train_data_last, train_result_last["Survived"])
    # test_data_after_transform = feature_selection.transform(test_data)

    from sklearn.linear_model import LogisticRegression
    estimators = [ensemble.RandomForestClassifier(),
                  DecisionTreeClassifier(),
                  ensemble.AdaBoostClassifier(),
                  GaussianNB(),
                  SVC(),
                  SGDClassifier(loss="hinge"),
                  NearestCentroid(),
                  ensemble.BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
                  ensemble.GradientBoostingClassifier(),
                  KNeighborsClassifier(3),
                  ]

    estimator_name = "RandomForestClassifier"
    estimator = list(filter(lambda e: type(e).__name__ == estimator_name, estimators))[0]

    tuned_parameters = [{'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9]}]
    print("before estimate train data size:", train_data_last.shape)
    cross_val_score_for_all_estimators(train_data_last, train_result_last, estimators)
    # test_estimator(estimator, train_data_last, train_result_last, tuned_parameters, para_optimize=False)

    print(train_data_last.head(10))
    estimator.fit(train_data_last, train_result_last["Survived"])
    train_predict = estimator.predict(train_data_last)
    print("")
    print("accuracy:", metrics.accuracy_score(train_result_last, train_predict))
    print("mcc:", metrics.matthews_corrcoef(train_result_last, train_predict))
    print("remprt:", metrics.classification_report(train_result_last, train_predict))

    test_predict = estimator.predict(test_data_after_transform)

    df_result = pd.DataFrame(pd.read_csv("test.csv")["PassengerId"], columns=["PassengerId"])
    df_result["Survived"] = test_predict
    df_result.reindex(["PassengerId", "Survived"])

    print(df_result["Survived"].value_counts())
    print(df_result["Survived"].value_counts(normalize=True))
    df_result.to_csv("result.csv", index=False)


def cross_val_score_for_all_estimators(train_data, train_result, estimators):
    print("")
    for estimator in estimators:
        result = cross_validation.cross_val_score(estimator, train_data, train_result["Survived"], cv=5)
        print(type(estimator).__name__, result.mean())

def test_estimator(estimator, train_data_last, train_result_last, tuned_parameters, para_optimize):
    results = cross_validation.cross_val_score(estimator, train_data_last, train_result_last["Survived"], cv=5)
    print(results, results.mean())

    if not para_optimize:
        return
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
    clf = ensemble.IsolationForest()
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
    forest = ensemble.ExtraTreesClassifier(random_state=0, min_samples_split=min_samples_split)
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