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
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split

except:
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import GridSearchCV


import pandas as pd
import matplotlib.pyplot as plt

# from xgboost import XGBClassifier

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


def pre_process_data(data):
    data["FirstName"] = data.Name.apply(lambda n: n.split(",")[0].strip())
    data["Temp"] = data.Name.apply(lambda n: n.split(",")[1])
    data["Title"] = data.Temp.apply(lambda n: n.split(".")[0].strip())

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    data["FarmilyName"] = data.Temp.apply(lambda n: n.split(".")[1].strip())

    data["FarmilySize"] = data["Parch"] + data["SibSp"]

    data['CabinLetter'] = data['Cabin'].apply(lambda x: str(x)[0])
    data["CabinNumber"] = data['Cabin'].apply(lambda x: get_number(str(x)))

    data["Ticket"] = data['Ticket'].apply(lambda x: get_number(str(x)))
    return data

def fillna_data(data):
    # all_data = train.append(test)
    data = data.groupby(data["Pclass"]).apply(lambda d: d.fillna(d["Fare"].mean()))
    data = data.groupby(data["Title"]).apply(lambda d: d.fillna(d["Age"].mean()))

    return data

def post_process_data(data):
    data["Embarked"] = data["Embarked"].factorize()[0]
    data.loc[data['FarmilySize'] >= 4, 'FarmilySize'] = 4

    data['Title'] = data['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'the Countess'],
        'Other')

    # data['AgeRegion'] = pd.cut(data['Age'], 5)
    # data.loc[ data['Age'] <= 16, 'Age'] = 0
    # data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    # data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    # data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    # data.loc[data['Age'] > 64, 'Age'] = 4

    data = data.drop(["Cabin"], axis=1)
    data = data.drop("Temp", axis=1)
    # data = data.drop(["Name", "SibSp", "Parch", "FarmilyName", "FirstName", "CabinNumber"], axis=1)
    data = data.drop(["Name", "SibSp", "Parch", "FarmilyName", "FirstName", "Age", "Ticket", "Embarked", "CabinLetter", "CabinNumber", "FarmilySize"], axis=1)
    return data


def predict():
    origin_data = pd.read_csv("train.csv")

    print("origin_data shape:", origin_data.shape)
    train_data = pre_process_data(origin_data)

    test_data = pd.read_csv("test.csv")
    test_data = pre_process_data(test_data)

    train_data = fillna_data(train_data)
    test_data = fillna_data(test_data)

    train_data = train_data.dropna()
    train_result = train_data.Survived
    train_data = train_data.drop(["Survived", "PassengerId"], axis=1)

    train_data = post_process_data(train_data)
    test_data = post_process_data(test_data)

    # test_data["Age"].fillna(test_data["Age"].mean(), inplace=True)
    test_data = test_data.drop(["PassengerId"], axis=1)


    print("shape:", train_data.shape, test_data.shape)
    print("column", train_data.columns)

    for y in train_data.columns:
        if str(train_data[y].dtype) != "object":
            continue

        label_encoder = preprocessing.LabelEncoder()
        column = train_data.columns.get_loc(y)

        label_transform_data = train_data.ix[:, column].append(test_data.ix[:, column])
        label_encoder.fit(label_transform_data)
        print(y, label_encoder.classes_)
        train_data.ix[:, column] = label_encoder.transform(train_data.ix[:, column])
        test_data.ix[:, column] = label_encoder.transform(test_data.ix[:, column])

    train_result_last = pd.DataFrame(train_result)
    # show(train_data, train_result_last, "Age", "Title")

    train_data_last = train_data

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
                  ensemble.AdaBoostClassifier(n_estimators=70, learning_rate=1.5),
                  # ensemble.AdaBoostClassifier(GaussianNB()),
                  GaussianNB(),
                  SVC(),
                  SGDClassifier(loss="hinge"),
                  NearestCentroid(),
                  ensemble.BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
                  ensemble.GradientBoostingClassifier(),
                  KNeighborsClassifier(10, p=1, weights="distance",  n_jobs=8, algorithm="ball_tree", leaf_size=30),
                  # KNeighborsClassifier(),
                  # XGBClassifier()
                  ]

    estimator_name = "DecisionTreeClassifier"
    estimator = list(filter(lambda e: type(e).__name__ == estimator_name, estimators))[0]

    results = cross_validation.cross_val_score(estimator, train_data_last, train_result_last["Survived"], cv=5)
    print(results, results.mean())

    tuned_parameters = get_tuned_parameter(estimator)
    estimator_para_optimize(estimator, train_data_last, train_result_last, tuned_parameters, para_optimize=True)


    estimator.fit(train_data_last, train_result_last["Survived"])
    train_predict = estimator.predict(train_data_last)
    test_predict = estimator.predict(test_data_after_transform)


    # print("before estimate train data size:", train_data_last.shape)
    # cross_val_score_for_all_estimators(train_data_last, train_result_last, estimators)
    #
    # print(train_data_last.head(2))
    # eclf1 = ensemble.VotingClassifier(estimators=[
    #     ('AdaBoostClassifier', ensemble.AdaBoostClassifier()),
    #     ('RandomForestClassifier', ensemble.RandomForestClassifier()),
    #     ('GradientBoostingClassifier', ensemble.GradientBoostingClassifier()),
    #     # ("GaussianNB", GaussianNB()),
    #     # ("SVC", SVC())
    #     # ("BaggingClassifier", ensemble.BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5))
    #     ], voting='hard')
    # train_predict = eclf1.fit(train_data_last, train_result_last["Survived"]).predict(train_data_last)
    # test_predict = eclf1.predict(test_data_after_transform)

    print("")
    print("accuracy:", metrics.accuracy_score(train_result_last, train_predict))
    print("mcc:", metrics.matthews_corrcoef(train_result_last, train_predict))
    print(metrics.classification_report(train_result_last, train_predict))

    df_result = pd.DataFrame(pd.read_csv("test.csv")["PassengerId"], columns=["PassengerId"])
    df_result["Survived"] = test_predict
    df_result.reindex(["PassengerId", "Survived"])

    print(df_result["Survived"].value_counts())
    print(df_result["Survived"].value_counts(normalize=True))
    df_result.to_csv("result.csv", index=False)


def get_tuned_parameter(estimator):
    estimator_parameter_map = {
        "Adaboost": [{
            "n_estimators": [40, 50, 60, 70],"learning_rate": [1.5]
        }],

        "GradientBoostingClassifier": [{
         #    "loss": ['deviance', "exponential"], "learning_rate": [0.1, 1, 10], "n_estimators": [10, 100, 1000],
         # "subsample": [1.0, 0.5, 0.25, 0.75], "criterion": ['friedman_mse', "mse", "mae"],
         #
         # "min_samples_leaf": [1, 2, 5, 10, 0.05, 0.1, 0.2, 0.3], "min_weight_fraction_leaf": [0, 0.05, 0.1, 0.2, 0.3],
         # "max_depth": [1, 3, 10, 100], "min_impurity_split": [1e-7], "max_features": [1, 2, 3, 4],
         # "max_leaf_nodes": [None, 10, 100]

        "loss": ['deviance'],   #'deviance', "exponential"
        "learning_rate": [0.05],
        "n_estimators":[7, 8, 9],       #[ 8, 10, 12, 14],
        "min_samples_split":[0.2, 0.25, 0.15],            #[2, 5, 10, 0.05, 0.1, 0.15, 0.2, 0.25],
        # "subsample": [1.0, 0.9, 0.8, 0.75, 0.7],
        # "criterion": ['friedman_mse', "mse", "mae"]
        }],

        "KNeighborsClassifier": [{
            "n_neighbors": [2, 3, 5, 10], "weights": ['uniform', "distance"],
            "algorithm": ["ball_tree", "kd_tree", "brute"],
            "leaf_size": [30],
            "p": [1, 2], "n_jobs": [8]
        }],

        "SVC": [{
            "C": [1.0], "kernel": ['rbf'], "degree": [3], "gamma": ['auto'],
            "coef0": [0.0], "shrinking": [True], "probability": [False], "tol": [1e-3],
            "cache_size": [200], "class_weight": [None],
            "verbose": [False], "max_iter": [-1]
        }],

        "RandomForestClassifier": [{
            "criterion": ["gini", "entropy"], "min_samples_leaf": [1, 5, 10], "min_samples_split": [2, 4, 10, 12, 16],
            "n_estimators": [50, 100, 400, 700, 1000]
        }],

        "DecisionTreeClassifier": [{
            # "criterion":  ["gini", "entropy"],
            # "max_depth": [10, 100, 500, 1000],
            # "min_samples_leaf": [1, 5, 10],
            # "min_samples_split": [2, 3, 4, 5, 10],
            "min_weight_fraction_leaf":  [0.07, 0.072, 0.075, 0.077, 0.08],
            "max_leaf_nodes": [10, 30, 40, 50, 60, 80],
        }]
    }


    tuned_parameters = estimator_parameter_map[type(estimator).__name__]
    return tuned_parameters

def get_scores(estimator, x, y):
    yPred = estimator.predict(x)

    accuracy = metrics.accuracy_score(y, yPred)
    precision = metrics.precision_score(y, yPred, pos_label=3, average='macro')
    recall = metrics.recall_score(y, yPred, pos_label=3, average='macro')
    print(accuracy, precision, recall)
    return accuracy

def cross_val_score_for_all_estimators(train_data, train_result, estimators):
    print("")
    for estimator in estimators:
        result = cross_validation.cross_val_score(estimator, train_data, train_result["Survived"], cv=5, scoring=get_scores)
        print(type(estimator).__name__, result.mean())

def estimator_para_optimize(estimator, train_data_last, train_result_last, tuned_parameters, para_optimize):
    # results = cross_validation.cross_val_score(estimator, train_data_last, train_result_last["Survived"], cv=5)
    # print(results, results.mean())

    if not para_optimize:
        return

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