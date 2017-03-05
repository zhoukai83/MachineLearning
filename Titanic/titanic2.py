from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest

from sklearn import metrics
from sklearn import cross_validation
from sklearn import ensemble
from sklearn import feature_extraction
from sklearn import feature_selection


from sklearn import linear_model
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier



# try:
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
# except:
#     from sklearn.cross_validation import train_test_split
#     from sklearn.cross_validation import GridSearchCV


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from xgboost import XGBClassifier

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

    data["Mother"] = (data["Title"] == "Mrs") & (data["SibSp"] > 0)

    return data

def fill_age_by_machine_learning(data):
    age_df = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    if unknown_age.shape[0] == 0:
        return data

    y = known_age[:, 0]
    X = known_age[:, 1:]

    rfr = ensemble.RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    predictedAges = rfr.predict(unknown_age[:, 1::])
    data.loc[(data.Age.isnull()), 'Age'] = predictedAges
    return data


def fillna_data(data):
    # all_data = train.append(test)
    data = data.groupby(data["Pclass"]).apply(lambda d: d.fillna(d["Fare"].mean()))
    data = data.groupby(data["Title"]).apply(lambda d: d.fillna(d["Age"].mean()))
    data = fill_age_by_machine_learning(data)

    data["Child"] = data["Age"] > 16
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
    data = data.drop(["Name", "Ticket","FarmilyName", "FirstName",], axis=1)
    # data = data.drop(["Name", "SibSp", "Parch", "FarmilyName", "FirstName", "Ticket", "Embarked", "CabinLetter", "CabinNumber", "FarmilySize"], axis=1)
    return data


def transform_data(train_data, test_data):
    for columnName in train_data.columns:
        if str(train_data[columnName].dtype) != "object":
            continue

        label_encoder = preprocessing.LabelEncoder()
        columnIndex = train_data.columns.get_loc(columnName)

        label_transform_data = train_data.ix[:, columnIndex].append(test_data.ix[:, columnIndex])
        label_encoder.fit(label_transform_data)
        print(columnName, label_encoder.classes_)
        train_data.ix[:, columnIndex] = label_encoder.transform(train_data.ix[:, columnIndex])
        test_data.ix[:, columnIndex] = label_encoder.transform(test_data.ix[:, columnIndex])

    # for columnName in ["Age", "Fare"]:
    for columnName in ["Fare"]:
        if columnName not in train_data.columns:
            continue

        columnIndex = train_data.columns.get_loc(columnName)
        label_transform_data = train_data.ix[:, columnIndex].append(test_data.ix[:, columnIndex])

        scaler = preprocessing.StandardScaler()
        scaler.fit(label_transform_data)
        train_data.ix[:, columnIndex] = scaler.transform(train_data.ix[:, columnIndex])
        test_data.ix[:, columnIndex] = scaler.transform(test_data.ix[:, columnIndex])

    train_data, test_data = one_bot_transform(train_data, test_data)

    return train_data, test_data

def one_bot_transform(train_data, test_data):
    removed_column = []
    for columnName in train_data.columns:

        columnIndex = train_data.columns.get_loc(columnName)

        label_transform_data = train_data.ix[:, columnIndex].append(test_data.ix[:, columnIndex])
        unique_column_value = label_transform_data.unique()
        value_count = len(unique_column_value)
        if value_count <= 2 or value_count >= 9:
            continue

        one_hot_encoder = preprocessing.OneHotEncoder()
        one_hot_encoder.fit(label_transform_data.reshape(-1, 1))
        df_train_ob = pd.DataFrame(one_hot_encoder.transform(train_data.ix[:, columnIndex].reshape(-1, 1)).toarray())
        df_test_ob = pd.DataFrame(one_hot_encoder.transform(test_data.ix[:, columnIndex].reshape(-1, 1)).toarray())
        for index in range(value_count):
            df_train_ob.rename(columns={unique_column_value[index]: columnName + "_" + str(unique_column_value[index])},
                               inplace=True)
            df_test_ob.rename(columns={unique_column_value[index]: columnName + "_" + str(unique_column_value[index])},
                              inplace=True)
        train_data = train_data.join(df_train_ob)
        test_data = test_data.join(df_test_ob)

        removed_column.append(columnName)

    for columnName in removed_column:
        train_data.drop(columnName, axis=1, inplace=True)
        test_data.drop(columnName, axis=1, inplace=True)

    return train_data, test_data

def one_hot_transform_using_dummy(data):
    removed_column = []
    df_join = pd.DataFrame()
    for columnName in data.columns:
        columnIndex = data.columns.get_loc(columnName)

        unique_column_value = data[columnName].unique()
        value_count = len(unique_column_value)
        # print("one hot", columnName, columnIndex, value_count)
        if value_count <= 2 or value_count >= 9:
            continue

        df_dummies = pd.get_dummies(data[columnName])

        for index in range(value_count):
            df_dummies.rename(columns={unique_column_value[index]: columnName + "_" + str(unique_column_value[index])},
                              inplace=True)

        removed_column.append(columnName)
        data = data.join(df_dummies)

    for columnName in removed_column:
        data.drop(columnName, axis=1, inplace=True)

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
    train_data, test_data = transform_data(train_data, test_data)

    train_result_last = pd.DataFrame(train_result)
    # show(train_data, train_result_last, "Age", "Title")

    train_data_last = train_data

    # show(train_data_last, train_result_last, "Embarked", "Fare")

    # train_data_last, train_result_last = remove_outliers(train_data_last, train_result_last)

    min_samples_split = 2

    from sklearn.linear_model import LogisticRegression
    estimators = [ensemble.RandomForestClassifier(),
                  DecisionTreeClassifier(),
                  ensemble.AdaBoostClassifier(n_estimators=70, learning_rate=1.5),
                  # ensemble.AdaBoostClassifier(GaussianNB()),
                  GaussianNB(),
                  SVC(),
                  linear_model.SGDClassifier(loss="hinge"),
                  linear_model.LogisticRegression(),
                  NearestCentroid(),
                  ensemble.BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5),
                  ensemble.GradientBoostingClassifier(),
                  KNeighborsClassifier(10, p=1, weights="distance",  n_jobs=8, algorithm="ball_tree", leaf_size=30),
                  # KNeighborsClassifier(),
                  XGBClassifier()
                  ]

    estimator_name = "XGBClassifier"
    estimator = list(filter(lambda e: type(e).__name__ == estimator_name, estimators))[0]

    pre_feature_selection(train_data_last, train_result_last, min_samples_split)
    test_data_after_transform = test_data
    fs = SelectKBest(k=10)
    # train_data_last = fs.fit_transform(train_data_last, train_result_last["Survived"])
    fs = feature_selection.SelectPercentile(percentile=75)
    train_data_last = fs.fit_transform(train_data_last,  train_result_last["Survived"])
    test_data_after_transform = fs.transform(test_data)
    show_feature_selection(estimator, train_data_last, train_result_last["Survived"])


    results = cross_validation.cross_val_score(estimator, train_data_last, train_result_last["Survived"], cv=5)
    print(results, results.mean())
    plot_learning_curve(estimator, estimator_name + "learn curve", train_data_last, train_result_last["Survived"])

    # tuned_parameters = get_tuned_parameter(estimator)
    # estimator_para_optimize(estimator, train_data_last, train_result_last, tuned_parameters, para_optimize=True)


    train_x, test_x, train_y, test_y = train_test_split(train_data_last, train_result_last)
    estimator.fit(train_x, train_y["Survived"])
    test_predict = estimator.predict(test_x)
    print("")
    print(estimator_name, " accuracy:", metrics.accuracy_score(test_y, test_predict))
    print("mcc:", metrics.matthews_corrcoef(test_y, test_predict))
    print(metrics.classification_report(test_y, test_predict))
    print(metrics.confusion_matrix(test_y, test_predict))

    # print("before estimate train data size:", train_data_last.shape)
    # cross_val_score_for_all_estimators(train_data_last, train_result_last, estimators)

    predict_test_data(estimator, train_data_last, train_result_last, test_data_after_transform)

def predict_test_data(estimator, train_data_last, train_result_last, test_data_after_transform):

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
    train_predict = estimator.fit(train_data_last, train_result_last["Survived"]).predict(train_data_last)
    test_predict = estimator.predict(test_data_after_transform)

    print("")
    print("accuracy:", metrics.accuracy_score(train_result_last, train_predict))
    print("mcc:", metrics.matthews_corrcoef(train_result_last, train_predict))
    print(metrics.classification_report(train_result_last, train_predict))
    print(metrics.confusion_matrix(train_result_last, train_predict))

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


from sklearn.learning_curve import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
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
    print("ExtraTreesClassifier feature importance")
    forest = ensemble.ExtraTreesClassifier(random_state=0, min_samples_split=min_samples_split)
    forest.fit(train_data, train_result["Survived"])
    columns_weight = zip(train_data.columns, forest.feature_importances_)
    columns_weight = sorted(columns_weight, key=lambda x: x[1], reverse=True)
    print("\r\n".join([str(c) for c in columns_weight]))

    print("")
    print("LogisticRegression feature importance")
    regression = linear_model.LogisticRegression()
    regression.fit(train_data, train_result["Survived"])
    result = pd.DataFrame({"columns": list(train_data.columns), "coef": list(regression.coef_.T)})
    print(result.sort_values("coef"))


def show_feature_selection(estimator, train_data, train_result):
    results = []
    std = []
    percentiles = range(5, 90, 3)
    for i in range(5, 90, 3):
        fs = feature_selection.SelectPercentile(percentile=i)
        train_data_fs = fs.fit_transform(train_data, train_result)
        scores = cross_validation.cross_val_score(estimator, train_data_fs, train_result, cv=5)
        results = np.append(results, scores.mean())
        std = np.append(std, scores.std())

    optimize_percentile = np.where(results == results.max())[0]
    print("Optimize:", optimize_percentile, np.array(percentiles)[optimize_percentile], np.array(results)[optimize_percentile])

    plt.figure()
    plt.xlabel("Num")
    plt.ylabel("Accu")
    plt.fill_between(percentiles, results - std, results + std, alpha=0.1, color="b")
    plt.plot(percentiles, results)

def main():
    for num in range(1):
        predict()
        # print(test_data)
        # save_result(test_data, test_predict)


if __name__ == "__main__":
    main()