import numpy as np
from sklearn import cross_validation, neighbors, svm, ensemble, tree, linear_model, preprocessing
import pandas as pd


def main():
    df = pd.read_csv('./Testing_Oceans_data.csv')
    df = df.convert_objects(convert_numeric=True)

    prediction_label = 'Sound_Velocity(m/s)'

    X = np.array(df.drop([prediction_label], 1))
    y = np.array(df[prediction_label])

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    evaluations = [
        ('Elastic Net', linear_model.ElasticNet(alpha=0.1), X_train, y_train, X_test, y_test),
        ('Lasso', linear_model.Lasso(alpha=0.1), X_train, y_train, X_test, y_test),
        ('Ridge', linear_model.Ridge(alpha=.1), X_train, y_train, X_test, y_test),
        ('Ensemble Random Forest', ensemble.RandomForestRegressor(), X_train, y_train, X_test, y_test),
        ('Ensemble Extra Trees', ensemble.ExtraTreesRegressor(), X_train, y_train, X_test, y_test),
        ('Ensemble Bagging Regressor', ensemble.BaggingRegressor(), X_train, y_train, X_test, y_test),
        ('Ensemble Gradiant Boosting Regressor', ensemble.GradientBoostingRegressor(), X_train, y_train, X_test, y_test),
        ('Ensemble Ada Boost Regressor', ensemble.AdaBoostRegressor(), X_train, y_train, X_test, y_test),
        ('SVR Kernel Linear', svm.SVR(kernel='linear'), X_train, y_train, X_test, y_test),
        ('SVR Kernel RBF', svm.SVR(kernel='rbf'), X_train, y_train, X_test, y_test)
    ]

    for evaluation in evaluations:
        evaluate(*evaluation)


def evaluate(algorithm, clf, X_train, y_train, X_test, y_test):
    print('Testing', algorithm)

    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(accuracy)
    depth=53.4
    temperature = 28.44
    salinity = 3

    prediction = clf.predict([[ depth, temperature, salinity]])
    print(prediction)
    print()


main()
