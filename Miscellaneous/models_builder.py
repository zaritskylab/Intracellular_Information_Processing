from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import shap
import tensorflow

from Miscellaneous.utils import *
from sklearn.linear_model import ElasticNet, ElasticNetCV


def elastic_net():
    """

    :return:
    """

    # read the dataset
    df = pd.read_pickle('dataset.pkl')

    to_X = df.drop([LABEL], axis=1)
    X = pd.DataFrame(to_X, columns=to_X.columns)
    y = pd.Series(df[LABEL])

    # split the data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    cv_model = ElasticNetCV(cv=10, random_state=0)
    cv_model.fit(X_train, y_train)

    # print('Optimal alpha: %.8f' % cv_model.alpha_)
    # print('Optimal l1_ratio: %.3f' % cv_model.l1_ratio_)
    # print('Number of iterations %d' % cv_model.n_iter_)

    regr = ElasticNet(alpha=cv_model.alpha_, l1_ratio=cv_model.l1_ratio_, max_iter=cv_model.n_iter_)
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    feature_importance(regr.coef_)

    score = calc_distance_metric_between_signals(y_test, y_pred, 'rmse')

    print(score)


def mlp():

    # read the dataset
    df = pd.read_pickle('dataset.pkl')

    to_X = df.drop([LABEL], axis=1)
    X = pd.DataFrame(to_X, columns=to_X.columns)
    y = pd.Series(df[LABEL])

    # split the data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


    # regr = MLPRegressor(random_state=0)
    #
    # parameter_space = {
    #     'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [0.0001, 0.05],
    #     'learning_rate': ['constant', 'adaptive'],
    # }
    #
    # cv_model = GridSearchCV(regr, parameter_space, cv=3)
    # cv_model.fit(X_train, y_train)
    # print('Best parameters found:\n', cv_model.best_params_)

    regr = MLPRegressor(random_state=0, hidden_layer_sizes=(50, 100, 50,))


    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_test)

    score = calc_distance_metric_between_signals(y_test, y_pred, 'rmse')

    # e = shap.DeepExplainer(regr, X_train)
    # shap_values = e.shap_values(X_test)
    # shap.force_plot(e.expected_value, shap_values)
    # shap.summary_plot(shap_values, X_train, plot_type="bar")

    print(score)


def feature_importance(importance):
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


if __name__ == '__main__':
    # elastic_net()
    mlp()
