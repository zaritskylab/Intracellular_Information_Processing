from matplotlib import pyplot
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import shap
import tensorflow
from Miscellaneous.utils import *
from sklearn.linear_model import ElasticNet, ElasticNetCV
from Miscellaneous.consts import *


def elastic_net():
    """

    :return:
    """

    # read the dataset
    df = pd.read_pickle('../PreparedDatasets/dataset_20160820_10A_FB_xy13.pkl')

    to_X = df.drop([LABEL], axis=1)
    X = pd.DataFrame(to_X, columns=to_X.columns)
    y = pd.Series(df[LABEL])

    # split the data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regr = ElasticNet(random_state=0)

    parameters = {
        'alpha': [0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0],
        'l1_ratio': [0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 1.0],
        'fit_intercept': [True, False],
        'normalize': [True, False],
        'max_iter': [200, 500, 1000, 2000],
        'positive': [True, False],
        'selection': ['cyclic', 'random']
    }

    best_model = grid_search(regr, parameters, cv=10)
    best_model.fit(X_train, y_train)
    y_pred = best_model.best_estimator_.predict(X_test)
    print(best_model.best_params_)

    error = calc_distance_metric_between_signals(y_test, y_pred, 'rmse')

    print("elasticNet rmse: " + str(error))


def mlp(file_path: str):

    filename, file_extension = os.path.splitext(file_path)

    if file_extension == '.pkl':
        # read the dataset
        df = pd.read_pickle(file_path)

    if file_extension == '.csv':
        # read the dataset
        df = pd.read_csv(file_path)


    to_X = df.drop([LABEL], axis=1)
    X = pd.DataFrame(to_X, columns=to_X.columns)
    y = pd.Series(df[LABEL])

    # split the data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regr = MLPRegressor(random_state=0, max_iter=1000, hidden_layer_sizes=(20, 40, 20,), alpha=0.0001, batch_size=10)

    # parameters = {
    #     # 'hidden_layer_sizes': [(20, 40, 20, ), (5, 10, 5), (5, 3,)],
    #     # 'activation': ['relu', 'tanh', 'logistic', 'identity'],
    #     # 'solver': ['sgd', 'adam', 'lbfgs'],
    #     # 'alpha': [0.0001, 0.001, 0.005, 0.05, 0.1],
    #     # 'batch_size': [10, 20, 30],
    #     # 'learning_rate': ['constant', 'adaptive', 'invscaling'],
    #     # 'learning_rate_init': [0.001, 0.01, 0.005, 0.008, 0.05]
    # }
    #
    # best_model = grid_search(regr, parameters, cv=3)
    # best_model.fit(X_train, y_train)
    # y_pred = best_model.best_estimator_.predict(X_test)
    # print(best_model.best_params_)

    # regr = MLPRegressor(random_state=0, max_iter=1000, batch_size=10, hidden_layer_sizes=(20, 40, 20,))
    #
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    error = calc_distance_metric_between_signals(y_test, y_pred, 'rmse')

    print("mlp rmse: " + str(error))

    # e = shap.DeepExplainer(regr, X_train)
    # shap_values = e.shap_values(X_test)
    # shap.force_plot(e.expected_value, shap_values)
    # shap.summary_plot(shap_values, X_train, plot_type="bar")


def grid_search(model, parameters: dict, cv: int = None):
    scoring = make_scorer(my_custom_loss_func, greater_is_better=False)
    best_model = GridSearchCV(estimator=model, param_grid=parameters, scoring=scoring, cv=cv)
    return best_model


def my_custom_loss_func(y_true: np.array, y_pred: np.array, metric: str = 'rmse'):
    score = calc_distance_metric_between_signals(y_true, y_pred, metric)
    return score


def feature_importance(importance):
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


if __name__ == '__main__':
    # elastic_net()
    mlp('../PreparedDatasets/dataset_DMEM+7.5uM_erastin.csv')
