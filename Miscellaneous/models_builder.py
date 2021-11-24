import pickle

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




def mlp(file_path: str, metric: str = None, manual_split: bool = False, leave_out: (bool, str) = (False, '')):
    filename, file_extension = os.path.splitext(file_path)

    if file_extension == '.pkl':
        # read the dataset
        df = pd.read_pickle(file_path)

    if file_extension == '.csv':
        # read the dataset
        df = pd.read_csv(file_path)

    if len(df[UNNAMED_COLUMN]) > 0:
        df = df.drop([UNNAMED_COLUMN], axis=1)

    if leave_out[0]:
        df = df.drop([leave_out[1]], axis=1)

    if metric == KL_DIVERGENCE:
        df = adjust_data_to_kl_divergence(df)

    to_X = df.drop([LABEL], axis=1)
    X = pd.DataFrame(to_X, columns=to_X.columns)
    y = pd.Series(df[LABEL])

    if manual_split:
        X_train, X_test, y_train, y_test = my_train_test_split(X, y)

    else:
        X = X.drop([CELL_IDX], axis=1)
        # split the data to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # regr = MLPRegressor(random_state=0, max_iter=1000, hidden_layer_sizes=(20, 40, 20,), alpha=0.0001, batch_size=10)
    #
    # parameters = {
    #     'hidden_layer_sizes': [(20, 40, 20, ), (5, 10, 5), (20, 10, 20,), (20, 10, 5,), (10, 5, 10,)],
    #     # 'activation': ['relu', 'tanh', 'logistic', 'identity'],
    #     # 'solver': ['sgd', 'adam', 'lbfgs'],
    #     'alpha': [0.0001, 0.001, 0.005, 0.01, 0.1],
    #     'batch_size': [10, 20, 30],
    #     # 'learning_rate': ['constant', 'adaptive', 'invscaling'],
    #     # 'learning_rate_init': [0.001, 0.01, 0.005, 0.008, 0.05]
    # }
    #
    # best_model = grid_search(regr, parameters, cv=3)
    # best_model.fit(X_train, y_train)
    # y_pred = best_model.best_estimator_.predict(X_test)
    # print(best_model.best_params_)

    regr = MLPRegressor(random_state=0, max_iter=1000, batch_size=10, hidden_layer_sizes=(20, 40, 20), alpha=0.001)

    regr.fit(X_train, y_train)

    # path_to_save = '../TrainedModels/fitted_mlp_model_superkiller_test_one_cell_' + metric + '.sav'
    # pickle.dump(regr, open(path_to_save, 'wb'))

    y_pred = regr.predict(X_test)

    error = calc_distance_metric_between_signals(y_test, y_pred, metric)

    print("mlp " + metric + ": " + str(error))

    # e = shap.DeepExplainer(regr, X_train)
    # shap_values = e.shap_values(X_test)
    # shap.force_plot(e.expected_value, shap_values)
    # shap.summary_plot(shap_values, X_train, plot_type="bar")


def grid_search(model, parameters: dict, cv: int = None):
    scoring = make_scorer(my_custom_loss_func, greater_is_better=False)
    best_model = GridSearchCV(estimator=model, param_grid=parameters, scoring=scoring, cv=cv)
    return best_model


def my_custom_loss_func(y_true: np.array, y_pred: np.array, metric: str = KL_DIVERGENCE):
    score = calc_distance_metric_between_signals(y_true, y_pred, metric)
    return score


def feature_importance(importance):
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


def my_train_test_split(X, y):
    most_frequent_idx = X[CELL_IDX].mode()
    X_train = X[X[CELL_IDX] != most_frequent_idx[0]]
    X_test = X[X[CELL_IDX] == most_frequent_idx[0]]
    y_train = y[X[CELL_IDX] != most_frequent_idx[0]]
    y_test = y[X[CELL_IDX] == most_frequent_idx[0]]

    X_train = X_train.drop([CELL_IDX], axis=1)
    X_test = X_test.drop([CELL_IDX], axis=1)

    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    # elastic_net()
    mlp('../PreparedDatasets/dataset_with_%_alive_cells_param_DMEM_F12+50ng_mL_superkiller_TRAIL.csv',
        KL_DIVERGENCE,
        manual_split=True,
        leave_out=(True, NUM_ALIVE_CELLS_BY_ALL_CELLS))
