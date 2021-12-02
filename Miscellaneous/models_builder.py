import pickle

from matplotlib import pyplot
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import shap
import tensorflow
from sklearn.svm import SVR

from Miscellaneous.utils import *
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegression
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




def mlp(file_path: str,
        metric: str = None,
        num_of_cells_split: int = None,
        leave_out: (bool, str) = (False, '')):
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
        df = adjust_data_to_positive_values(df)

    to_X = df.drop([LABEL], axis=1)
    X = pd.DataFrame(to_X, columns=to_X.columns)
    y = pd.Series(df[LABEL])

    if num_of_cells_split is not None:
        X_train, X_test, y_train, y_test = my_train_test_split(X, y, num_of_cells_split)

    else:
        X = X.drop([CELL_IDX], axis=1)
        # split the data to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    regr = MLPRegressor(random_state=0, max_iter=1000)

    parameters = {
        'hidden_layer_sizes': [(20, 40, 20, ), (5, 10, 5), (20, 10, 20,), (20, 10, 5,), (10, 5, 10,)],
        # 'activation': ['relu', 'tanh', 'logistic', 'identity'],
        # 'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.0001, 0.001, 0.005, 0.01, 0.1],
        'batch_size': [10, 20, 30],
        # 'learning_rate': ['constant', 'adaptive', 'invscaling'],
        # 'learning_rate_init': [0.001, 0.01, 0.005, 0.008, 0.05]
    }

    best_model = grid_search(regr, parameters, cv=3)
    best_model.fit(X_train, y_train)
    y_pred = best_model.best_estimator_.predict(X_test)
    print(best_model.best_params_)

    # regr = MLPRegressor(random_state=0, max_iter=1000, batch_size=20, hidden_layer_sizes=(5, 10, 5), alpha=0.005)

    # regr.fit(X_train, y_train)

    # path_to_save = '../TrainedModels/fitted_mlp_model_superkiller_test_one_cell_' + metric + '.sav'
    # pickle.dump(regr, open(path_to_save, 'wb'))

    # y_pred = regr.predict(X_test)

    error = calc_distance_metric_between_signals(y_test, y_pred, metric)

    print("mlp " + metric + ": " + str(error))



def split_data(file_path: str,
               metric: str = None,
               num_of_cells_split: int = None,
               leave_out: (bool, str) = (False, '')):

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
        df = adjust_data_to_positive_values(df)

    to_X = df.drop([LABEL], axis=1)
    X = pd.DataFrame(to_X, columns=to_X.columns)
    y = pd.Series(df[LABEL])

    if num_of_cells_split is not None:
        X_train, X_test, y_train, y_test = my_train_test_split(X, y, num_of_cells_split)

    else:
        X = X.drop([CELL_IDX], axis=1)
        # split the data to train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    svr_model(X_test, X_train, y_test, y_train, metric)


def split_fac_bso_data(file_path: str, metric: str = RMSE):
    df = pd.read_csv(file_path)
    # filenames = set(df[FILE_NAME])
    filenames = df[FILE_NAME]

    if len(df[UNNAMED_COLUMN]) > 0:
        df = df.drop([UNNAMED_COLUMN], axis=1)

    to_X = df.drop([LABEL], axis=1)
    X = pd.DataFrame(to_X, columns=to_X.columns)
    y = pd.Series(df[LABEL])

    # for file in filenames:
    #     X_train = X[X[FILE_NAME] != file]
    #     X_test = X[X[FILE_NAME] == file]
    #     y_train = y[X[FILE_NAME] != file]
    #     y_test = y[X[FILE_NAME] == file]
    #
    #     X_train = X_train.drop([FILE_NAME], axis=1)
    #     X_test = X_test.drop([FILE_NAME], axis=1)
    #
    #     svr_model(X_test, X_train, y_test, y_train, metric, file)

    X_train = X[X[FILE_NAME] != filenames[0]]
    X_test = X[X[FILE_NAME] == filenames[0]]
    y_train = y[X[FILE_NAME] != filenames[0]]
    y_test = y[X[FILE_NAME] == filenames[0]]

    X_train = X_train.drop([FILE_NAME], axis=1)
    X_test = X_test.drop([FILE_NAME], axis=1)

    svr_model(X_test, X_train, y_test, y_train, metric, filenames[0])


def svr_model(X_test, X_train, y_test, y_train, metric, filename=None):
    regr = SVR(kernel='rbf', gamma='auto', C=4, tol=0.001, shrinking=False, epsilon=0.5)
    # regr = SVR(kernel='rbf', gamma='scale', C=50, tol=0.001, shrinking=False, epsilon=0.8)
    # parameters = {
    #     'kernel': ['linear', 'rbf', 'sigmoid', 'precomputed'],
    #     'gamma': ['scale', 'auto'],
    #     'C': [1, 1.5, 0.5, 2],
    #     'tol': [0.0001, 0.001, 0.005, 0.01, 0.1, 4, 10, 50, 100],
    #     'shrinking': [True, False],
    #     'cache_size': [100, 200, 300, 400],
    #     'epsilon': [0.001, 0.01, 0.1, 0.2, 0.5, 0.8],
    #  }
    #
    # best_model = grid_search(regr, parameters, cv=3)
    # best_model.fit(X_train, y_train)
    # y_pred = best_model.best_estimator_.predict(X_test)
    # print("best params for test " + filename + ":" + best_model.best_params_)

    regr.fit(X_train, y_train)
    # path_to_save = '../TrainedModels/FAC&BSO_trained_models/fitted_svr_model_fac&bso_' + filename + '_' + metric + '.sav'
    # pickle.dump(regr, open(path_to_save, 'wb'))
    y_pred = regr.predict(X_test)

    error = calc_distance_metric_between_signals(y_test, y_pred, metric)
    print("svr " + filename + " " + metric + ": " + str(error))


def grid_search(model, parameters: dict, cv: int = None):
    scoring = make_scorer(my_custom_loss_func, greater_is_better=False)
    best_model = GridSearchCV(estimator=model, param_grid=parameters, scoring=scoring, cv=cv)
    return best_model


def my_custom_loss_func(y_true: np.array, y_pred: np.array, metric: str = KL_DIVERGENCE):
    score = calc_distance_metric_between_signals(y_true, y_pred, RMSE)
    return score


def feature_importance(importance):
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()


def my_train_test_split(X, y, num_cells_to_test):
    if num_cells_to_test == 1:
        most_frequent_idx = X[CELL_IDX].mode()
        X_train = X[X[CELL_IDX] != most_frequent_idx[0]]
        X_test = X[X[CELL_IDX] == most_frequent_idx[0]]
        y_train = y[X[CELL_IDX] != most_frequent_idx[0]]
        y_test = y[X[CELL_IDX] == most_frequent_idx[0]]

    else:
        unique_idx = pd.Series(list(set(X[CELL_IDX])))
        cells_to_test = unique_idx.sample(n=num_cells_to_test, random_state=1)
        X_train = X[~X[CELL_IDX].isin(cells_to_test)]
        X_test = X[X[CELL_IDX].isin(cells_to_test)]
        y_train = y[~X[CELL_IDX].isin(cells_to_test)]
        y_test = y[X[CELL_IDX].isin(cells_to_test)]

    X_train = X_train.drop([CELL_IDX], axis=1)
    X_test = X_test.drop([CELL_IDX], axis=1)

    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    # elastic_net()
    # split_data('../PreparedDatasets/dataset_with_%_alive_cells_param_DMEM+7.5uM_erastin.csv',
    #            RMSE)

    split_fac_bso_data('../PreparedDatasets/dataset_with_%_alive_cells_param_FAC&BSO')
