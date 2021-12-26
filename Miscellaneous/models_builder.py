import pickle

from matplotlib import pyplot
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import shap
import tensorflow
from sklearn.svm import SVR

from AnalyzerModels.BaselineCell2Cell_InfluenceAnalyzer import BaselineCell2CellInfluenceAnalyzer
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


def leave_one_exp_out(file_path: str, metric: str = RMSE):
    df = pd.read_csv(file_path)
    filenames = set(df[FILE_NAME])

    if len(df[UNNAMED_COLUMN]) > 0:
        df = df.drop([UNNAMED_COLUMN], axis=1)

    to_X = df.drop([LABEL], axis=1)
    X = pd.DataFrame(to_X, columns=to_X.columns)
    y = pd.Series(df[LABEL])

    df_results = pd.DataFrame(columns=['File Name', 'Metric', 'Baseline Error', 'Model Error'])

    i = 0
    for file in filenames:
        i += 1
        X_train = X[X[FILE_NAME] != file]
        X_test = X[X[FILE_NAME] == file]
        y_train = y[X[FILE_NAME] != file]
        y_test = y[X[FILE_NAME] == file]

        X_train = X_train.drop([FILE_NAME], axis=1)
        X_test = X_test.drop([FILE_NAME], axis=1)

        # my model error
        model_error = svr_model(X_test, X_train, y_test, y_train, metric, file)

        # baseline error
        single_file_path = NON_COMPRESSED_FILE_MAIN_DIR + '\\' + file
        if os.path.isfile(single_file_path):
            baseline_influence_analyzer = BaselineCell2CellInfluenceAnalyzer(file_full_path=single_file_path,
                                                                             kwargs={
                                                                                 'distance_metric_from_true_times_of_death': 'rmse'})
            baseline_error = baseline_influence_analyzer.calc_prediction_error()
        else:
            baseline_error = ["error", "error"]

        print(i, file, " svr result - ", model_error, " baseline error ", baseline_error)

        df_results = df_results.append(
            pd.DataFrame({'File Name': file,
                          'Metric': metric,
                          'Baseline Error': str(baseline_error),
                          'Model Error': model_error}, index=[0]), ignore_index=True)

    df_results.to_csv('DMEM_F12-AA+400uM_FAC&BSO_rmse_grid_search_results.csv')

    # #TODO:
    # file_name = '20161129_MCF7_FB_xy13.csv'
    # X_train = X[X[FILE_NAME] != file_name]
    # X_test = X[X[FILE_NAME] == file_name]
    # y_train = y[X[FILE_NAME] != file_name]
    # y_test = y[X[FILE_NAME] == file_name]
    #
    # X_train = X_train.drop([FILE_NAME], axis=1)
    # X_test = X_test.drop([FILE_NAME], axis=1)
    #
    # error = svr_model(X_test, X_train, y_test, y_train, metric, file_name)
    # # baseline error
    # single_file_path = NON_COMPRESSED_FILE_MAIN_DIR + '\\' + file_name
    # baseline_influence_analyzer = BaselineCell2CellInfluenceAnalyzer(file_full_path=single_file_path,
    #                                                                          kwargs={
    #                                                                              'distance_metric_from_true_times_of_death': 'rmse'})
    # baseline_error = baseline_influence_analyzer.calc_prediction_error()
    #
    # print(file_name, " svr result - ", error, " baseline error ", baseline_error)



def svr_model(X_test, X_train, y_test, y_train, metric, filename=None):
    # regr = SVR(kernel='rbf', gamma='auto', C=4, tol=0.001, shrinking=False, epsilon=0.5)
    path_to_save = '../TrainedModels/fitted_svr_model_' + filename + '_' + metric + '_best' + '.sav'

    if (os.path.isfile(path_to_save)):
        regr = pickle.load(open(path_to_save, 'rb'))
    else:
        # regr = SVR(kernel='rbf', gamma='scale', C=50, tol=0.001, shrinking=False, epsilon=0.8)
        regr = SVR(kernel='rbf', gamma='auto', C=2, tol=0.0001, shrinking=True, epsilon=0.001)

        parameters = {
            'kernel': ['linear', 'rbf', 'sigmoid', 'precomputed'],
            'gamma': ['scale', 'auto'],
            'C': [1, 1.5, 0.5, 2, 5, 10, 50],
            'tol': [0.0001, 0.001, 0.005, 0.01, 0.1],
            'shrinking': [True, False],
            # 'cache_size': [100, 200, 300, 400],
            'epsilon': [0.001, 0.01, 0.0005, 0.005, 0.1],
         }

        best_model = grid_search(regr, parameters, cv=3)
        best_model.fit(X_train, y_train)
        pickle.dump(best_model, open(path_to_save, 'wb'))
        y_pred = best_model.best_estimator_.predict(X_test)
        print("best params for test " + filename + ":" + str(best_model.best_params_))

        # regr.fit(X_train, y_train)
        # pickle.dump(regr, open(path_to_save, 'wb'))

    # y_pred = regr.predict(X_test)

    error = calc_distance_metric_between_signals(y_test, y_pred, metric)
    # print("svr " + filename + " " + metric + ": " + str(error))
    return error


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

    leave_one_exp_out('../PreparedDatasets/dataset_with_%_alive_cells_param_DMEM_F12-AA+400uM_FAC&BSO.csv')
