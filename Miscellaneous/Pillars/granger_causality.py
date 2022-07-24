from statsmodels.tsa.stattools import kpss
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from Pillars.analyzer import *
from Pillars.consts import *
import pandas as pd


def grangers_causation_matrix(data, variables, maxlag=4, test='ssr_chi2test'):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var for var in variables]
    df.index = [var for var in variables]
    return df


def stationary_test(df, method='adf'):
    non_stationary = []
    for i, p in enumerate(df.columns):
        if method == 'adf':
            result = adfuller(df[p])
            if result[1] > Consts.gc_pvalue_threshold:
                non_stationary.append(p)
        if method == 'kpss':
            result = kpss(df[p])
            if result[1] < Consts.gc_pvalue_threshold:
                non_stationary.append(p)
    print(non_stationary)
    return non_stationary


def var_model(df):
    model = VAR(df)
    aic_res = []
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        result = model.fit(i)
        try:
            aic_res.append((i, result.aic))
        except:
            continue
    if len(aic_res) == 0:
        maxlag = [4]
    else:
        maxlag = min(aic_res, key=lambda t: t[1])

    return maxlag[0]


# def get_gc_df():
#     gc_df = get_gc_df_private()
#     # TODO: probably not needed, debug and check if can delete. see if pillars_to_drop is empty when fully = True
#     if Consts.fully_alive:
#         fully_alive_pillars = get_alive_pillars(get_mask_for_each_pillar())
#         fully_alive_pillars_str = []
#         for p in fully_alive_pillars:
#             fully_alive_pillars_str.append(str(p))
#
#         all_pillars = gc_df.columns
#         pillars_to_drop = []
#         for p in all_pillars:
#             if p not in fully_alive_pillars_str:
#                 pillars_to_drop.append(p)
#
#
#         gc_df = gc_df.drop(pillars_to_drop, axis=0)
#         gc_df = gc_df.drop(pillars_to_drop, axis=1)
#         return gc_df
#
#     return gc_df


def get_gc_df():
    if Consts.USE_CACHE and os.path.isfile(Consts.gc_df_cache_path):
        with open(Consts.gc_df_cache_path, 'rb') as handle:
            gc_df = pickle.load(handle)
            return gc_df

    p2i = get_alive_pillars_to_intensities()
    p2i_df = pd.DataFrame({str(k): v for k, v in p2i.items()})
    p2i_df_diff = p2i_df.diff().dropna()
    df_size = len(p2i_df_diff.columns)

    # Check if data is stationary
    non_stationary_adf = stationary_test(p2i_df_diff, method='adf')
    stationary_adf_percentage = 100 - ((len(non_stationary_adf) / df_size) * 100)
    print("adf test - non stationary pillars: " + str(non_stationary_adf))
    print("adf test - stationary percentage: " + str(stationary_adf_percentage) + "%")
    if stationary_adf_percentage > Consts.PILLAR_PERCENTAGE_MUST_PASS:
        non_stationary_kpss = stationary_test(p2i_df_diff, method='kpss')
        stationary_kpss_percentage = 100 - ((len(non_stationary_kpss) / df_size) * 100)
        print("kpss test - non stationary pillars: " + str(non_stationary_kpss))
        print("kpss test - stationary percentage: " + str(stationary_kpss_percentage) + "%")
        if stationary_kpss_percentage > Consts.PILLAR_PERCENTAGE_MUST_PASS:
            non_stationary_adf.extend(non_stationary_kpss)
            stationary_pillars_intens = p2i_df_diff.drop(non_stationary_adf, axis=1)
            maxlag = var_model(stationary_pillars_intens)
            gc_df = grangers_causation_matrix(stationary_pillars_intens, stationary_pillars_intens.columns,
                                              maxlag=maxlag)
            total_passed = 100 - (len(set(non_stationary_adf)) / df_size * 100)
            print("total passed stationary test: " + str(total_passed) + "%")

            if Consts.USE_CACHE:
                with open(Consts.gc_df_cache_path, 'wb') as handle:
                    pickle.dump(gc_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return gc_df
        else:
            raise RuntimeError("Not enough pillars passed stationary kpss test. Pillars passed: " + str(
                stationary_kpss_percentage) + "%")
    else:
        raise RuntimeError(
            "Not enough pillars passed stationary adf test. Pillars passed: " + str(stationary_adf_percentage) + "%")


def get_non_stationary_pillars_lst():
    p2i = get_alive_pillars_to_intensities()
    p2i_df = pd.DataFrame({str(k): v for k, v in p2i.items()})
    p2i_df_diff = p2i_df.diff().dropna()

    non_stationary_adf = stationary_test(p2i_df_diff, method='adf')
    non_stationary_kpss = stationary_test(p2i_df_diff, method='kpss')

    non_stationary_pillars = non_stationary_adf.copy()
    non_stationary_pillars.extend(non_stationary_kpss)

    total_passed = 1 - len(set(non_stationary_pillars)) / len(p2i_df_diff.columns)
    print("total passed stationary test: " + str(total_passed * 100) + "%")

    return non_stationary_pillars, total_passed



# if __name__ == '__main__':
#     p2i = get_alive_pillars_to_intensities()
#     p2i_df = pd.DataFrame({str(k): v for k, v in p2i.items()})
#     p2i_df_diff = p2i_df.diff().dropna()
#     gc_df = get_gc_df()
#     # build_directed_graph(gc_df, only_alive=True)
#     edges_distribution_plots(gc_df)
