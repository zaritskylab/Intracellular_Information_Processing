from Miscellaneous.pillars_utils import *
from statsmodels.tsa.stattools import kpss


PILLAR_PERCENTAGE_MUST_PASS = 85

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
            if result[1] > 0.05:
                non_stationary.append(p)
        if method == 'kpss':
            result = kpss(df[p])
            if result[1] < 0.05:
                non_stationary.append(p)
    print(non_stationary)
    return non_stationary


def var_model(df):
    model = VAR(df)
    aic_res = []
    lags = []
    for i in [1,2,3,4,5,6,7,8,9,10,11,12]:
        result = model.fit(i)
        try:
            aic_res.append((i, result.aic))
        except:
            continue
    maxlag = min(aic_res, key=lambda t: t[1])

    return maxlag[0]


def edges_distribution_plots(gc_df, pillar_intensity_dict):
    alive_pillars_correlation = alive_pillars_symmetric_correlation()
    neighbors = get_pillar_to_neighbors()
    no_edge = []
    one_sided_edge = []
    two_sided_edge = []
    for col in gc_df.keys():
        for row, _ in gc_df.iterrows():
            if eval(row) in neighbors[eval(col)]:
                # corr = pearsonr(pillar_intensity_dict[col], pillar_intensity_dict[row])[0]
                corr = alive_pillars_correlation[col][row]
                if gc_df[col][row] < 0.05 and gc_df[row][col] < 0.05:
                    two_sided_edge.append(corr)
                if (gc_df[col][row] < 0.05 and gc_df[row][col] > 0.05) or (gc_df[col][row] > 0.05 and gc_df[row][col] < 0.05):
                    one_sided_edge.append(corr)
                else:
                    no_edge.append(corr)
    sns.histplot(data=no_edge, kde=True)
    plt.xlabel("Correlations")
    plt.title('Correlation of no edges between neighbors')
    plt.show()
    sns.histplot(data=one_sided_edge, kde=True)
    plt.xlabel("Correlations")
    plt.title('Correlation of 1 sided edges between neighbors')
    plt.show()
    sns.histplot(data=two_sided_edge, kde=True)
    plt.xlabel("Correlations")
    plt.title('Correlation of 2 sided edges between neighbors')
    plt.show()



if __name__ == '__main__':
    p2i = get_alive_pillars_to_intensities()
    p2i_df = pd.DataFrame({str(k): v for k, v in p2i.items()})
    p2i_df_diff = p2i_df.diff().dropna()
    # derivative_df = pd.DataFrame({str(k): np.gradient(v) for k, v in p2i.items()})
    # derivative_df = derivative_df.diff().dropna()

    # Check if data is stationary
    non_stationary_adf = stationary_test(p2i_df_diff, method='adf')
    if 100 - ((len(non_stationary_adf)/len(p2i_df_diff))*100) > PILLAR_PERCENTAGE_MUST_PASS:
        non_stationary_kpss = stationary_test(p2i_df_diff, method='kpss')
        if 100 - ((len(non_stationary_kpss) / len(p2i_df_diff)) * 100) > PILLAR_PERCENTAGE_MUST_PASS:
            non_stationary_adf.extend(non_stationary_kpss)
            # stationary_pillars_intens = pd.DataFrame({str(k): np.gradient(v) for k, v in p2i.items() if str(k) not in non_stationary_derivative})
            # stationary_pillars_intens = pd.DataFrame({str(k): v for k, v in p2i.items() if str(k) not in non_stationary_adf})
            stationary_pillars_intens = p2i_df_diff.drop(non_stationary_adf, axis=1)
            maxlag = var_model(stationary_pillars_intens)
            gc_df = grangers_causation_matrix(stationary_pillars_intens, stationary_pillars_intens.columns, maxlag=maxlag)
            # build_directed_graph(gc_df, only_alive=True)
            edges_distribution_plots(gc_df, p2i_df_diff)

