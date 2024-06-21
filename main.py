from riskpremia import *

if __name__ == '__main__':
    from pandas_datareader import data as web

    ###################
    ########### ff5 factor #############
    #################
    port = pd.read_csv('D:/workspace/project/Risk premia Estimation/data/port.csv', index_col=[0],
                       parse_dates=[0]).dropna()
    port /= 100

    ff_factor = 'F-F_Research_Data_5_Factors_2x3'
    ff_factor_data = web.DataReader(ff_factor, 'famafrench', start=port.index[0])[0]
    ff_factor_data.index = ff_factor_data.index.to_timestamp()
    ff_factor_data /= 100

    start, end = max(port.index[0], ff_factor_data.index[0]), min(port.index[-1], ff_factor_data.index[-1])
    ff_factor_data = ff_factor_data.loc[start:end]
    port = port.loc[start:end]
    port = port - ff_factor_data['RF'].values.reshape(-1, 1)

    Est = Estimator(port, ff_factor_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
    res_two = Est.two_pass(True, lag=1).T
    res_three = Est.three_pass(max_k=300, lag=1).T

    print(res_two)
    print(res_three)

    ################
    ##### Macro Factor ####
    #############


    """
    The macro factors were constructed based on the methodology(=Mimicking Portfolios)  of 
    Macro Factor-Mimicking Portfolios (2022).
    """

    

    macro = pd.read_csv('D:/workspace/project/Risk premia Estimation/data/macro.csv',
                        index_col=[0], parse_dates=[0])[
        ['growth', 'inflation', 'real_rates', 'credit', 'currency', 'liquidity', 'tail_risk']]
    port = pd.read_csv('D:/workspace/project/Risk premia Estimation/data/port.csv', index_col=[0],
                       parse_dates=[0]).dropna()
    port /= 100
    start, end = max(port.index[0], macro.index[0]), min(port.index[-1], macro.index[-1])
    macro = macro.loc[start:end]
    port = port.loc[start:end]

    ff_factor = 'F-F_Research_Data_5_Factors_2x3'
    ff_factor_data = web.DataReader(ff_factor, 'famafrench', start=port.index[0], end=port.index[-1])[0]
    ff_factor_data /= 100

    port = port - ff_factor_data['RF'].values.reshape(-1, 1)
    Est = Estimator(port, macro)
    res_two = Est.two_pass(True, lag=1).T
    res_three = Est.three_pass(max_k=300, lag=1).T
    print(res_two)
    print(res_three)
