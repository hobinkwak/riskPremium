from riskpremia import *

if __name__ == '__main__':
    from pandas_datareader import data as web

    ######### FRED-MD PCA Factor ######
    ### 할거면 FRED MD 서, 아래 시간 축 맞추고 돌린 후 진행 ####
    macro = pd.read_csv('D:/workspace/project/Macro Factor Mimicking Portfolio/fmp/data/factors.csv',
                        index_col=['date'], parse_dates=['date'])
    port = pd.read_csv('D:/workspace/project/Risk premia Estimation/data/port.csv', index_col=[0],
                       parse_dates=[0]).dropna()
    port /= 100
    start, end = max(port.index[0], macro.index[0]), min(port.index[-1], macro.index[-1])
    macro = macro.loc[start:end]
    port = port.loc[start:end]

    ff_factor = 'F-F_Research_Data_5_Factors_2x3'
    ff_factor_data = web.DataReader(ff_factor, 'famafrench', start=start, end=end)[0]
    ff_factor_data /= 100

    port = port - ff_factor_data['RF'].values.reshape(-1, 1)
    Est = Estimator(port, macro)

    res_two = Est.two_pass(True).T
    res_three = Est.three_pass(max_k=300).T
    print(res_two)
    print(res_three)

    ###################
    ########### ff5 factor #############
    #################

    ff_factor = 'F-F_Research_Data_5_Factors_2x3'
    ff_factor_data = web.DataReader(ff_factor, 'famafrench', start=start, end=end)[0]
    ff_factor_data.index = ff_factor_data.index.to_timestamp()
    ff_factor_data /= 100
    port = pd.read_csv('D:/workspace/project/Risk premia Estimation/data/port.csv', index_col=[0],
                       parse_dates=[0]).dropna()
    port /= 100

    start, end = max(port.index[0], ff_factor_data.index[0]), min(port.index[-1], ff_factor_data.index[-1])
    ff_factor_data = ff_factor_data.loc[start:end]
    port = port.loc[start:end]
    port = port - ff_factor_data['RF'].values.reshape(-1, 1)

    Est = Estimator(port, ff_factor_data)
    res_two = Est.two_pass(True).T
    res_three = Est.three_pass(max_k=300).T

    print(res_two)
    print(res_three)

    ###################
    ##### FRED-MD group별 First PCA #####
    ###########################
    group_info = {1: 'Output and income',
                  2: 'Labor market',
                  3: 'Housing',
                  4: 'Consumption, orders, and inventories',
                  5: 'Money and credit',
                  6: 'Interest and exchange rates',
                  7: 'Prices',
                  8: 'Stock market'}
    group_factors = {}
    for g in group_info.values():
        group_factors[g] = pd.read_csv(f'D:/workspace/project/MacroPCA/result/factors {g}.csv', index_col=['date'], parse_dates=['date'])['0']
    macro = pd.DataFrame(group_factors)
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
    res_two = Est.two_pass(True).T
    res_three = Est.three_pass(max_k=300).T
    print(res_two)
    print(res_three)

    ################
    ##### Macro Factor ####
    #############

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
    res_two = Est.two_pass(True).T
    res_three = Est.three_pass(max_k=300).T
    print(res_two)
    print(res_three)

