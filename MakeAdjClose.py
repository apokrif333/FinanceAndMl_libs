import yfinance as yf
import numpy as np
import time
from json.decoder import JSONDecodeError
import pandas as pd
import os

# Settings
pd.options.display.max_rows = 1_500
pd.options.display.max_columns = 500
pd.set_option('display.width', 1400)
pd.options.display.float_format = '{:.6f}'.format


def DJI_1885():
    dji = pd.read_csv(r"E:\Биржа\Stocks. BigData\Цены\Дейли\sintetic\DJI_1885.csv", parse_dates=['date'])
    dji['Y_M'] = dji['date'].dt.to_period('M')

    dividend = pd.read_csv(r"C:\Users\Alex\Desktop\ForDel\MULTPL-SP500_DIV_YIELD_MONTH.csv", parse_dates=['Date'])
    dividend['Y_M'] = dividend['Date'].dt.to_period('M')

    final_div = pd.merge(dji, dividend, on='Y_M').drop_duplicates(['Y_M', 'Value']).drop_duplicates(['Y_M'])
    final_div = final_div[['date', 'Value']].rename(columns={'Value': 'div_percent'})

    dji = pd.merge(dji, final_div, on='date', how='left')
    dji['div_percent'] = dji['div_percent'].fillna(0)
    dji['div_abs'] = dji['div_percent'] / 100 * dji['Close'] / 12
    dji['adjClose'] = dji['Close'].copy(deep=True)
    dji.sort_values('date', ascending=False, inplace=True)
    dji.set_index('date', inplace=True)

    for row in dji.itertuples():
        cur_div = getattr(row, 'div_abs')
        if cur_div == 0:
            continue

        cur_df = dji[row.Index:].iloc[1:]
        if len(cur_df) == 0:
            continue

        prev_close = cur_df['Close'].iloc[0]
        div_mult = (prev_close - cur_div) / prev_close
        dji.loc[cur_df['adjClose'].index, 'adjClose'] = cur_df['adjClose'] * div_mult

    dji.sort_values('date', ascending=True, inplace=True)
    dji.to_csv(r"E:\Биржа\Stocks. BigData\Цены\Дейли\sintetic\DJI_1885_adj.csv")


def static_abs_div(path: str, ticker: str, fix_div: int = 40, close_col: str = 'Price') -> None:
    """ На основании указанного размера fix_div в долларах, рассчитывает ежемесячные дивы и счиатет adjClose


    :param path:
    :param ticker:
    :param fix_div: размер ежегодного фиксированного дивиденда
    :param close_col: колонка с нескоректированной ценой актива
    :return: None
    """
    df = pd.read_csv(fr"{path}\{ticker}.csv", parse_dates=['date'])
    df['Y_M'] = df['date'].dt.to_period('M')

    df_div = df.drop_duplicates(['Y_M']).drop_duplicates(['Y_M'])
    df_div['absDiv'] = fix_div / 12
    df = pd.merge(df, df_div[['date', 'absDiv']], on='date', how='left')
    df['absDiv'].fillna(0, inplace=True)

    df['adjClose'] = df[close_col].copy(deep=True)
    df.sort_values('date', ascending=False, inplace=True)
    df.set_index('date', inplace=True)
    for row in df.itertuples():
        cur_div = getattr(row, 'absDiv')
        if cur_div == 0:
            continue

        cur_df = df[row.Index:].iloc[1:]
        if len(cur_df) == 0:
            continue

        prev_close = cur_df[close_col].iloc[0]
        div_mult = (prev_close - cur_div) / prev_close
        df.loc[cur_df['adjClose'].index, 'adjClose'] = cur_df['adjClose'] * div_mult

    df.sort_values('date', ascending=True, inplace=True)
    df = df[['Price', 'Yield', 'absDiv', 'adjClose']]
    cur_disc = os.getcwd().split('\\')[0]
    df.to_csv(fr"{cur_disc}\Биржа\Stocks. BigData\Цены\Дейли\sintetic\{ticker}_adj.csv")


if __name__ == "__main__":
    # DJI_1885()
    static_abs_div(
        "E:\Биржа\Stocks. BigData\Projects\TreasuryPrice",
        "DGS3_Close.csv"
    )

