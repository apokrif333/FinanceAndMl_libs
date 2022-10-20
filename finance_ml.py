from APIs.API_Tiingo import multithread
from APIs.API_Brokers.Ameritrade import TOS_API
from termcolor import colored
from pprint import pprint
from dateutil.relativedelta import relativedelta as rd
from plotly.subplots import make_subplots

import plotly.graph_objs as go
import plotly.io as pio
import time
import plotly
import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import os

# Settings
pd.options.display.max_rows = 1_500
pd.options.display.max_columns = 500
pd.set_option('display.width', 1400)
pd.options.display.float_format = '{:.6f}'.format


def are_files_exist_and_actual(files: list, files_path: str, local_timezone: datetime.timezone) -> list:
    """
    Проверим существукет ли csv файл и был ли он обновлён сегодня. Если нет, то вернуть его.

    Returns
    -------
    list: Лист неактуальных и  несуществующих файлов

    Parameters
    ----------
    :param files: лист файлов для проверки.
    :param files_path: путь для проверки файлов
    :param local_timezone: локальный часовой пояс, для проверки актуальности файлов

    """
    time_now = datetime.datetime.now(tz=local_timezone)
    files_for_download = []
    for file in files:
        file_path = os.path.join(files_path, f'{file}.csv')
        if os.path.isfile(file_path):
            create_time = datetime.datetime.fromtimestamp(os.stat(file_path).st_mtime, tz=local_timezone)
            if (time_now - create_time).total_seconds() >= 36_000:
                files_for_download.append(file)
        else:
            files_for_download.append(file)

    return files_for_download


def download_tickers(
        tickers: list,
        downloader: str = 'tiingo',
        reload: bool = True,
        threads: int = 5
) -> None:
    """
    Получить исторические цены по списку тикеров. Нужно добавить выкачку из ФРС и fmpcloud.io.
    Tiingo получает валюты без спец. символов, типа EURUSD.
    Yahoo обозначает валюты как EURUSD=X.
    TOS не работает для индикаторов, фьючерсов и валют.

    Returns
    -------
    None: Сохраняет скаченные инструменты в их директории

    Parameters
    ----------
    :param tickers: лист инструментов.
    :param downloader: 'tiingo', 'yahoo', 'tos'. Через какой функционал ищем, через что скачиваем.
    :param reload: перекачать ли указанные тикеры? Если False, то будут перекачены тикеры, дата изменения которых моложе
        сегодняшней даты или которые ранее не были скачены.
    :param threads: количество потоков для пареллельной скачки. Для tiingo или yahoo.

    """
    local_timezone = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    data_path = os.path.join(os.getcwd().split('Projects')[0], 'Цены\Дейли')
    tickers_for_download = tickers

    if downloader == 'tiingo':
        price_path = os.path.join(data_path, 'tiingo', 'usa')
        if reload is False:
            tickers_for_download = are_files_exist_and_actual(tickers_for_download, price_path, local_timezone)

        if len(tickers_for_download) == 0:
            return None
        if len(tickers_for_download) < threads:
            threads = len(tickers_for_download)

        multithread.download_tickers('usa', tickers_for_download, threads)

    elif downloader == 'yahoo':
        price_path = os.path.join(data_path, 'yahoo')
        if reload is False:
            tickers_for_download = are_files_exist_and_actual(tickers_for_download, price_path, local_timezone)

        if len(tickers_for_download) == 0:
            return None
        if len(tickers_for_download) < threads:
            threads = len(tickers_for_download)

        df_tickers = yf.download(tickers_for_download, group_by='ticker', actions=True, threads=threads)

        if len(tickers_for_download) > 1:
            for ticker in tickers_for_download:
                if len(df_tickers[ticker.upper()].dropna()) == 0:
                    continue
                file_path = os.path.join(price_path, f'{ticker}.csv')
                df_tickers[ticker].dropna().to_csv(file_path)
        else:
            if len(df_tickers.dropna()) != 0:
                file_path = os.path.join(price_path, f'{tickers_for_download[0]}.csv')
                df_tickers.dropna().to_csv(file_path)

    elif downloader == 'tos':
        price_path = os.path.join(data_path, 'tos')
        if reload is False:
            tickers_for_download = are_files_exist_and_actual(tickers_for_download, price_path, local_timezone)

        if len(tickers_for_download) == 0:
            return None

        tickers_dict = TOS_API().get_historical_data(tickers_for_download)
        for ticker in tickers_dict:
            file_path = os.path.join(price_path, f'{ticker}.csv')
            tickers_dict[ticker].to_csv(file_path)

    else:
        raise Exception(f"Неверный параметр downloader - {downloader}")

    # Все ли тикеры получены
    print(price_path)
    for ticker in tickers:
        file_path = os.path.join(price_path, f'{ticker}.csv')
        if os.path.isfile(file_path) is False:
            print(colored(f"{ticker} не скачен. Попробуй поменять downloader", 'red'))


def get_tickers(tickers: list) -> dict:
    """
    Получить указанные инструменты и привести их к одинаковому виду

    Returns
    -------
    dict: Словарь, где ключ - тикер, а значения стандартизированный pd.DataFrame

    Parameters
    ----------
    :param tickers: лист инструментов.

    """
    data_path = os.path.join(os.getcwd().split('Projects')[0], 'Цены\Дейли')
    tickers_paths = {}
    for dir, folders, files in os.walk(data_path):
        if (r'tiingo' in dir) or ('yahoo' in dir) or ('tos' in dir) or ('sintetic' in dir) or ('pandas' in dir):
            tickers_paths[dir] = files
    tiingo_paths = [path for path in tickers_paths.keys() if 'tiingo' in path]
    tiingo_paths.reverse()
    yahoo_path = [path for path in tickers_paths.keys() if 'yahoo' in path][0]
    tos_path = [path for path in tickers_paths.keys() if 'tos' in path][0]
    sintetic_path = [path for path in tickers_paths.keys() if 'sintetic' in path][0]
    pandas_paths = [path for path in tickers_paths.keys() if 'pandas' in path]

    dict_tickers = {}
    for ticker in tickers:
        t_csv = f'{ticker}.csv'

        in_tiingo = False
        for tiingo_path in tiingo_paths:
            if t_csv in tickers_paths[tiingo_path]:
                cur_path = os.path.join(tiingo_path, t_csv)
                print(cur_path)
                dict_tickers[ticker] = pd.read_csv(cur_path, parse_dates=['date'])
                dict_tickers[ticker]['date'] = pd.to_datetime(dict_tickers[ticker]['date'].dt.strftime('%Y-%m-%d'))
                dict_tickers[ticker].set_index('date', inplace=True)
                in_tiingo = True
                break

        in_pandas = False
        for pandas_path in pandas_paths:
            if t_csv in tickers_paths[pandas_path]:
                cur_path = os.path.join(pandas_path, t_csv)
                print(cur_path)
                dict_tickers[ticker] = pd.read_csv(cur_path, parse_dates=['date'])
                dict_tickers[ticker]['date'] = pd.to_datetime(dict_tickers[ticker]['date'].dt.strftime('%Y-%m-%d'))
                dict_tickers[ticker].set_index('date', inplace=True)
                in_pandas = True
                break

        if in_tiingo:
            continue

        elif in_pandas:
            continue

        elif t_csv in tickers_paths[yahoo_path]:
            cur_path = os.path.join(yahoo_path, t_csv)
            print(cur_path)
            dict_tickers[ticker] = pd.read_csv(cur_path, parse_dates=['Date'])
            if 'Stock Splits' in dict_tickers[ticker].columns:
                dict_tickers[ticker]['Stock Splits'] = dict_tickers[ticker]['Stock Splits'].replace(0, 1)
            dict_tickers[ticker].columns = pd.Series(dict_tickers[ticker].columns).str.lower()
            dict_tickers[ticker].rename(columns={
                'adj close': 'adjClose',
                'open': 'adjOpenNoDiv',
                'high': 'adjHighNoDiv',
                'low': 'adjLowNoDiv',
                'close': 'adjCloseNoDiv',
                'dividends': 'adjDivCash',
                'stock splits': 'splitFactor',
            }, inplace=True)
            dict_tickers[ticker].set_index('date', inplace=True)

        elif t_csv in tickers_paths[tos_path]:
            cur_path = os.path.join(tos_path, t_csv)
            print(cur_path)
            dict_tickers[ticker] = pd.read_csv(cur_path)
            dict_tickers[ticker].rename(columns={
                'open': 'adjOpenNoDiv',
                'high': 'adjHighNoDiv',
                'low': 'adjLowNoDiv',
                'close': 'adjCloseNoDiv',
                'Open': 'adjOpenNoDiv',
                'High': 'adjHighNoDiv',
                'Low': 'adjLowNoDiv',
                'Close': 'adjCloseNoDiv',
                'datetime': 'date',
                'Date': 'date',
            }, inplace=True)
            dict_tickers[ticker]['date'] = pd.to_datetime(dict_tickers[ticker]['date'])
            dict_tickers[ticker].set_index('date', inplace=True)

        elif t_csv in tickers_paths[sintetic_path]:
            cur_path = os.path.join(sintetic_path, t_csv)
            print(cur_path)
            dict_tickers[ticker] = pd.read_csv(cur_path)
            dict_tickers[ticker].rename(columns={
                'Open': 'adjOpen',
                'Low': 'adjHigh',
                'High': 'adjLow',
                'Close': 'adjClose',
                'Dividend': 'divCash',
                'Date': 'date',
            }, inplace=True)
            dict_tickers[ticker]['date'] = pd.to_datetime(dict_tickers[ticker]['date'])
            dict_tickers[ticker].set_index('date', inplace=True)

        else:
            print(colored(f"{ticker} нет в базе", 'red'))

    return dict_tickers


def wma(s: pd.Series, period: int) -> pd.Series:
    """
    Weighted moving average

    """
    return s.rolling(period).apply(
        lambda x: ((np.arange(period) + 1) * x).sum() / (np.arange(period) + 1).sum(),
        raw=True
    )


def hma(s: pd.Series, period: int) -> pd.Series:
    """
    Hull moving average

    """
    return wma(
        wma(s, period // 2).multiply(2).sub(wma(s, period)),
        int(np.sqrt(period))
    )


def make_moving_average(
        df: pd.DataFrame,
        col_name: str,
        ticker_name: str,
        ma_lens: list = [200],
        ma_type: str = 'sma'
) -> pd.DataFrame:
    """
    Расчёт мувингов.

    Returns
    -------
    pd.DataFrame: фрейм, дополненный МА расчётами

    Parameters
    ----------
    :param df: данные по инструменту, где индексом таблицы является дата
    :param col_name: имя клонки, которую требуется обработать
    :param ticker_name: название инструмента для имени новосозданной колонки
    :param ma_lens: длины мувингов, которые требуется рассчитать
    :param ma_type: 'sma', 'ema', 'wma', 'hma', 'smoothma'. Виды мувингов.

    """
    assert ma_type in ['sma', 'ema', 'wma', 'hma', 'smoothma'], f"Неизвестный тип ma - {ma_type}"

    for ma_len in ma_lens:
        if ma_type == 'sma':
            df[f'{ticker_name}_{col_name}_sma{ma_len}'] = df[col_name].rolling(ma_len).mean()
            df[f'{ticker_name}_{col_name}_sma{ma_len}'].fillna(df[col_name].expanding().mean(), inplace=True)
        elif ma_type == 'ema':
            df[f'{ticker_name}_{col_name}_ema{ma_len}'] = df[col_name].ewm(span=ma_len).mean()
        elif ma_type == 'wma':
            df[f'{ticker_name}_{col_name}_wma{ma_len}'] = wma(df[col_name], ma_len)
        elif ma_type == 'hma':
            df[f'{ticker_name}_{col_name}_hma{ma_len}'] = hma(df[col_name], ma_len)
        elif ma_type == 'smoothma':
            weights = np.repeat(1.0, ma_len) / ma_len
            smooth = np.convolve(df[col_name], weights, 'valid')
            mask = df.index[len(df.index) - len(smooth):]
            df.loc[mask, f'{ticker_name}_{col_name}_smoothma{ma_len}'] = smooth

    return df


def make_momentum(
        df: pd.DataFrame,
        col_name: str,
        ticker_name: str,
        mom_period: str = 'monthly',
        mom_lens: list = [12],
) -> pd.DataFrame:
    """
    Расчёт моментума.

    Returns
    -------
    pd.DataFrame: фрейм, дополненный расчётами моментума

    Parameters
    ----------
    :param df: данные по инструменту, где индексом таблицы является дата
    :param col_name: имя клонки, которую требуется обработать
    :param ticker_name: название инструмента для имени новосозданной колонки
    :param mom_period: 'bars', 'daily', 'weekly', 'monthly', 'end_week', 'end_month'. фрейм для расчёта моментума.
        Сравнить бары / дни / недели / месяцы / завершения недель / завершения месяцев.
    :param mom_lens: за сколько баров, дней, недель, месяцев замеряем моментум.

    """
    assert mom_period in ['bars', 'daily', 'weekly', 'monthly', 'end_week', 'end_month'], \
        f"Неизвестный период моментума - {mom_period}"

    df_index = pd.Series(df.index)
    for mom_len in mom_lens:
        new_col_name = f'{ticker_name}_{col_name}_mom{mom_len}_{mom_period}'

        if mom_period in ['end_month', 'end_week']:
            if mom_period == 'end_month':
                mask = (df_index.dt.month != df_index.shift(-1).dt.month).values
            else:
                mask = (df_index.dt.isocalendar().week != df_index.shift(-1).dt.isocalendar().week).values
            df_cut = df[mask].copy()
            df[new_col_name] = (df_cut[col_name] / df_cut[col_name].shift(mom_len)).fillna(
                df_cut[col_name] / df_cut[col_name].iloc[0]
            )

        elif mom_period in {'monthly', 'weekly', 'daily'}:
            new_values = []
            for row in df.itertuples():
                if mom_period == 'monthly':
                    past_date = row.Index - rd(months=mom_len)
                elif mom_period == 'weekly':
                    past_date = row.Index - rd(weeks=mom_len)
                else:
                    past_date = row.Index - rd(days=mom_len)
                past_value = df.loc[past_date:, col_name].iloc[0]
                new_values.append(getattr(row, col_name) / past_value)

            df[new_col_name] = new_values

        elif mom_period == 'bars':
            df[new_col_name] = (df[col_name] / df[col_name].shift(mom_len)).fillna(
                df[col_name] / df[col_name].iloc[0]
            )

    return df


def calc_tuomo_vola(s: pd.Series, vola_type: str = 'standart', period: int = 252) -> pd.Series:
    """
    Годовая волатильность, рассчитанная так же, как на сайте portfoliovisualizer.

    Parameters
    ----------
    :param s: значения выраженные в абсолюте
    :param vola_type: 'standart', 'downside'
    :param period: сколько элементнов в году. Если дневной период то 252, если месячный, то 12.

    """
    port_cng = np.diff(s) / s[:-1] * 100
    if vola_type == 'downside':
        port_cng = port_cng [port_cng < 0]

    return np.std(port_cng, ddof=1) * np.sqrt(period)


def calc_standart_vola(s: pd.Series, period: int = 252) -> pd.Series:
    """
    Классическая годовая волатильность. Рассчитывается на порядок быстрее, нежели calc_tuomo_vola.

    Parameters
    ----------
    :param s: отношение значений, рейтио
    :param period: сколько элементнов в году. Если дневной период то 252, если месячный, то 12.

    """

    return np.log(s).std() * np.sqrt(period)


def make_volatility(
        df: pd.DataFrame,
        col_name: str,
        ticker_name: str,
        vola_type: str = 'standart',
        vola_lens: list = [1],
        vola_period: str = 'end_month'
) -> pd.DataFrame:
    """
    Расчёт волатильности. Если vola_period не 'end_week', 'end_month', то заполняет Nan через expanding, после чего
        неизвестные остатки заполняет нулями.

    Returns
    -------
    pd.DataFrame: фрейм, дополненный расчётами волатильности

    Parameters
    ----------
    :param df: данные по инструменту, где индексом таблицы является дата
    :param col_name: имя клонки, которую требуется обработать
    :param ticker_name: название инструмента для имени новосозданной колонки
    :param vola_type: 'standart', 'downside'
    :param vola_lens: за сколько баров, дней, недель, месяцев замеряем волу.
    :param vola_period: 'bars', 'daily', 'weekly', 'monthly', 'end_week', 'end_month'

    """
    assert vola_type in ['standart', 'downside'], \
        f"Неизвестный тип волатильности - {vola_type}"
    assert vola_period in ['bars', 'daily', 'weekly', 'monthly', 'end_week', 'end_month'], \
        f"Неизвестный период волатильности - {vola_period}"

    df_index = pd.Series(df.index)
    for vola_len in vola_lens:
        new_col_name = f'{ticker_name}_{col_name}_vol{vola_len}_{vola_period}_{vola_type}'

        if vola_period in ['end_month', 'end_week']:
            df[new_col_name] = None
            if vola_period == 'end_month':
                mask = (df_index.dt.month != df_index.shift(-1).dt.month).values
            else:
                mask = (df_index.dt.isocalendar().week != df_index.shift(-1).dt.isocalendar().week).values
            end_dates = pd.Series(df[mask][col_name])

            for end_date in end_dates.index:
                start_value = end_dates[:end_date].iloc[-vola_len-1:-vola_len]
                if len(start_value) == 0:
                    start_date = end_dates.index[0]
                else:
                    start_date = start_value.index[0]

                cur_df = df[start_date:end_date]
                if len(cur_df) == 1:
                    continue

                if vola_type in ['standart', 'downside']:
                    df.loc[end_date, new_col_name] = calc_tuomo_vola(cur_df[col_name], vola_type)

        elif vola_period in {'monthly', 'weekly', 'daily'}:
            new_values = []
            df['returns'] = df[col_name] / df[col_name].shift(1)
            for row in df.itertuples():
                if vola_period == 'monthly':
                    past_date = row.Index - rd(months=vola_len)
                elif vola_period == 'weekly':
                    past_date = row.Index - rd(weeks=vola_len)
                else:
                    past_date = row.Index - rd(days=vola_len)
                cur_df = df[past_date:row.Index]

                if vola_type in ['standart', 'downside']:
                    if vola_type == 'downside':
                        df['returns'] = np.where(df['returns'] < 1, df['returns'], 1)
                    new_values.append(calc_standart_vola(cur_df['returns']))

            df.drop('returns', axis=1, inplace=True)
            df[new_col_name] = new_values
            df[new_col_name] = df[new_col_name].fillna(0)

        elif vola_period == 'bars':
            df['returns'] = df[col_name] / df[col_name].shift(1)
            period = df.groupby(pd.Grouper(freq='M')).count()[col_name].median() * 12

            if vola_type in ['standart', 'downside']:
                if vola_type == 'downside':
                    df['returns'] = np.where(df['returns'] < 1, df['returns'], 1)
                df[new_col_name] = np.log(df['returns']).rolling(vola_len).std() * np.sqrt(period)
                df[new_col_name] = df[new_col_name].fillna(np.log(df['returns']).expanding().std() * np.sqrt(period))
                df[new_col_name] = df[new_col_name].fillna(0)

            df.drop('returns', axis=1, inplace=True)

    return df


def check_dict_weight_is1(ports: dict) -> dict:
    """
    Если сумма весов тикеров одного из портфелей не равна 1, то задать всем тикерам в данном портфеле равный вес

    Parameters
    -------
    :param ports: словарь вида {'port_name_1': {some_ticker_1: weight_1, some_ticker_n: weight_n}, ... port_name_n: {}}

    """
    for risk_key, risk in ports.items():
        if np.round(np.sum(list(risk.values())), 2) != 1.0:
            new_weight = 1 / len(risk.keys())
            ports[risk_key] = dict.fromkeys(ports[risk_key], new_weight)

    return ports


def calc_cagr(port: pd.Series) -> pd.Series:
    """
    Рассчитаем CAGR

    Parameters
    -------
    :param port: динамика капитала за исследумый период в абсолюте. В качестве индекса timeseries.

    """
    if len(port) != 1:
        years_cnt = (port.index[-1] - port.index[0]).days / 365
    else:
        years_cnt = 0.05
    gain = port.iloc[-1] / port.iloc[0]

    return gain ** (1 / years_cnt) - 1


def calc_sortino(port: pd.Series) -> pd.Series:
    """
    Рассчитаем Sortino

    Parameters
    -------
    :param port: динамика капитала за исследумый период в абсолюте. В качестве индекса timeseries.

    """
    period = port.groupby(pd.Grouper(freq='M')).count().median() * 12
    cur_pct_cng = port.pct_change()
    downside_vol = cur_pct_cng[cur_pct_cng < 0].std() * np.sqrt(period)

    # return cur_pct_cng.mean() * period / downside_vol
    return calc_cagr(port) / downside_vol


def make_chart_with_dd(
        df: pd.DataFrame,
        chart_name: str,
        use_log: bool = True,
        port_col: str = 'port_close',
        port_dd_col: str = 'port_dd',
        bh_col: str = 'bh_close',
        bh_name: str = 'Buy&Hold',
        bench_col: str = 'bench_close',
        bench_name: str = 'Benchmark'
) -> plotly.graph_objs:
    """
    Выведем график с динамикой портфеля, бенча, бай_энд_холда и просадки порта. Таймсерия должна быть в качестве индекса
        в DataFrame.
    
    Parameters
    -------
    :param df: DataFrame с динамикой порта, байхолда, бенча и просадки порта
    :param chart_name: заголовок графика
    :param use_log: логарифмировать ли ось Y
    :param port_col: имя колонки капитала порта
    :param port_dd_col: имя колонки просадки порта
    :param bh_col: имя колонки капитала BH
    :param bh_name: имя для BH на графике
    :param bench_col: имя колонки капитала бенчмарка
    :param bench_name: имя для бенчмарка на графике

    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df.index, y=df[port_col], name=port_col),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df[port_dd_col], name=port_dd_col, mode='markers', marker={'size': 3}),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df[bh_col], name=bh_name),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df[bench_col], name=bench_name),
        secondary_y=False
    )
    fig.update_layout(title=chart_name)
    if use_log:
        fig.update_layout(yaxis1=dict(type='log'))

    return fig


def make_chart_multiple_y(df, columns_add_y: list, chart_title: str) -> go.Figure:
    """
    Выведем график, где все колонки будут отрисованы. Колонки, которые требует дополнительно ось y, требуется указать
        отдельно.

    Parameters
    -------
    :param df: DataFrame. Все его колонки будут отрисованы. Индекс - дата.
    :param columns_add_y: для каждой колонки из листа будет добавлена новая ось Y
    :param chart_title: название графика

    """
    normal_cols = list(set(list(df.columns)) - set(columns_add_y))

    layout = {'title': chart_title}
    layout['yaxis1'] = {'title': 'Price', 'titlefont': {'color': 'blue'}, 'tickfont': {'color': 'blue'}, 'type': 'log',
                        'spikedash': 'dash'}
    traces = [{}]
    for col in normal_cols:
        traces.append({'y': df[col], 'name': col, 'x': df.index})
    cnt = 2
    for col in columns_add_y:
        traces.append({'y': df[col], 'name': col, 'yaxis': f'y{cnt}', 'x': df.index, 'line': {'dash': 'dot'}})
        layout[f'yaxis{cnt}'] = {
            'title': col, 'side': 'right', 'overlaying': 'y', 'position': 1 - (cnt - 2) * 0.04, 'spikedash': 'dash',
            # 'titlefont': {'color': 'green'}, 'tickfont': {'color': 'blue'},
        }
        cnt += 1
    layout['xaxis'] = {'domain': [0.0, 1 - (cnt - 3) * 0.04], 'spikedash': 'dash'}

    fig = go.Figure(data=traces, layout=layout)
    # pio.show({'data': traces, 'layout': layout})

    return fig


def calc_slope(series: pd.Series, window: int = 21) -> pd.Series:
    """
    Векторизированный расчёт угола наклона у поданного вектора. Изменение угла наклона = изменение скорости вектора

    @param series:
    @param window: за какой период учитывать изменение наклона.
    @return:
    """
    from numpy.lib.stride_tricks import as_strided

    new_index = series.index

    ys = series.to_numpy()
    stride = ys.strides
    slopes, intercepts = np.polyfit(
        np.arange(window),
        as_strided(ys, (len(series) - window + 1, window), stride + stride, writeable=False).T,
        deg=1
    )

    return pd.Series(slopes, index=new_index[window - 1:]).shift(1)


def make_same_index(dict_data: dict, concat_col: str) -> dict:
    """
    Приводит все датафреймы в словаре к одному индексу. Изначально у инструментов в словаре таймсерия должна быть в
        качестве индекса.

    Parameters
    -------
    :param dict_data: словарь, где ключ - имя инструмента, а значения - его данные.
    :param concat_col: имя колонки, которая есть во всех инструментах.

    """
    for ticker in dict_data.keys():
        dict_data[ticker] = dict_data[ticker][~dict_data[ticker].index.duplicated(keep='first')]
    col_for_concat = [dict_data[ticker][concat_col] for ticker, data in dict_data.items()]
    true_index = pd.concat(col_for_concat, axis=1).dropna().index
    for ticker in dict_data.keys():
        dict_data[ticker] = dict_data[ticker].loc[true_index]

    return dict_data


def make_new_period(df: pd.DataFrame, period: str = 'week') -> pd.DataFrame:
    """
    Оставляет только конец периода в данных, удаляя иное. Таймсерия должна быть в качестве индекса.

    Parameters
    -------
    :param df: pd.DataFrame
    :param period: какой период требуется создать. 'week', 'month'

    """
    assert period in ['day', 'week', 'month'], f"Неверно указан period - {period}. Требуется day week, month"

    df_new = df.reset_index()
    date_name = list(df_new.columns)[0]
    if period == 'week':
        mask = df_new[date_name].dt.dayofweek > df_new[date_name].shift(-1).dt.dayofweek
        mask.iloc[-1] = True
        df = df[mask.values]
    elif period == 'month':
        mask = df_new[date_name].dt.month != df_new[date_name].shift(-1).dt.month
        df = df[mask.values]

    return df


def make_data_column(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Если формат колонки позволяет, то приводит колонку к виду %Y-%m-%d

    :param df:
    :param columns: имена колонок для изменения
    :return:
    """

    for col in columns:
        df[col] = pd.to_datetime(pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d'))

    return df


def get_today_prices(dict_data: dict, concat_col: str='adjClose', take_col: str='close') -> dict:
    """
    Скачивает текущие биржевые цены. Внимание, используется Ameritrade, а значит что фьючерсы и курсы валют не доступны.

    Parameters
    -------
    :param dict_data: словарь, где ключи - это тикеры, а значения pd.DataFrame с ценами
    :param concat_col: к какой колонке добавить новые данные
    :param take_col: какую колонку извлекаем из данных TOSа

    """
    tos = TOS_API()
    tickers_list = [ticker for ticker in dict_data.keys()]
    dict_today = tos.get_historical_data(tickers=tickers_list, startDate=datetime.datetime.now() - rd(days=20))

    for ticker in dict_today.keys():
        today_df = dict_today[ticker].rename(columns={take_col: concat_col})[[concat_col]]
        today_df.index = pd.to_datetime(today_df.index)
        need_index = set(today_df.index) - set(dict_data[ticker].index)
        dict_data[ticker] = pd.concat([dict_data[ticker], today_df.loc[need_index, :]])
        dict_data[ticker].index.name = 'date'

    return dict_data


def portfolio_enters_by_cols_sorter(
        df: pd.DataFrame,
        rank_col_name: str,
        gain_col_name: str,
        tickers: list,
        cnt_positions: int = 3,
        sort_largest: bool = True
) -> (pd.DataFrame, list):
    """
    Получает pd.DataFrame, в котором должны содержаться колонки с %cng активов, колонки ранжируя которые функция
        определит позиции на каждый искомый период. В качестве индекса pd.Timestamp

    @param df:
    @param rank_col_name: часть названия колонок с данными для ранжирования, типа '{rank_col_name}_{ticker}'
    @param gain_col_name: часть названия колонок с %cng активов, типа 'f'{gain_col_name}_{ticker}'
    @param tickers: список тикеров для анализа
    @param cnt_positions: количество позиций, которые будут выбраны после ранкинга
    @param sort_largest: ранжировать от большего к меньшему, или наоборот
    @return: обогащённый pd.DataFrame. Название колонок со входами.

    """
    rank_cols = [col for col in df.columns if rank_col_name in col]
    if sort_largest:
        sort_columns = df[rank_cols].apply(lambda s: ",".join(s.nlargest(cnt_positions).index.tolist()), axis=1)
    else:
        sort_columns = df[rank_cols].apply(lambda s: ",".join(s.nsmallest(cnt_positions).index.tolist()), axis=1)

    enter_cols = []
    for ticker in tickers:
        idx_mask = sort_columns[sort_columns.str.contains(f'{rank_col_name}_{ticker}')].index
        df.loc[idx_mask, f'enter_{ticker}'] = df[f'{gain_col_name}_{ticker}']
        enter_cols.append(f'enter_{ticker}')

    return df, enter_cols


def make_pivot_table(df: pd.DataFrame, x: str, y: str, z: str) -> pd.DataFrame:
    """
    Pivot-таблица, наглядный пример.

    @param df: pd.DataFrame
    @param x: колонка для создания индекса
    @param y: колонка для создания имён новых колонок
    @param z: колонка, которая будет служить значениями в новой таблице
    @return:

    """
    return pd.pivot_table(df[[x, y, z]], values=z, index=x, columns=y)


def make_3D_surface(df: pd.DataFrame, values_name: str, title: str) -> go.Figure:
    """
    Отрисовка 3D плоскости, где df имеет вид pivot_table

    @param df: pd.DataFrame, в виде pivot_table
    @param values_name: имя для значений таблицы
    @param title: название графика
    @return: go.Figure
    """
    fig = go.Figure(go.Surface(x=df.columns, y=df.index, z=df.values))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=df.columns.name, yaxis_title=df.index.name, zaxis_title=values_name),
        # width=1_200,
        # height=800,
        # margin=dict(l=65, r=50, b=65, t=90),
    )

    return fig


if __name__ == '__main__':
    # download_tickers(['USDCHF'], downloader='tiingo', reload=False)
    dict_tickers = get_tickers(['HOG', 'VUSTX', 'SPY', '^VIX', '$DJT', 'GOLD_H', 'DJSUKUK'])
    # make_moving_average(dict_tickers['HOG'], 'adjClose', 'HOG', ma_type='smoothma')
    # make_momentum(dict_tickers['HOG'], 'adjClose', 'HOG', mom_period='daily')
    dict_tickers['HOG'] = make_volatility(
        dict_tickers['HOG'], 'adjClose', 'HOG',
        vola_lens=[12], vola_period='end_month', vola_type='downside'
    )

    print(dict_tickers['HOG'][:500])
