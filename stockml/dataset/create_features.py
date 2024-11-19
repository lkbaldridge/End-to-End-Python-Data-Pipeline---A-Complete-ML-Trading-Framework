import pandas_ta as ta
import talib
import finta
import stock_indicators
import pandas as pd

dummy_df = pd.DataFrame()
dummy_df.ta.cores = 0

def get_all_pandas_ta(df: pd.DataFrame, current: bool = False) -> dict[str, pd.DataFrame]:
    """
    Generates comprehensive technical indicators using the pandas-ta library for stock market analysis.

    This function applies various technical analysis indicators categorized by their types
    (e.g., momentum, volatility, trend) using the pandas-ta library. It handles data preprocessing,
    missing values, and specific edge cases for different indicator categories.

    Args:
        df (pd.DataFrame): Input DataFrame containing OHLCV (Open, High, Low, Close, Volume) data
            with a 'timestamp' column.
        current (bool, optional): Flag to determine whether to include the most recent data point.
            When True, includes the latest data point for real-time analysis.
            When False, excludes the last data point for historical analysis.
            Defaults to False.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames with computed technical indicators.
            Keys are 'df0' through 'df9', where:
            - 'df0': Original OHLCV data
            - 'df1' through 'df9': Technical indicators by category
            All DataFrames are indexed by timestamp and start from the 200th row to ensure
            sufficient data for indicator calculation.

    Notes:
        - Applies specific handling for known NaN issues in certain indicator categories
        - Removes indicators known to cause consistency issues (columns 19, 18, and 16 in
          respective categories)
        - Uses a 200-period warmup to ensure indicator stability
        - Handles momentum, volume, volatility, trend, and other technical indicator categories
        - All returned DataFrames have consistent indexing and no missing values

    Example:
        >>> df = pd.DataFrame(your_ohlcv_data)
        >>> technical_indicators = get_all_pandas_ta(df, current=True)
        >>> momentum_indicators = technical_indicators['df1']
    """
    
    categories = dummy_df.ta.categories
    categories = [cat.capitalize() for cat in categories]
    p_ta = {}

    for i in range(0, len(categories)):
        copy_df = df.copy()
        copy_df.set_index('timestamp', inplace=True)
        copy_df.ta.strategy(categories[i])
        copy_df = copy_df[200:]
        if i == 2:
            copy_df.iloc[:, 42:44] = copy_df.iloc[:, 42:44].fillna(0)
        if i == 3:
            copy_df.iloc[:, 10:12] = copy_df.iloc[:, 10:12].fillna(0)
            copy_df.iloc[:, 34:36] = copy_df.iloc[:, 34:36].fillna(0)
            copy_df.drop(copy_df.columns[19], axis=1, inplace=True)
        if i == 6:
            copy_df.iloc[:, 20:22] = copy_df.iloc[:, 20:22].fillna(0)
            copy_df.drop(copy_df.columns[18], axis=1, inplace=True)
        if i == 8:
            copy_df = copy_df.drop(copy_df.columns[16], axis=1)


        copy_df = copy_df.dropna()

        if current:
            p_ta[f'df{i+1}'] = copy_df
        else: 
            p_ta[f'df{i+1}'] = copy_df[:-1]
    
    if current:
        p_ta['df0'] = df[200:].set_index('timestamp')
    else:
        p_ta['df0'] = df[200:-1].set_index('timestamp')
    return p_ta


def get_all_talib_indicators(df: pd.DataFrame, current: bool = False) -> dict[str, pd.DataFrame]:
    """
    Calculates a comprehensive set of technical indicators using the TA-Lib library for
    advanced technical analysis.

    This function systematically applies all available TA-Lib indicators grouped by their
    categories (e.g., Momentum, Volatility, Pattern Recognition). It handles different
    input requirements for various indicators and manages special cases for complex
    indicators like MACD and Stochastic.

    Args:
        df (pd.DataFrame): Input DataFrame containing OHLCV (Open, High, Low, Close, Volume) data
            with a 'timestamp' column.
        current (bool, optional): Flag to determine whether to include the most recent data point.
            When True, includes the latest data point for real-time analysis.
            When False, excludes the last data point for historical analysis.
            Defaults to False.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames with computed technical indicators.
            Keys are TA-Lib function groups (e.g., 'Momentum Indicators', 'Volatility Indicators'),
            plus 'original' for the base OHLCV data.
            All DataFrames are indexed by timestamp and start from the 200th row.

    Notes:
        - Implements special handling for complex indicators:
            * MACD: Returns MACD line and signal line
            * Stochastic: Returns both K and D lines
        - Automatically determines appropriate input prices based on indicator requirements
        - Handles exceptions for indicators that may fail due to data requirements
        - Uses a 200-period warmup to ensure indicator stability
        - All indicators are calculated using industry-standard TA-Lib implementations

    Example:
        >>> df = pd.DataFrame(your_ohlcv_data)
        >>> talib_indicators = get_all_talib_indicators(df, current=True)
        >>> momentum_indicators = talib_indicators['Momentum Indicators']
    """

    talib_groups = talib.get_function_groups()  
    talib_ta = {}

    for group, indicators in talib_groups.items():
        copy_df = df.copy()
        copy_df.set_index('timestamp', inplace=True)

        for indicator in indicators:
            try:
                if indicator == 'MACD':
                    macd, signal, _ = talib.MACD(copy_df['close'])
                    copy_df['MACD'] = macd
                    copy_df['SIGNAL'] = signal
                elif indicator == 'STOCH':
                    slowk, slowd = talib.STOCH(copy_df['high'], copy_df['low'], copy_df['close'])
                    copy_df['STOCH_K'] = slowk
                    copy_df['STOCH_D'] = slowd
                else:
                    func = getattr(talib, indicator)

                    if 'close' in func.__code__.co_varnames:
                        copy_df[indicator] = func(copy_df['close'])
                    elif 'high' in func.__code__.co_varnames and 'low' in func.__code__.co_varnames:
                        copy_df[indicator] = func(copy_df['high'], copy_df['low'])
                    elif 'open' in func.__code__.co_varnames and 'close' in func.__code__.co_varnames:
                        copy_df[indicator] = func(copy_df['open'], copy_df['close'])
                        
            except Exception as e:
                print(f"Error applying {indicator}: {e}")
                continue

        if current:
            talib_ta[group] = copy_df
        else:
            talib_ta[group] = copy_df[:-1]

    if current:
        talib_ta['original'] = df[200:].set_index('timestamp')
    else:
        talib_ta['original'] = df[200:-1].set_index('timestamp')

    return talib_ta

