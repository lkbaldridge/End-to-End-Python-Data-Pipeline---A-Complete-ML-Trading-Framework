import pandas as pd
import numpy as np
import copy
from typing import Any, Iterator

def extract_value(y: Any) -> Any:
    """
    Extracts the first value from a dictionary or returns the input value if not a dictionary.

    Used primarily for handling River ML's prediction outputs which may be nested in dictionaries.

    Args:
        y (Any): Input value or dictionary to extract from.

    Returns:
        Any: First value from dictionary if input is dict, otherwise returns input unchanged.
    """
    if isinstance(y, dict):
        return next(iter(y.values()))
    return y


def create_datastream(X: pd.DataFrame, y: pd.Series, features: list[str] = None) -> Iterator[tuple[dict, int]]:
    """
    Creates a data stream iterator specifically formatted for River ML's progressive_val_score function.

    Transforms feature matrix X and target variable y into a stream of (x, y) tuples where x is a
    dictionary of features and y is the corresponding target value. This format is required for
    online learning evaluation using River ML's progressive validation scoring.

    Args:
        X (pd.DataFrame): Feature matrix containing predictor variables.
        y (pd.Series): Target variable series.
        features (List[str], optional): List of feature names to include in the stream.
            If None, all features from X are used. Defaults to None.

    Returns:
        Iterator[Tuple[dict, Any]]: Generator yielding tuples of (features_dict, target_value),
            where features_dict is a dictionary of feature names and their values.

    Example:
        >>> stream = create_datastream(X_train, y_train, ['feature1', 'feature2'])
        >>> metric = metrics.RMSE()
        >>> score = progressive_val_score(model, stream, metric)
    """

    if features == None:
        X_stream = X.to_dict('records')
    else:
        X_stream = X[features].to_dict('records')

    y_stream = list(y)
    data_stream = zip(X_stream, y_stream)

    return data_stream


def create_multistep_dataset(X: pd.DataFrame, y: pd.Series, timestep: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a time series dataset with multiple timesteps for sequence-based prediction models.

    Transforms the input data into a sliding window format where each row contains 'timestep'
    number of historical observations. This is particularly useful for LSTM, RNN, or other
    sequence-based models.

    Args:
        X (pd.DataFrame): Feature matrix containing time series data.
        y (pd.Series): Target variable series.
        timestep (int): Number of time steps to include in each sequence window.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - X_df: DataFrame where each row contains flattened sequence data with columns
                   named as feature1, feature2, ..., featureN for each timestep.
            - y_df: DataFrame containing target values aligned with the sequence windows.

    Notes:
        - Column names in returned X_df are formatted as {original_column_name}{timestep_number}
        - The number of rows in output will be len(X) - timestep + 1
        - All features are preserved and reshaped into a sliding window format
    """

    dataX, dataY = [], []
    X_val = X.values
    X_cols = X.columns
    y_array = np.array(y)
    X_array = np.array(X)

    for i in range(len(X)-timestep + 1):
        a = X_array[i:(i+timestep), :]   # Get the data for the current window
        dataX.append(a.reshape(-1))  # Reshape the window data into a 1D array
        dataY.append(y_array[i + timestep - 1])

    new_columns = [f"{col}{j+1}" for j in range(timestep) for col in X_cols]

    X_df = pd.DataFrame(dataX, columns=new_columns)
    y_df = pd.DataFrame(dataY)

    return X_df, y_df


def generate_target_vars(X: pd.DataFrame) -> pd.DataFrame:
    """
    Generates various financial target variables and technical indicators from OHLCV data.

    Creates multiple derived features including price shifts, percentage changes, and price
    differentials useful for financial time series prediction. Handles both timestamped
    and non-timestamped data.

    Args:
        X (pd.DataFrame): Input DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
            Must contain 'open', 'high', 'low', 'close' columns. Optional 'timestamp' column.

    Returns:
        pd.DataFrame: DataFrame with additional columns including:
            - Price shifts (low_shift, open_shift, high_shift, close_shift)
            - Price drops and peaks (drop_shift, peak_shift)
            - Percentage changes (drop_percent_shift, peak_percent_shift, change)
            - Price differentials (high_diff, low_diff, diff, diff_shift)
            Last row is removed to align shifted values.

    Notes:
        - All '_shift' columns are forward-looking (shifted -1)
        - Percentage calculations use the next day's opening price as reference
        - Returns data excluding the last row due to forward-looking calculations
    """

    df = copy.copy(X)
    cols = X.columns.to_list()
    
    if 'timestamp' not in cols:
        pass
    else:
        df = df.set_index('timestamp')

    df['low_shift'] = df['low'].shift(-1)
    df['open_shift'] = df['open'].shift(-1)
    df['drop_shift'] = df['low_shift'] - df['open_shift']
    df['drop_percent_shift'] = df['drop_shift']/df['open_shift'] * 100
    df['high_shift'] = df['high'].shift(-1)
    df['peak_shift'] = df['high_shift'] - df['open_shift']
    df['peak_percent_shift'] = df['peak_shift']/df['open_shift']*100
    df['change'] = df['close'].pct_change()*100
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = df['low'].diff()
    df['diff'] = df['close'].diff()
    df['diff_shift'] = df['diff'].shift(-1)
    df['change_shift'] = df['change'].shift(-1)
    df['close_shift'] = df['close'].shift(-1)

    #shifts = [col for col in df.columns.to_list() if 'shift' in col]
    return df[:-1]


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates and preprocesses a financial DataFrame by removing metadata and shift-based features.

    Handles special cases for financial data preprocessing, including ticker information
    preservation and feature filtering. Separates predictive features from target variables.

    Args:
        df (pd.DataFrame): Input DataFrame containing financial data and potentially
            metadata columns like 'ticker', 'price_id', and 'datasource'.

    Returns:
        pd.DataFrame: Cleaned DataFrame containing only predictive features
            (excluding shift-based features and metadata columns).

    Notes:
        - Preserves ticker information in DataFrame attributes if present
        - Removes metadata columns: 'price_id', 'datasource', 'ticker'
        - Filters out any columns containing 'shift' in their names
        - Maintains original data structure for remaining features
    """

    feats = df.columns.to_list()
    
    if 'ticker' in feats:
        df.attrs['ticker'] = df.loc[0, 'ticker']
        df.drop(columns=['price_id', 'datasource', 'ticker'], inplace=True)
        feats = df.columns.to_list()

    filtered = [feat for feat in feats if 'shift' not in feat]
    targets = [feat for feat in feats if 'shift' in feat]

    return df[filtered]