from IPython.display import display, HTML, Image
import pandas as pd
from typing import Dict, Any, Union

def display_side_by_side(*args: Union[pd.DataFrame, pd.Series]):
    """
    Renders multiple pandas DataFrames or Series side-by-side in a Jupyter notebook.

    Accepts an arbitrary number of pandas DataFrame or Series objects and displays
    them horizontally for easy comparison.

    Args:
        *args (Union[pd.DataFrame, pd.Series]): A variable number of pandas
            DataFrame or Series objects to display.
    """
    html_str = '<div style="display: flex; justify-content: flex-start;">'
    for df in args:
        if isinstance(df, pd.Series):
            df = df.to_frame()
        try:
            html_str += f'<div style="flex: 1; padding: 0;">{df.to_html(index=True)}</div>'
        except Exception:
            html_str += f'<div style="flex: 1; padding: 0;">{df.to_html()}</div>'
    html_str += '</div>'
    
    display(HTML(html_str))


def check_dict_length_index(dict_df: Dict[str, Any]):
    """
    Checks and prints the length and last index of each item in a dictionary.

    This is a debugging utility function to quickly inspect the contents of a 
    dictionary, typically one containing pandas DataFrames or similar objects.

    Args:
        dict_df (Dict[str, Any]): A dictionary where keys are strings and values
            are objects that support len() and have an .index attribute.
    """
    for key in dict_df.keys():
        print(key, len(dict_df[key]))
        print(dict_df[key].index[-1])
        print('\n')

def get_column_indexes(df: pd.DataFrame) -> Dict[int, str]:
    """
    Creates a dictionary mapping column indexes to column names for a DataFrame.

    This utility function is useful for quickly looking up column names by their
    integer position.

    Args:
        df (pd.DataFrame): The input pandas DataFrame.

    Returns:
        Dict[int, str]: A dictionary where keys are the integer column indices
            and values are the corresponding column name strings.
    """
    col_dict = {i: df.columns[i] for i in range(len(df.columns))}
    return col_dict

def check_same_length_and_identify(dataframes: Dict[str, Any]) -> bool | Dict[str, int]:
    """
        Checks if all DataFrames in a dictionary have the same length and identifies
        the DataFrames that have lengths different from the majority.

        Args:
            dataframes (Dict[str, Any]): A dictionary where keys are strings and values
                are pandas DataFrames.

        Returns:
            bool: True if all DataFrames have the same length, 
                otherwise returns a dictionary where keys are the names of the
                DataFrames with lengths different from the most common length, and
                values are their lengths.
    """


    lengths = {key: len(df) for key, df in dataframes.items()} 
    
    # Find the most common length
    most_common_length = max(set(lengths.values()), key=list(lengths.values()).count)

    # Identify DataFrames with lengths different from the most common length
    different_lengths = {key: length for key, length in lengths.items() if length != most_common_length}

    # Check if first and last index are the same
    first_index = next(iter(dataframes.values())).index[0]
    last_index = next(iter(dataframes.values())).index[-1]

    different_indices = {}
    for key, df in dataframes.items():
        if df.index[0] != first_index or df.index[-1] != last_index:
            different_indices[key] = {"first_index": df.index[0], "last_index": df.index[-1]}

    if not different_lengths:
        print("All DataFrames have the same length.")
    else:
        return different_lengths  # Return DataFrames with different lengths
    
    if not different_indices:
        print("All DataFrames have the same first and last index.")
    else:
        return different_indices
    return True