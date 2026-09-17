from ..utils import tools
import plotly.express as px
from statsmodels.stats.descriptivestats import describe

def display_and_describe(df):
    """
    Displays visualizations and descriptive statistics for a given DataFrame.

    This function generates and displays two visualizations:
    1. A histogram of all columns in the DataFrame.
    2. A horizontal box plot of all columns in the DataFrame.

    Additionally, it calculates descriptive statistics for the DataFrame and 
    displays them side by side using a helper tool.

    Args:
        df (pandas.DataFrame): The input DataFrame that contains the data to be visualized and described.

    Returns:
        dict: A dictionary containing the descriptive statistics of the DataFrame, 
              where each key is a column name and each value is a dictionary of statistics 
              (e.g., mean, standard deviation, percentiles).
    
    Notes:
        - The function uses Plotly's `px.histogram` and `px.box` to generate the visualizations.
        - Descriptive statistics are calculated using a helper function from statsmodels `describe(df)`.
        - The `tools.display_side_by_side` function is used to display the first half 
          and second half of the descriptive statistics side by side for better readability.
    """

    fig1 = px.histogram(df)
    fig2 = px.box(df, orientation='h')

    fig1.show()
    fig2.show()
    
    describe_df = describe(df) 
    tools.display_side_by_side(describe_df[:15], describe_df[15:])

    return describe_df.to_dict()


def categorize_instances(y_df, lower_bound, upper_bound):
    """
    Categorizes instances based on specified percentiles and visualizes the results.

    This function categorizes values in the 'change_shift' column of the input DataFrame into three categories:
    - `-1`: Values below the lower percentile (e.g., lower 33rd percentile).
    - `0`: Values between the lower and upper percentiles (e.g., between 33rd and 66th percentiles).
    - `1`: Values above the upper percentile (e.g., upper 66th percentile).

    The function also generates a histogram to visualize the distribution of these categories.

    Args:
        y_df (pandas.DataFrame): A DataFrame containing at least two columns: 'change_shift' and 'diff_shift'.
        lower_bound (float): The lower percentile bound (e.g., 0.33 for the 33rd percentile).
        upper_count (float): The upper percentile bound (e.g., 0.66 for the 66th percentile).

    Returns:
        pandas.DataFrame: A copy of the input DataFrame with an additional column 'change_encoded', 
                          where values are encoded as:
                          - `-1` for values below the lower bound,
                          - `0` for values between the bounds,
                          - `1` for values above the upper bound.
    
    Notes:
        - The function uses Plotly's `px.histogram` to visualize the distribution of categorized instances.
        - Percentiles are calculated using Pandas' `quantile()` method.
        - The new column 'change_encoded' represents whether each instance falls below or above 
          certain percentiles or within a middle range.
        
    """

    pct_lb = y_df['change_shift'].quantile(lower_bound)
    pct_ub = y_df['change_shift'].quantile(upper_bound)
    print(f'Lower bound value:{pct_lb:.4f}//Upper bound value:{pct_ub:.4f}')

    y_copy = y_df[['change_shift', 'diff_shift']].copy()
    y_copy['change_encoded'] = y_copy['change_shift'].apply(lambda x: -1 if x < pct_lb else (1 if x > pct_ub else 0))

    labels_histogram = px.histogram(y_copy['change_encoded'], histnorm='')
    labels_histogram.update_traces(
        marker_line_width=2,
        marker_line_color="darkslategray",
    )

    labels_histogram.show()

    return y_copy