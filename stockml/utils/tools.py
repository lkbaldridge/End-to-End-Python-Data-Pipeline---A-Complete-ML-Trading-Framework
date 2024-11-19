from IPython.display import display, HTML, Image
import pandas

def display_side_by_side(*args):
    html_str = '<div style="display: flex; justify-content: flex-start;">'
    for df in args:
        if isinstance(df, pandas.Series):
            df = pandas.DataFrame(df)
        try:
            html_str += f'<div style="flex: 1; padding: 0;">{df.to_html(index=True)}</div>'
        except Exception:
            html_str += f'<div style="flex: 1; padding: 0;">{df.to_html()}</div>'
    html_str += '</div>'
    
    display(HTML(html_str))

def check_dict_length_index(dict_df):
    for key in dict_df.keys():
        print(key, len(dict_df[key]))
        print(dict_df[key].index[-1])
        print('\n')

def get_column_indexes(df):
    col_dict = {i: df.columns[i] for i in range(len(df.columns))}
    return col_dict