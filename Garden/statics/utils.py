import pandas as pd
from tabulate import tabulate


def show_markdown_df(df: pd.DataFrame):
    pd.options.display.max_columns = None
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=True))
