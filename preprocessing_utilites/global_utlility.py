import pandas as pd


def remove_nan(df: pd.DataFrame, method: str = 'drop', columns: list = None, strategy: str = 'mean') -> pd.DataFrame:
    """
    Removes or handles NaN values from the DataFrame.

    Parameters:
        df (pd.DataFrame): The input pandas DataFrame.
        method (str): The method to handle NaN values. Options are 'drop' (default), 'impute'.
                      - 'drop': Drops rows or columns with NaN values.
                      - 'impute': Fills missing values with a specified strategy.
        columns (list): List of specific columns to process. If None, all columns are processed.
        strategy (str): Imputation strategy (used only if method='impute'). Options are 'mean', 'median', or 'mode'.
                       - 'mean': Replaces NaN with the column mean.
                       - 'median': Replaces NaN with the column median.
                       - 'mode': Replaces NaN with the column mode (most frequent value).

    Returns:
        pd.DataFrame: The DataFrame with NaN values handled based on the specified method.
    """

    if columns is None:
        columns = df.columns

    if method == 'drop':
        # Drop rows with NaN values in the specified columns
        df_cleaned = df.dropna(subset=columns)

    elif method == 'impute':
        df_cleaned = df.copy()
        for col in columns:
            if strategy == 'mean':
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
            elif strategy == 'median':
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
            elif strategy == 'mode':
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
            else:
                raise ValueError(
                    "Invalid strategy. Choose from 'mean', 'median', or 'mode'.")
    else:
        raise ValueError("Invalid method. Choose from 'drop' or 'impute'.")

    return df_cleaned
