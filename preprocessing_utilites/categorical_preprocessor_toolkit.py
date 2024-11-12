import pandas as pd
from sklearn.preprocessing import LabelEncoder

class CategoricalUtilityToolkit:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def handle_missing(self, columns: list = None, fill_value: str = 'missing') -> pd.DataFrame:

        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns
        df_filled = self.df.copy()
        df_filled[columns] = df_filled[columns].fillna(fill_value)
        return df_filled

    def label_encode(self, columns: list) -> pd.DataFrame:
        df_encoded = self.df.copy()
        le = LabelEncoder()

        for col in columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        return df_encoded

    def label_encode(self, columns: list) -> pd.DataFrame:

        df_encoded = self.df.copy()
        le = LabelEncoder()

        for col in columns:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        return df_encoded

    def one_hot_encode(self, columns: list) -> pd.DataFrame:

        df_encoded = pd.get_dummies(self.df, columns=columns, drop_first=True)
        return df_encoded

    def handle_rare_categories(self, columns: list, threshold: float = 0.05) -> pd.DataFrame:

        df_processed = self.df.copy()

        for col in columns:
            freq = df_processed[col].value_counts(normalize=True)
            rare_labels = freq[freq < threshold].index
            df_processed[col] = df_processed[col].apply(
                lambda x: 'Rare' if x in rare_labels else x)

        return df_processed

if __name__ == '__main__':

    data = {
        'city': ['Paris', 'London', 'Berlin', 'Paris', 'Berlin', None, 'London', 'Rome'],
        'room_type': ['Entire home', 'Private room', 'Shared room', 'Entire home', 'Private room', 'Shared room', 'Entire home', None],
        'host_is_superhost': ['t', 'f', 't', None, 'f', 't', 'f', 't']
    }
    df = pd.DataFrame(data)

    cat_toolkit = CategoricalUtilityToolkit(df)

    df_no_missing = cat_toolkit.handle_missing()
    print("DataFrame after handling missing values:")
    print(df_no_missing)

    df_label_encoded = cat_toolkit.label_encode(columns=['city', 'room_type'])
    print("\nDataFrame after label encoding:")
    print(df_label_encoded)

    df_one_hot_encoded = cat_toolkit.one_hot_encode(columns=['room_type'])
    print("\nDataFrame after one-hot encoding:")
    print(df_one_hot_encoded)

    df_rare_handled = cat_toolkit.handle_rare_categories(
        columns=['city'], threshold=0.20)
    print("\nDataFrame after handling rare categories:")
    print(df_rare_handled)