import pandas as pd

class EstimDataLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self):
        df = pd.read_csv(self.path)
        text_columns = df.select_dtypes(
            exclude=["int", "float"]
        ).columns.tolist()
        df_cleaned = df.dropna()
        df_cleaned.reset_index(drop=True, inplace=True)
        return df_cleaned, text_columns
