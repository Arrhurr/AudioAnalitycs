import pandas as pd

class EstimDataLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.path)
        df_cleaned = df.dropna()
        df_cleaned.reset_index(drop=True, inplace=True)
        return df_cleaned
