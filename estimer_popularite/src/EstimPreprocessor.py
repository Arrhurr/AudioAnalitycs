from sklearn.preprocessing import StandardScaler, OrdinalEncoder

class EstimPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.ordinal = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        self.text_columns = []

    def fit(self, X, text_columns):
        self.text_columns = text_columns
        self.ordinal.fit(X[self.text_columns])
        self.scaler.fit(X)
        return self

    def transform(self, X):
        X = X.copy()

        X[self.text_columns] = self.ordinal.transform(X[self.text_columns])
        return self.scaler.transform(X)

    def fit_transform(self, X, text_columns):
        self.fit(X, text_columns)
        return self.transform(X)
