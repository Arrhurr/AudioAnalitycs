from sklearn.preprocessing import StandardScaler, OrdinalEncoder

class EstimPreprocessor:
    def __init__(self):
        self.ordinal = OrdinalEncoder()
        self.text_columns = []

    def transform(self, X,text_columns):
        for text in text_columns:
            cat = X[[text]]
            X[text] = self.ordinal.fit_transform(cat)
        return X
