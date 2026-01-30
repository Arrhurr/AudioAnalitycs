class EstimPipeline:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def train(self, X_train, y_train):
        X_scaled = self.preprocessor.fit_transform(X_train)
        self.model.train(X_scaled, y_train)

    def predict(self, X):
        X_scaled = self.preprocessor.transform(X)
        return self.model.predict(X_scaled)
