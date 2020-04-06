import lightgbm as lgb


class LgbmWrapper:
    def __init__(self, **param):
        self.param = param
        self.trained = None

    def fit(self, X_train, y_train):
        train = lgb.Dataset(X_train, label=y_train)
        self.trained = lgb.train(self.param, train)
        self.feature_importances_ = self.trained.feature_importance()
        return self.trained

    def predict(self, X_val):
        return self.trained.predict(X_val, num_iteration=self.trained.best_iteration)
