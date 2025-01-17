import xgboost as xgb
from config import FEATURES_LIST, TARGET

class TrainXGBModel:

    def __init__(self, params, train_input, test_input, validation_input = None, is_realtime = True, model_file = None):
        self.is_realtime = is_realtime
        if is_realtime:
            self.train_input, self.predict_input = train_input, test_input
            self.model_file = model_file
            self.params_update = params
        else:
            self.params = params
            self.train_input, self.test_input, self.validation_input = train_input, test_input, validation_input

    def train_xgb_model(self):
        
        if self.is_realtime:
            dtrain_new = xgb.DMatrix(self.train_input[FEATURES_LIST], label=self.train_input[TARGET])
            self.model = xgb.train(
                self.params_update,
                dtrain_new,
                num_boost_round=50, #Number of new trees to add
                xgb_model = self.load_model() #Pass the initial model to continue training
            )
        else:
            dtrain_initial = xgb.DMatrix(self.train_input[FEATURES_LIST], label=self.train_input[TARGET])
            self.model = xgb.train(self.params, dtrain_initial, num_boost_round=self.params['n_estimators'])
            #  self.model = XGBoost(self.train_input, self.label, self.model)

    def save_model(self):
        self.model.save_model(CONFIG.PATH['SAVE_MODEL_PATH'])

    def load_model(self):
        if self.is_realtime:
            model = xgb.Booster().load_model(self.model_file)
            
            return model

    def predict(self, model):
        predictions = model.predict(self.test_input)
        return predictions

    # def evaluate(self):
    #     pass