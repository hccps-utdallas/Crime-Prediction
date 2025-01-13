from data import PrepareTrainingData
import xgboost as xgb
# from config import PATH

class TrainXGBModel:
        
        def __init__(self, update):
            self.update = update
            self.train_input, self.label, self.predict_input = PrepareTrainingData()

        def train_xgb_model(self):
            # if self.update:
            #     self.model_updated = xgb.train(
            #         params_update,
            #         dtrain_new,
            #         num_boost_round=50, #Number of new trees to add
            #         xgb_model=model #Pass the initial model to continue training
            #     )
            # else:
            #     self.model = xgb.train(params, dtrain_initial, num_boost_round=params['n_estimators'])
            #     #  self.model = XGBoost(self.train_input, self.label, self.model)
            pass

        def save_model(self):
            if self.update:
                # self.model_updated.save_model(PATH['SAVE_MODEL_PATH'])
                pass
            else:
                # self.model.save_model(PATH['SAVE_MODEL_PATH'])
                pass
    
        def load_model(self):
            if self.update:
                # self.model = load_model(PATH['LOAD_MODEL_PATH'])
                pass
    
        def predict(self):
            predictions = self.model_updated.predict(self.predict_input)
            pass
    
        def evaluate(self):
            pass