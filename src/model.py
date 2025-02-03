import xgboost as xgb
import numpy as np

import yaml
from config import FEATURES_LIST, TARGET, SAVED_MODEL_FILE_PATH

class TrainXGBModel:

    def __init__(self, params_path, train_input, test_input, validation_input = None, is_realtime = True, model_file = None):
        self.is_realtime = is_realtime
        if is_realtime:
            self.train_input, self.test_input = train_input, test_input
            self.model_file = model_file
            with open(params_path, 'r') as file:
                self.params_update = yaml.safe_load(file)
        else:
            with open(params_path, 'r') as file:
                self.params = yaml.safe_load(file)
            self.train_input, self.test_input, self.validation_input = train_input, test_input, validation_input

    def poisson_obj(self, preds, dtrain):
        labels = dtrain.get_label()
        preds = np.exp(preds)  # Ensure predictions are positive
        grad = preds - labels  # Gradient: difference between prediction and true count
        hess = preds           # Hessian: prediction values (positive)
        return grad, hess

    def train_xgb_model(self):
        
        if self.is_realtime:
            dtrain_new = xgb.DMatrix(self.train_input[FEATURES_LIST], label=self.train_input[TARGET], enable_categorical = True)
            self.model = xgb.train(
                self.params_update,
                dtrain_new,
                num_boost_round=self.params_update['n_estimators'], #Number of new trees to add
                xgb_model = self.load_model(), #Pass the initial model to continue training,
                obj=self.poisson_obj
            )

        else:
            dtrain_initial = xgb.DMatrix(self.train_input[FEATURES_LIST], label=self.train_input[TARGET])
            self.model = xgb.train(self.params, dtrain_initial, num_boost_round=self.params['n_estimators'], obj=self.poisson_obj)
            #  self.model = XGBoost(self.train_input, self.label, self.model)

    def save_model(self):
        self.model.save_model(SAVED_MODEL_FILE_PATH)

    def load_model(self):
        if self.is_realtime:
            model = xgb.Booster().load_model(SAVED_MODEL_FILE_PATH)
            
            return model

    def predict(self, model):
        pred_input = xgb.DMatrix(self.test_input[FEATURES_LIST], enable_categorical = True)
        predictions = model.predict(pred_input)
        y_pred_exp = np.exp(predictions)  # Back-transform predictions
        y_pred_rounded = np.clip(np.round(y_pred_exp).astype(int), 0, None) 
        return y_pred_rounded

    # def evaluate(self):
    #     pass
