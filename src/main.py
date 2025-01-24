import pandas as pd

import config
import utils
from model import TrainXGBModel
from data import PrepareTrainingData


import os
print('!!!!!!!!!!!!!!!!!!!!!HERE!!!!!!!!!!!!!!!!!!!')
print(os.getcwd())
print('!!!!!!!!!!!!!!!!!!!!!HERE!!!!!!!!!!!!!!!!!!!')

if __name__ == "__main__":
        # params = config['params']
        realtime = True
        model_file = config.SAVED_MODEL_FILE_PATH
        # params_update = config.get('params_update', None)

        ## get data
        if realtime:
            data_preparer = PrepareTrainingData(filepath_or_dataframe = None, is_realtime=realtime, city='dallas', grid_size=0.05)
            data_preparer.assign_events_to_grid()
            data_preparer.integrate_data()
            data_preparer.apply_feature_engineering()

            data_preparer.split_data()
            data_preparer.save_data()
        
        
        else:
            data_preparer = PrepareTrainingData(filepath_or_dataframe='/path/to/your/data.csv', is_realtime=False, city='dallas', grid_size=0.05)
            data_preparer.assign_events_to_grid()
            data_preparer.integrate_data()
            data_preparer.apply_feature_engineering()

            data_preparer.split_data(train_ratio=0.8, test_ratio=0.1)
            data_preparer.save_data(train_path='/path/to/save/train_data.csv', valid_path='/path/to/save/valid_data.csv', test_path='/path/to/save/test_data.csv')

        
        ## Yesterday's prediction and ground truth
        df1 = pd.read_csv(config.PREDICTION_DATA_PATH)
        df2 = data_preparer.train_data[config.TARGET]

        ## Initialize and train the model
        model_trainer = TrainXGBModel(params_path = 'model/candidate-model-1/params.yaml', is_realtime = True, 
                                    model_file = config.SAVED_MODEL_FILE_PATH, train_input = data_preparer.train_data, test_input = data_preparer.test_data)
        model_trainer.train_xgb_model()


        predictions = model_trainer.predict(model_trainer.model)
        df3 = pd.DataFrame(predictions)
        df3.to_csv(config.PREDICTION_DATA_PATH, index=False)

        ## Save the trained model
        model_trainer.save_model()
        utils.save_three_values(df1, df2, df3, config.VISUALIZATION_DATA_PATH)
        