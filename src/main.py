import config
from model import TrainXGBModel
from data import PrepareTrainingData, FeatureEngineering


if __name__ == "__main__":

        params = config['params']
        is_realtime = config.get('is_realtime', False)
        model_file = config.get('model_file', None)
        params_update = config.get('params_update', None)

        ## get data
        if is_realtime:
            data_preparer = PrepareTrainingData(is_realtime=True, city='dallas', grid_size=0.05)
            data_preparer.assign_events_to_grid()
            data_preparer.integrate_data()
            data_preparer.apply_feature_engineering()
            
            data_preparer.split_data(train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1)
            data_preparer.save_data(train_path='/path/to/save/train_data.csv', test_path='/path/to/save/test_data.csv')
        
        
        else:
            data_preparer = PrepareTrainingData(filepath_or_dataframe='/path/to/your/data.csv', is_realtime=False, city='dallas', grid_size=0.05)
            data_preparer.assign_events_to_grid()
            data_preparer.integrate_data()
            data_preparer.apply_feature_engineering()

            data_preparer.split_data(train_ratio=0.8, test_ratio=0.1)
            data_preparer.save_data(train_path='/path/to/save/train_data.csv', test_path='/path/to/save/test_data.csv')


        # Initialize and train the model
        model_trainer = TrainXGBModel(params, is_realtime, model_file)
        model_trainer.train_xgb_model()

        # Save the trained model
        model_trainer.save_model()

        # Load the model and make predictions
        predictions = model_trainer.predict(model_trainer.model)