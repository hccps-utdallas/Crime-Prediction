

UNEMPLOYMENT_DATA = {
    "Statistical_Area": [
        "Dallas-Fort Worth-Arlington, TX Metropolitan Statistical Area",
        "Dallas-Fort Worth-Arlington, TX Metropolitan Statistical Area"
        ],
        "Unemployment_Rate(%)": [3.9, 3.7],
        "Unemployment_Rank": [226, 228],
        "Year": [2024, 2023]
        }

CPI_DATA = {
    "Statistical_Area": [
        "Dallas-Fort Worth-Arlington, TX Metropolitan Statistical Area",
        ],
        "CPI(Annual)": [287.974],
        "Year": [2023]}

LOCAL_EVENT_DATA_PATH = "data/local_files/local_event_data.csv"

BATCH_TRAIN_FILE_PATH = "data/local_files/train.csv"
BATCH_VALID_FILE_PATH = "data/local_files/valid.csv"
BATCH_TEST_FILE_PATH = "data/local_files/test.csv"

## load and save most recent_date model, currently safe separately with date
SAVED_MODEL_FILE_PATH = "model/xgb_model.xgb"

## concatenate previous date input data with true event count
PREV_DATA_PATH = "data/deployed_files/last_day_input_data.csv"
## save the prediction after model update - use for visualization
PREDICTION_DATA_PATH = "data/deployed_files/prediction_data.csv"

VISUALIZATION_DATA_PATH = "data/deployed_files/visualization_data.csv"

SAVED_GRID_PATH = "data/preload_grid"

DPD_API_URL = "www.dallasopendata.com"
DPD_DATASET_ID = "qv6i-rri7"

FEATURES_LIST = ['lon_bin', 'lat_bin', 'date_year', 'day_of_year_sin', 'day_of_year_cos', 'week_of_year_sin', 'week_of_year_cos',
            'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'Unemployment_Rate(%)', 'Unemployment_Rank', 'CPI(Annual)']
# ['lon_bin', 'lat_bin', 'date_year', 'day_of_year_sin', 'day_of_year_cos', 'week_of_year_sin', 'week_of_year_cos',
#             'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'Unemployment_Rate(%)', 'Unemployment_Rank', 'CPI(Annual)', 'Nearest Station', 'Distance (km)']
TARGET = ['unique_event_count']
LOC = ['lon_bin', 'lat_bin']

# # Define model parameters
# params = {
#     'objective': 'reg:squarederror',
#     'eval_metric': 'rmse',
#     'eta': 0.1,
#     'max_depth': 6,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'n_estimators': 100
# }