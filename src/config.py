

UNEMPLOYMENT_DATA = {
    "Statistical_Area": [
        "Dallas-Fort Worth-Arlington, TX Metropolitan Statistical Area",
        "Dallas-Fort Worth-Arlington, TX Metropolitan Statistical Area"
        ],
        "Unemployment_Rate(%)": [3.9, 3.7],
        "Unemployment_Rank": [226, 228],
        "Year": [2024, 2023]
        }

CPI_DATA = {}

LOCAL_EVENT_DATA_PATH = ""

SAVED_GRID_PATH = "../../data/preload_grid"

DPD_API_URL = "www.dallasopendata.com"
DPD_DATASET_ID = "qv6i-rri7"

FEATURES_LIST = ['lon_bin', 'lat_bin', 'date_year', 'day_of_year_sin', 'day_of_year_cos', 'week_of_year_sin', 'week_of_year_cos',
            'tavg', 'tmin', 'tmax', 'prcp', 'snow', 'Unemployment_Rate(%)', 'Unemployment_Rank', 'CPI(Annual)']
TARGET = 'unique_event_count'