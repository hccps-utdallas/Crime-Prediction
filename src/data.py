import os
import pandas as pd
import numpy as np
import datetime

import json
import meteostat
from sodapy import Socrata

import config
from utils import clean_up_loc, load_local_csv

# import dotenv

# dotenv(../../../.env)
class FeatureEngineering:
    def __init__(self, df):
        self.df = df
    
    def get_time_features(self):
        # Extract year, day of year, week of year, and month-day components
        self.df['date_year'] = self.df['date'].apply(lambda x: x.timetuple().tm_year)
        # self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['day_of_year'] = self.df['date'].apply(lambda x: x.timetuple().tm_yday)
        self.df['week_of_year'] = pd.to_datetime(self.df['date']).dt.isocalendar().week

        # Encode day_of_year cyclically
        self.df['day_of_year_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['day_of_year_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)

        # Encode week_of_year cyclically
        self.df['week_of_year_sin'] = np.sin(2 * np.pi * self.df['week_of_year'] / 52)
        self.df['week_of_year_cos'] = np.cos(2 * np.pi * self.df['week_of_year'] / 52)

    def get_dataframe(self):
        return self.df[config.FEATURES_LIST+config.TARGET]


class PrepareTrainingData:
    
    # def __init__(self, grid_list, from_local=False, local_file_path = None):
    #     """Use the grid list and the data prepartion aim to attach different sources of data to the grid
    #     """
    #     self.grid = grid_list
    #     self.from_local = from_local
    #     self.file_path = local_file_path
    #     self.df = pd.DataFrame()

    def __init__(self, filepath_or_dataframe, is_realtime=False, city = 'dallas', 
                 grid_size = 0.05):
        
        ## set online/offline mode and time
        self.is_realtime = is_realtime
        self.grid_size = grid_size

        if is_realtime:
            self.fixed_date = pd.to_datetime(datetime.datetime.today()).date()
            self.data = self.load_event_data()
        else:
            self.fixed_date = None
            self.data = self.load_event_data(filepath_or_dataframe)
        
        # Load fixed grid coordinates
        self.grids_df = pd.read_json(f"{config.SAVED_GRID_PATH}/{city.lower()}_{str(grid_size).replace('.', '')}.json", 
                                     orient='records', lines=True)
        
        self.weather_df = pd.DataFrame()
        self.df_macro = pd.DataFrame()

    def load_event_data(self, data_source = None):
        """Fetch event data from a CSV file or an API endpoint."""
        
        if self.is_realtime:
            return self.fetch_event_data_from_api()
        
        else:
            if isinstance(data_source, str):
                return pd.read_csv(data_source)
            elif isinstance(data_source, pd.DataFrame):
                return data_source
            else:
                raise ValueError("Data input must be a file path or a pandas DataFrame")

    def fetch_event_data_from_api(self):
        try:
            client = Socrata(config.DPD_API_URL, None)
            date_filter = f"""date1='{(self.fixed_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d 00:00:00.0000000")}'"""
            print(f"Fetching data with filter: {date_filter}")
            
            results = client.get(config.DPD_DATASET_ID, where=date_filter)
            results_df = pd.DataFrame.from_records(results)
            
            if results_df.empty:
                raise ValueError("No data retrieved from API")
                
            print(f"Retrieved {len(results_df)} records")
            return results_df
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data from API: {str(e)}")

    def assign_events_to_grid(self):
        # Assign each event to the nearest grid center
        if self.is_realtime:
            if 'geocoded_column' not in self.data.columns:
                raise ValueError(f"Required column 'geocoded_column' missing from data. Available columns: {self.data.columns.tolist()}")
                    # Add logging
            print("Data columns:", self.data.columns.tolist())
            print("Sample data:", self.data.head(1).to_dict())
            self.data['longitude'] = self.data['geocoded_column'].apply(lambda x: float(x['longitude']) if isinstance(x, dict) and 'longitude' in x else None)
            self.data['latitude'] = self.data['geocoded_column'].apply(lambda x: float(x['latitude']) if isinstance(x, dict) and 'latitude' in x else None)

        else:
            self.data[['longitude', 'latitude']] = self.data['Location1'].apply(lambda x: pd.Series(clean_up_loc(x)))

        self.data['lon_bin'] = ((self.data['longitude'] // self.grid_size) * self.grid_size + self.grid_size / 2).round(3)
        self.data['lat_bin'] = ((self.data['latitude'] // self.grid_size) * self.grid_size + self.grid_size / 2).round(3)
        
        # Merge events with grid centers to ensure all grid centers are included
        if 'Incident Number w/year' in self.data.columns:
            self.grouped_data = self.data.groupby(['date1', 'lon_bin', 'lat_bin'])['Incident Number w/year'].nunique().reset_index(name='unique_event_count')
        else:
            self.grouped_data = self.data.groupby(['date1', 'lon_bin', 'lat_bin'])['incidentnum'].nunique().reset_index(name='unique_event_count')
        
        self.grids_df = self.grids_df.merge(pd.DataFrame(self.grouped_data['date1'].unique(), columns=['date1']), how = 'cross')
        # self.binned_event_data = self.grids_df.merge(grouped_data, how = 'left', on=['date1', 'lon_bin', 'lat_bin'])
        # self.binned_event_data['unique_event_count'].fillna(0, inplace=True)

        return self.grouped_data
    
    def fetch_load_weather_data(self, start_date, end_date=None, timezone='US/Central', max_retries=5):
        
        if end_date is None:
            end_date = start_date
        start_time = datetime.datetime.combine(start_date, datetime.datetime.min.time())
        end_time = datetime.datetime.combine(end_date, datetime.datetime.min.time())
        # datetime.datetime.strptime(end_date, '%Y-%m-%d')

        weather_data = []
        
        for longitude, latitude in self.grids_df[['lon_bin', 'lat_bin']].values:
            
            ## initialize the location
            radius_km = 50  # Find weather station within 50 km
            max_stations = 10
            stations = meteostat.Stations().nearby(latitude, longitude).fetch(max_stations)

            data_found = False
            # print(f"Found {len(stations)} nearby weather stations within {radius_km} km")
            # traverse each weather station
            for station_id in stations.index:
                for attempt in range(max_retries + 1):
                    # print the current station
                    try:
                        # print(f"Attempt {attempt + 1}: Requesting data from station {station_id}")
                        
                        data_daily = meteostat.Daily(station_id, start_time, end_time)
                        data = data_daily.fetch()

                        # return if fetch data successfully
                        if not data.empty:
                            # print(f"Data found from station {station_id}!")
                            data['lon_bin'] = longitude
                            data['lat_bin'] = latitude
                            weather_data.append(data.reset_index(drop = False))
                            data_found = True
                            break
                    except Exception as e:
                        print(f"Error fetching data: {e}")
                        if attempt == max_retries:
                            print("Max retries reached, moving to next station.")
                if data_found:
                    break
                # data['longitude'] = longitude
                # data['latitude'] = latitude
                # weather_data.append(data)
        
        if weather_data:
            self.weather_df = pd.concat(weather_data, ignore_index=True)
        else:
            self.weather_df = pd.DataFrame()
        
        ## clean up format
        self.weather_df.rename(columns={'time': 'date'}, inplace=True)
        self.weather_df['date'] = pd.to_datetime(self.weather_df['date']).dt.date

        return self.weather_df

    @staticmethod
    def load_macro_data():
        df_unemp = pd.DataFrame(config.UNEMPLOYMENT_DATA)
        df_cpi = pd.DataFrame(config.CPI_DATA)
        df_macro = pd.merge(df_unemp, df_cpi, on=['Year','Statistical_Area'], how='inner')

        return df_macro
    
    def integrate_data(self):
        """ Integrate macroeconomic and weather data into the main dataset. """
        if self.is_realtime:
            dates = pd.DataFrame({'date': [self.fixed_date]})
        else:
            dates = pd.DataFrame({'date': pd.date_range(self.data['date1'].min(), self.data['date1'].max()).date})
        
        self.full_grid = pd.merge(dates, self.grids_df, how='cross')
        self.df_macro = self.load_macro_data()
        self.fetch_load_weather_data(self.full_grid['date'].min(), self.full_grid['date'].max())

        # Add year column for merging with macro data
        self.full_grid['Year'] = self.full_grid['date'].apply(lambda x: x.year)
        self.full_grid = self.full_grid.merge(self.df_macro, on='Year', how='left')
        self.full_grid = self.full_grid.merge(self.weather_df, on=['date', 'lon_bin', 'lat_bin'], how='left')
        self.full_grid =  self.full_grid.merge(self.grouped_data, on=['date1', 'lon_bin', 'lat_bin'], how='left')
        self.full_grid['unique_event_count'].fillna(0, inplace=True)
        
    def apply_feature_engineering(self):
        """Apply feature engineering to the data."""
        
        fe = FeatureEngineering(self.full_grid)
        fe.get_time_features()
        self.full_grid_engineered = fe.get_dataframe()
    
    def split_data(self, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
        """Split the data into training, validation, and testing datasets."""
        if not self.is_realtime:
            # Ensure the ratios sum to 1
            assert np.isclose(train_ratio + valid_ratio + test_ratio, 1), "Ratios must sum to 1"

            # Sort data by date to ensure temporal split
            self.full_grid_engineered = self.full_grid_engineered.sort_values('date')
            total_count = len(self.full_grid_engineered)

            train_end = int(total_count * train_ratio)
            valid_end = train_end + int(total_count * valid_ratio)

            self.train_data = self.full_grid_engineered.iloc[:train_end]
            self.valid_data = self.full_grid_engineered.iloc[train_end:valid_end]
            self.test_data = self.full_grid_engineered.iloc[valid_end:]
        # else:
        #     raise ValueError("Only offline data needs to be split.")
        else:
            # All data before the current date is used for training, current date for testing
            # self.full_grid_engineered = self.full_grid_engineered.sort_values('date')
            
            self.train_data = load_local_csv(config.PREV_DATA_PATH)[config.FEATURES_LIST].merge(
                self.full_grid_engineered[['lon_bin', 'lat_bin', 'unique_event_count']], on = ['lon_bin', 'lat_bin'], how = 'left')
            self.test_data = self.full_grid_engineered[config.FEATURES_LIST]

    def save_data(self):
        
        """Save the split data sets to files."""
        
        if self.is_realtime:
            self.test_data.to_csv(config.PREV_DATA_PATH, index=False)
        
        else:
            self.train_data.to_csv(config.BATCH_TRAIN_FILE_PATH, index=False)
            self.valid_data.to_csv(config.BATCH_VALID_FILE_PATH, index=False)
            self.test_data.to_csv(config.BATCH_TEST_FILE_PATH, index=False)
                
