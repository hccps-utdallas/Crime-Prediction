import os
import pandas as pd
import numpy as np
from datetime import datetime

import json
import meteostat
from sodapy import Socrata

import config
from utils import find_nearest_station


# import dotenv

# dotenv(../../../.env)
class FeatureEngineering:
    def __init__(self, df, features_list):
        self.df = df
        self.features_list = features_list
    
    def get_time_features(self):
        # Extract year, day of year, week of year, and month-day components
        self.df['date_year'] = self.df['date'].dt.year
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['week_of_year'] = self.df['date'].dt.isocalendar().week

        # Encode day_of_year cyclically
        self.df['day_of_year_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['day_of_year_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)

        # Encode week_of_year cyclically
        self.df['week_of_year_sin'] = np.sin(2 * np.pi * self.df['week_of_year'] / 52)
        self.df['week_of_year_cos'] = np.cos(2 * np.pi * self.df['week_of_year'] / 52)

    def preprocessing(self):
        self.data = self.data.dropna()
        # self.data  = self.data[feature list]
        pass
    
    def get_dataframe(self):
        return self.df[[self.features_list]]



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
        else:
            self.fixed_date = None
        
        self.data = self.load_event_data(filepath_or_dataframe)
        
        # Load fixed grid coordinates
        self.grids_df = pd.read_json(f"{config.SAVED_GRID_PATH}/{city.lower()}_{str(grid_size).replace('.', '')}.json", 
                                     orient='records', lines=True)
        
        self.weather_df = pd.DataFrame()
        self.df_macro = pd.DataFrame()

    def load_event_data(self, data_source):
        """Fetch event data from a CSV file or an API endpoint."""
        
        if self.is_realtime:
            return self.fetch_data_from_api()
        
        else:
            if isinstance(data_source, str):
                return pd.read_csv(data_source)
            elif isinstance(data_source, pd.DataFrame):
                return data_source
            else:
                raise ValueError("Data input must be a file path or a pandas DataFrame")

    def fetch_event_data_from_api(self):
        """Fetch data from API for real-time processing."""
        # Unauthenticated client only works with public data sets. Note 'None'
        # in place of application token, and no username or password:
        client = Socrata(config.DPD_API_URL, None)

        # Example authenticated client (needed for non-public datasets):
        # client = Socrata(www.dallasopendata.com,
        #                  MyAppToken,
        #                  username="user@example.com",
        #                  password="AFakePassword")
        
        results = client.get(config.DPD_DATASET_ID, 
                             where=f"""date1='{(self.fixed_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d 00:00:00.0000000")}'""")
        results_df = pd.DataFrame.from_records(results)

        return results_df

    def assign_events_to_grid(self):
        # Assign each event to the nearest grid center
        self.data['lon_bin'] = ((self.data['longitude'] // self.grid_size) * self.grid_size + self.grid_size / 2).round(3)
        self.data['lat_bin'] = ((self.data['latitude'] // self.grid_size) * self.grid_size + self.grid_size / 2).round(3)
        
        # Merge events with grid centers to ensure all grid centers are included
        self.binned_event_data = self.grid_df.merge(self.data, how='left', on=['lon_bin', 'lat_bin'])
        self.binned_event_data.groupby(['date1', 'lon_bin', 'lat_bin'])['Incident Number w/year'].nunique().reset_index(name='unique_event_count')
        self.binned_event_data['unique_event_count'].fillna(0, inplace=True)

        return self.binned_event_data
    
    def fetch_load_weather_data(self, start_date, end_date=None, timezone='US/Central', max_retries=5):
        
        if end_date is None:
            end_date = start_date
        start_time = datetime.strptime(start_date, '%Y-%m-%d')
        end_time = datetime.strptime(end_date, '%Y-%m-%d')

        weather_data = []
        
        for longitude, latitude in self.grids_df[['lon_bin', 'lat_bin']].values:
            
            ## initialize the location
            radius_km = 50  # Find weather station within 50 km
            max_stations = 10
            stations = meteostat.Stations().nearby(latitude, longitude).fetch(max_stations)
            print(f"Found {len(stations)} nearby weather stations within {radius_km} km")

            # traverse each weather station
            for station_id in stations.index:
                for attempt in range(max_retries + 1):
                    # print the current station
                    try:
                        print(
                            f"Attempt {attempt + 1}: Requesting data from station {station_id}")
                        
                        data_daily = meteostat.Daily(station_id, start_time, end_time)
                        data = data_daily.fetch()

                        # return if fetch data successfully
                        if not data.empty:
                            print(f"Data found from station {station_id}!")
                            data['longitude'] = longitude
                            data['latitude'] = latitude
                            weather_data.append(data)
                            break
                    except Exception as e:
                        print(f"Error fetching data: {e}")
                        if attempt == max_retries:
                            print("Max retries reached, moving to next station.")

                # data['longitude'] = longitude
                # data['latitude'] = latitude
                # weather_data.append(data)
        
        if weather_data:
            self.weather_df = pd.concat(weather_data, ignore_index=True)
        else:
            self.weather_df = pd.DataFrame()

        return self.weather_df

    @staticmethod
    def load_macro_data(self):
        df_unemp = pd.DataFrame(config.UNEMPLOYMENT_DATA)
        df_cpi = pd.DataFrame(config.CPI_DATA)
        self.df_macro = pd.merge(df_unemp, df_cpi, on='Year', how='inner')

        return self.df_macro
    
    
    def integrate_data(self):
        """ Integrate macroeconomic and weather data into the main dataset. """
        if self.is_realtime:
            dates = pd.DataFrame({'date': [self.fixed_date]})
        else:
            dates = pd.DataFrame({'date': pd.date_range(self.data['date'].min(), self.data['date'].max()).date})
        
        self.full_grid = pd.merge(dates, self.grids_df, how='cross')

        self.load_macro_data()
        self.fetch_load_weather_data(self.data['date'].min(), self.data['date'].max())
        
        # Add year column for merging with macro data
        self.full_grid['Year'] = pd.to_datetime(self.data['date']).dt.year
        self.full_grid = self.full_grid.merge(self.df_macro, on='Year', how='left')
        self.full_grid = self.full_grid.merge(self.weather_df, on=['date', 'lon_bin', 'lat_bin'], how='left')
        self.full_grid.merge(self.binned_event_data, on=['date', 'lon_bin', 'lat_bin'], how='left')
        
    def apply_feature_engineering(self):
        """Apply feature engineering to the data."""
        
        fe = FeatureEngineering(self.full_grid, config.FEATURES_LIST)
        fe.get_time_features()
        fe.preprocessing()
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
        else:
            # All data before the current date is used for training, current date for testing
            self.full_grid_engineered = self.full_grid_engineered.sort_values('date')
            
            self.train_data = self.full_grid_engineered[self.full_grid_engineered['date'] < self.fixed_date]
            self.test_data = self.full_grid_engineered[self.full_grid_engineered['date'] == self.fixed_date]

    def save_data(self, train_path, valid_path=None, test_path=None):
        """Save the split data sets to files."""
        self.train_data.to_csv(train_path, index=False)
        if valid_path and hasattr(self, 'valid_data'):
            self.valid_data.to_csv(valid_path, index=False)
        if test_path:
            self.test_data.to_csv(test_path, index=False)