import json
import pickle
import datetime
import pandas as pd
import numpy as np

import meteostat
from dateutil.relativedelta import relativedelta
from sodapy import Socrata
from openai import OpenAI

import config
import utils
from data import FeatureEngineering

def fetch_event_data_from_api(date):
    client = Socrata(config.DPD_API_URL, None)
    print(f"Successfully connected to API at {config.DPD_API_URL}")
    
    # Get previous day in Dallas time
    date_filter = f"""date1='{date} 00:00:00.0000000'"""

    results = client.get(config.DPD_DATASET_ID, where=date_filter, limit = 5000)
    results_df = pd.DataFrame.from_records(results)
    
    print("Data columns:", results_df.columns.tolist())
    print("Sample data:", results_df.head(1).to_dict())
    
    if results_df.empty:
        print("Warning: No data found for the date")
        return pd.DataFrame({
            'geocoded_column': [],
            'date1': [],
            'incidentnum': [],
            'Location1': []
        })
    
    return results_df

def fetch_load_weather_data(grids_df, start_date, end_date=None, timezone='US/Central', max_retries=5):
    weather_df = pd.DataFrame()
    
    if end_date is None:
        end_date = start_date
    start_time = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    end_time = datetime.datetime.combine(end_date, datetime.datetime.min.time())
    # datetime.datetime.strptime(end_date, '%Y-%m-%d')

    weather_data = []
    
    for longitude, latitude in grids_df[['lon_bin', 'lat_bin']].values:
        
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
        weather_df = pd.concat(weather_data, ignore_index=True)
    else:
        weather_df = pd.DataFrame()
    
    ## clean up format
    weather_df.rename(columns={'time': 'date'}, inplace=True)
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date

    return weather_df

def assign_events_to_grid2(data, grid_df, grid_size):
    # Assign each event to the nearest grid center
    data['longitude'] = data['geocoded_column'].apply(lambda x: float(x['longitude']) if isinstance(x, dict) and 'longitude' in x else None)
    data['latitude'] = data['geocoded_column'].apply(lambda x: float(x['latitude']) if isinstance(x, dict) and 'latitude' in x else None)

    data['lon_bin'] = ((data['longitude'] // grid_size) * grid_size + grid_size / 2).round(3)
    data['lat_bin'] = ((data['latitude'] // grid_size) * grid_size + grid_size / 2).round(3)
    
    # Merge events with grid centers to ensure all grid centers are included
    if 'Incident Number w/year' in data.columns:
        grouped_data = data.groupby(['date1', 'lon_bin', 'lat_bin'])['Incident Number w/year'].nunique().reset_index(name='unique_event_count')
    else:
        grouped_data = data.groupby(['date1', 'lon_bin', 'lat_bin'])['incidentnum'].nunique().reset_index(name='unique_event_count')
    
    
    grid_df = grid_df.merge(pd.DataFrame(grouped_data['date1'].unique(), columns=['date1']), how = 'cross')

    final_data_df = grid_df.merge(grouped_data, on=['date1', 'lon_bin', 'lat_bin'], how='left')
    final_data_df['unique_event_count'].fillna(0, inplace=True)

    return final_data_df

def generate_prediction(target_date, lon_bin, lat_bin, output_df, event_df, api_key, model_name):
    system_p = """You are Crime Predictor, a specialized AI assistant designed to forecast daily crime counts in specified regions. Your task is to predict the number of crimes that will occur in a given region for the next day, based on provided historical crime data and external factors.
        Follow Instructions provided below:
        1. Numerical Analysis:
        - Analyze the historical crime data, in the past seven days including daily crime counts, trends, seasonal patterns, and any relevant incident types.
        - Consider crime amount in past month same date, past year same date if available and helpful.
        2. Context Analysis:
        - Evaluate external factors such as weather conditions, socioeconomic indicators, holidays, or any other variables provided.
        - Summarize the event description in past seven days and use it as context if helps for your prediction.
        3. Generate a numerical prediction representing the expected crime count for the next day in the specified region.

        Only give the prediction in integer format do not include any reasoning or explanation in output.
    """

    user_p = """
    Here is the crime number at the same region in past seven days: 
    {amount_input}
    Here are the event-level crime details at the same region in past seven days in json format:
    {event_input_json}
    Here are some extra external factors at the same region in json format:
    {factors_input_json}
    Output predicted crime count in integer only without any other reasoning or explanation:
    """

    # Calculate the date range for the past 7 days
    target_date_obj = pd.to_datetime(target_date)
    start_date = (target_date_obj - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    end_date = (target_date_obj - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    date_range = (start_date, end_date)

    # Filter event details
    event_details = json.dumps(event_df[
        (event_df.lon_bin == lon_bin) & (event_df.lat_bin == lat_bin) & 
        (event_df.date1 >= date_range[0]) & (event_df.date1 <= date_range[1])
    ][['Offense Status', 'Modus Operandi (MO)', 'Family Offense', 'Hate Crime Description', 'Weapon Used', 'Gang Related Offense',
       'Drug Related Istevencident', 'NIBRS Crime Category', 'NIBRS Crime Against']].to_dict(orient='records'))

    # Filter and prepare test info
    test_info = output_df[
        (output_df.date1 >= date_range[0]) & (output_df.date1 <= date_range[1]) & 
        (output_df.lon_bin == lon_bin) & (output_df.lat_bin == lat_bin)
    ].set_index('date1')[
        ['day_of_year_sin', 'day_of_year_cos', 'week_of_year_sin', 'week_of_year_cos', 
         'Avg Temp', 'Min Temp', 'Max Temp', 'Precipitation', 'Snowfall', 'Unemployment_Rate(%)', 'CPI', 'Distance (km)']
    ].to_dict(orient='index')


    test_info["last month same date"] = output_df[
        (output_df.date1 == (target_date_obj - pd.DateOffset(months=1)).strftime('%Y-%m-%d')) & 
        (output_df.lon_bin == lon_bin) & (output_df.lat_bin == lat_bin)
    ]["unique_event_count"].values[0]
    test_info["last year same date"] = output_df[
        (output_df.date1 == (target_date_obj - pd.DateOffset(years=1)).strftime('%Y-%m-%d')) & 
        (output_df.lon_bin == lon_bin) & (output_df.lat_bin == lat_bin)
    ]["unique_event_count"].values[0]

    test_info_json = json.dumps(test_info)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Generate prediction
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_p},
            {
                "role": "user",
                "content": user_p.format(
                    amount_input=list(output_df[
                        (output_df.date1 >= date_range[0]) & (output_df.date1 <= date_range[1]) & 
                        (output_df.lon_bin == lon_bin) & (output_df.lat_bin == lat_bin)
                    ]["unique_event_count"].values),
                    event_input_json=event_details,
                    factors_input_json=test_info_json,
                )
            }
        ]
    )

    return int(completion.choices[0].message.content)


## run
if __name__ == "__main__":
    import os
    import json
    import time
    import pickle
    import datetime
    import pandas as pd
    import numpy as np
    
    import meteostat
    from sodapy import Socrata
    from openai import OpenAI

    import config
    import utils

    ## ENV var
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("API_KEY")
    
    ## Global variables
    dt = datetime.date.today()
    print(f"Current date: {dt}")

    detail_list = ['date1','lon_bin', 'lat_bin', 'Offense Status', 'Modus Operandi (MO)', 'Family Offense', 'Hate Crime Description', 
                   'Weapon Used', 'Gang Related Offense', 'Drug Related Istevencident', 'NIBRS Crime Category', 'NIBRS Crime Against']
    
    ## Check key availability
    if not api_key:
        raise ValueError("API_KEY is not set in the environment variables")
    
    # Use the API key for your API call or other purposes
    print("API key loaded successfully!")

    ## Load pre-defined grid
    with open('./data/preload_grid/grids_df.pkl', 'rb') as f:
        grids_df = pickle.load(f)
    
    ## Load econ data
    macro_df = pd.read_csv("./data/local_files/macro_data.csv")
    
    ## Get event data from API
    #### fetch past 7 days data
    event_df = pd.DataFrame()
    date_range = pd.date_range(start = (dt - datetime.timedelta(days=7)).strftime('%Y-%m-%d'), end = (dt - datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    for date in date_range:
        event_df = pd.concat([event_df, fetch_event_data_from_api(date=date.strftime('%Y-%m-%d'))], axis=0)
        time.sleep(3)
    
    
    #### fetch past 1 year data
    one_year_ago = dt - relativedelta(years=1)
    event_df = pd.concat([event_df, fetch_event_data_from_api(date=one_year_ago.strftime('%Y-%m-%d'))], axis=0)
    ## Fetch past 1 month data
    one_month_ago = dt - relativedelta(months=1)
    event_df = pd.concat([event_df, fetch_event_data_from_api(date=one_month_ago.strftime('%Y-%m-%d'))], axis=0)
    
    #### Clean up
    event_df.reset_index(drop=True, inplace=True)
    
    ## Get weather data
    weather_df = fetch_load_weather_data(
        grids_df, start_date=dt - datetime.timedelta(days=7), end_date=dt, timezone='US/Central', max_retries=5)
    
    event_df['date1'] = event_df['date1'].apply(lambda x: x.split(' ')[0])
    agg_event_data_df = assign_events_to_grid2(event_df, grids_df, 0.05)

    weather_df['date1'] = weather_df.date.astype(str)

    pre_fe_df1 = agg_event_data_df.merge(weather_df[['lon_bin', 'lat_bin', 'date1', 'tavg',
       'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun']], left_on=['date1', 'lon_bin', 'lat_bin'], right_on=['date1', 'lon_bin', 'lat_bin'], how='left')
    
    pre_fe_df1['Year-Month'] = pre_fe_df1.date1.apply(lambda x: '-'.join(x.split('-')[:2]))
    pre_fe_df1 = pre_fe_df1.merge(macro_df, on=['Year-Month'], how='left')
    pre_fe_df1['date'] = pd.to_datetime(pre_fe_df1['date1']).dt.date
    fe = FeatureEngineering(pre_fe_df1)
    fe.get_time_features()
    output_df = fe.get_dataframe() ## training_df
    
    ## Clean up
    output_df = pd.concat([pre_fe_df1[['date1']], output_df], axis=1)
    output_df.rename(columns={'tavg': 'Avg Temp', 'tmin': 'Min Temp', 'tmax': 'Max Temp', 'prcp': 'Precipitation','snow': 'Snowfall', 'wdir': 'Wind Direction', 'wspd': 'Wind Speed', 'wpgt': 'Wind Gust', 'pres': 'Atmospheric Pressure', 'tsun': 'Total Sunshine Duration'}, inplace=True)
    event_df.rename(columns={'status': 'Offense Status', 'mo':'Modus Operandi (MO)', 'family':'Family Offense','hatecrimedescriptn': 'Hate Crime Description', 'weaponused': 'Weapon Used', 'gang': 'Gang Related Offense', 'drug': 'Drug Related Istevencident', 'nibrs_crime_category': 'NIBRS Crime Category', 'nibrs_crimeagainst': 'NIBRS Crime Against'}, inplace=True)

    
    ## Obtain the predcition
    predictions = []
    for _, row in grids_df.iterrows():
        one_step_prediction = generate_prediction(dt.strftime('%Y-%m-%d'), row.lon_bin, row.lat_bin, output_df = output_df, 
                                                    event_df = event_df[detail_list], api_key = api_key, model_name = model_name)
        predictions.append(one_step_prediction)
    
        if _ % 10 == 0:
            print(_)
    
    ## Save the results
    df1 = pd.read_csv(config.PREDICTION_DATA_PATH)  ## yesterday's prediction
    df1.rename(columns={'today_date': 'yesterday_date'}, inplace=True)
    df2 = output_df[output_df.date1 == (dt - datetime.timedelta(days=1)).strftime('%Y-%m-%d')][['lon_bin', 'lat_bin', 'unique_event_count']] ## yesterday's ground truth
    df3 = pd.concat([grids_df[config.LOC], pd.DataFrame([dt.strftime('%Y-%m-%d')]*grids_df.shape[0], columns = ['today_date']), pd.DataFrame(predictions, columns=['pred'])], axis = 1) ## today's prediction
    df3.to_csv(config.PREDICTION_DATA_PATH, index=False)
    print(df1.shape, df2.shape, df3.shape)
    utils.save_three_values(df1, df2, df3, config.VISUALIZATION_DATA_PATH)
    # print(f"{model}, RMSE: {np.mean(np.sqrt(np.mean(np.array(list(res.values())) ** 2, axis=0)))}")