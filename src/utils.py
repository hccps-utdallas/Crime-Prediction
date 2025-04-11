## import packages
from datetime import datetime
import pandas as pd
import config


def clean_up_loc(loc_string):
    """
    Process and parse the longitude and latitude from orinal dataset
    """
    try:
        res = loc_string.split('\n')[-1][1:-1].split(',')
        return float(res[1]), float(res[0])
    except:
        return None, None
    
def load_local_csv(path):
    return pd.read_csv(path)

def save_three_values(df1, df2, df3, path):

    visualization_df = df1.merge(
        df2, how = "left", left_on=config.LOC, right_on=config.LOC).merge(df3, how = "left", left_on=config.LOC, right_on=config.LOC)
    visualization_df.columns = ['lon_bin', 'lat_bin', 'yesterday_pred', 'yesterday_date', 'yesterday_true', 'today_date', 'today_pred']
    
    visualization_df['yesterday_diff'] = visualization_df['yesterday_true'] - visualization_df['yesterday_pred']
    visualization_df = visualization_df[['lon_bin', 'lat_bin', 'yesterday_date', 'yesterday_pred', 'yesterday_true', 'today_date', 'today_pred', 'yesterday_diff']]
    visualization_df.to_csv(path, index = False)

