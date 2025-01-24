## import packages
from datetime import datetime
import pandas as pd


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
    
    visualization_df = pd.concat([df1,df2,df3], axis = 1)
    visualization_df.columns = [['yesterday_pred', 'yesterday_true', 'today_pred']]

    visualization_df.to_csv(path, index = False)

