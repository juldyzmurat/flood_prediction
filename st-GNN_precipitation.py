#%%
#create metadata for the neighbouring locations 

import pandas as pd 
import numpy as np 
import re

meta_locs = pd.DataFrame()

columns = ['Location',"Latitude","Longitude"]

def dms_to_dd(dms_str):
    # Parse DMS string using regular expression
    matches = re.findall(r'(\d+)°(\d+)′(\d+)″([NSEW])', dms_str)
    
    # Convert parsed values to DD
    dd_values = []
    for degrees, minutes, seconds, direction in matches:
        dd = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
        if direction == 'S' or direction == 'W':
            dd *= -1
        dd_values.append(dd)
    return dd_values


df_dict = {
    'Location': ['Amangeldi', 'Arkalyk', 'Tasty-Taldy', 'Ulutay'],
    'Latitude': [dms_to_dd("50°10′52″N 65°11′19″E")[0], dms_to_dd("50°14′53″N 66°55′40″E")[0], dms_to_dd("50°43′04″N 66°37′34″E")[0], dms_to_dd("48°39′20″N 67°00′14″E")[0]],
    'Longitude': [dms_to_dd("50°10′52″N 65°11′19″E")[1], dms_to_dd("50°14′53″N 66°55′40″E")[1],dms_to_dd("50°43′04″N 66°37′34″E")[1], dms_to_dd("48°39′20″N 67°00′14″E")[1]]
}


meta_locs  = pd.DataFrame(df_dict)

# %%
#generate an 