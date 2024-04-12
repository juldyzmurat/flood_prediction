#%%
#weather data 
import pandas as pd
import os
from tqdm import tqdm

# List of columns

listcol = ['s2001_inv1', 's2001_inv2', 's2001_inv3', 's2004_inv1', 's2004_inv2', 's2004_inv3', 's2005_inv1', 's2005_inv2',
           's2005_inv3', 's2006_inv1', 's2006_inv2', 's2006_inv3', 's2007_inv1', 's2007_inv2', 's2008_inv1', 's2008_inv2',
           's2008_inv3', 's2009_inv1', 's2009_inv2', 's2009_inv3', 's2010_inv1', 's2010_inv2', 's2010_inv3', 's2014_inv2',
           's2014_inv3', 's2017_inv1', 's2017_inv2', 's2017_inv3', 's2020_inv1', 's2020_inv2', 's2020_inv3', 's2021_inv1',
           's2021_inv2', 's2021_inv3', 's2022_inv1', 's2022_inv2', 's2022_inv3', 's2024_inv1', 's2024_inv2', 's2024_inv3',
           's2025_inv1', 's2025_inv2', 's2025_inv3', 's2027_inv1', 's2027_inv3']
listcolgroup = sorted(set([x[1:5] for x in listcol]))

startdate = pd.to_datetime('2012-12-01 05:30:00 UTC')
enddate = pd.to_datetime('2015-10-08 04:00:00 UTC')
hourly_datapoints = pd.date_range(start=startdate, end=enddate, freq='H').values
print(hourly_datapoints)
print(len(hourly_datapoints))
#%%
allsites = []
sufforder = []

directory = '/home/zxu4/CSE_MSE_RXF131/staging/sdle/pv-multiscale/Sunsmart_cleaned/'

file_paths = []
for root, directories, files in os.walk(directory):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)

# Iterate over files in the directory with progress bar
for x in tqdm(listcolgroup):
    matching_file_path = next((path for path in file_paths if "weather" in os.path.basename(path) and x in os.path.basename(path)), None)
    df = pd.read_csv(matching_file_path, index_col=0)
    listtt = []
    suffixrep = x
    count = sum(x == item[1:5] for item in listcol)

    for index, row in df.iterrows():
        timerrr = str(int(row['Year'])) + '-' + str(int(row['Month'])).zfill(2) + "-" + str(int(row['Day'])).zfill(
            2) + " " + str(int(row['Hour'])).zfill(2) + ":" + str(int(row['Minute'])) + " UTC"
        timerrr = pd.to_datetime(timerrr)
        if startdate<=timerrr and timerrr<=enddate:
            listtt.append([row['pressure'], row['wind_speed'], row['dni'], row['ghi'], row['dhi'],row['temp_air']])
            listtt.append([row['pressure'], row['wind_speed'], row['dni'], row['ghi'], row['dhi'],row['temp_air']])
    for i in range(count):
        allsites.append(listtt)
    sufforder.append(os.path.basename(matching_file_path).split("-")[0])

# %%
#concat the weather columns into a column 
import pandas as pd
df = pd.read_csv('/home/zxu4/CSE_MSE_RXF131/staging/sdle/pv-multiscale/Sunsmart_cleaned/SS2007-weather.csv', index_col=0)
df
#%%
startdate = pd.to_datetime('2012-12-01 05:30:00 UTC')
enddate = pd.to_datetime('2015-10-08 04:00:00 UTC')
df['time'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M UTC')
df['time'] = pd.to_datetime(df['time'])
subset_df = df[(df['time'] >= startdate) & (df['time'] <= enddate)]
subset_df
# %%
hourly_datapoints = pd.date_range(start=startdate, end=enddate, freq='H', tz='UTC')
hourly_datapoints = pd.to_datetime(hourly_datapoints)
len(hourly_datapoints)

# %%
missing_datapoints = set(hourly_datapoints) - set(subset_df['time'])
# Convert the missing_datapoints back to a list or a pandas DatetimeIndex, if needed
missing_datapoints = pd.DatetimeIndex(list(missing_datapoints))
missing_datapoints
# %%
