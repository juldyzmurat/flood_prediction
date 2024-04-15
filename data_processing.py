#%%
import os 
import pandas as pd
#turn .xlsx files in the drectory into csv
directory = "/Users/zhuldyzualikhankyzy/Documents/GitHub/flood_prediciton/data/ekidin"

# Iterate over the files in the directory
for file in os.listdir(directory):
    # Read the Excel file into a DataFrame
    
    fileread = pd.read_excel(os.path.join(directory, file))
    newname = (os.path.join(directory, file)).replace(".xlsx",".csv")
    fileread.to_csv(newname, index=False, header=True)

#%%
#create a file with all meteo features 

directory = "/Users/zhuldyzualikhankyzy/Documents/GitHub/flood_prediciton/data/ekidin"

for  file in os.listdir(directory):
    if file.endswith(".csv"):
        
#%%
#create a file with hydro and ice features 

df_hydro_ice = pd.DataFrame()

su_deng = pd.read_csv("./data/ekidin/ekidin_su_dengei.csv")
su_otim = pd.read_csv("./data/ekidin/ekidin_su_otimi.csv")

su_deng=  su_deng.merge(su_otim, on = "Күні",how = "inner")
su_deng = su_deng.drop(columns=su_deng.columns[4])
su_deng.columns = ["station_code","date","water_level","symbol","discharge"]
su_deng.to_csv("./data/ekidin_water_level_discharge.csv", index =False, header = True)

# %%
#combine meteo and water data 
import pandas as pd 
import numpy as np 

# su_deng = pd.read_csv("./data/ekidin/ekidin_su_dengei.csv")
# su_otim = pd.read_csv("./data/ekidin/ekidin_su_otimi.csv")

# su_deng=  su_deng.merge(su_otim, on = "Күні",how = "inner")
# su_deng = su_deng.drop(columns=su_deng.columns[4])
# su_deng.columns = ["station_code","date","water_level","symbol","discharge"]
# su_deng.to_csv("./data/ekidin_water_level_discharge.csv", index =False, header = True)

directory = "/Users/zhuldyzualikhankyzy/Documents/GitHub/flood_prediciton/data/ekidin"
df_hydro_ice = pd.DataFrame()
for  file in os.listdir(directory):
    
    if file.endswith(".csv") and not file.startswith("ekidin_su"):
        if  len(df_hydro_ice)==0:
            df_hydro_ice = pd.read_csv(os.path.join(directory,file),skiprows=2)
            print(df_hydro_ice)
        else: 
            lefter = pd.read_csv(os.path.join(directory,file),skiprows=2)
            lefter.drop("Станция",axis=1,inplace=True)
            varname = file[file.find("_")+1:file.find(".")]
            columns = lefter.columns.to_list()
            for i, col in enumerate(columns):
                if col != 'Дата':
                    columns[i] = varname + col
            print(columns)
            lefter.columns = columns
            print(lefter.columns)
            df_hydro_ice = df_hydro_ice.merge(lefter, on = "Дата",how = "inner")
        
df_hydro_ice


# %%
#check for NaN values
df_hydro_ice.isnull().sum()

# %%
#plot to see the missing valus for snezhnСт.покр. , snezhnВысота,см , osadkiСумма
import matplotlib.pyplot as plt

plt.plot(df_hydro_ice['Дата'], df_hydro_ice['snezhnСт.покр.'])
plt.xlabel('Дата')
plt.ylabel('snezhnСт.покр.')
plt.title('snezhnСт.покр. vs Date')
plt.show()

#%%
#check if there are values for 0 for snow coverage and height values
df_hydro_ice[df_hydro_ice['snezhnСт.покр.']==0]

#found out that there are no zero values. Hence, the missin dates stand for summer (fall/spring) times with no snow coverage
#hence, we can fill the missing values with 0 not only for snezhnСт.покр. but also for  snezhnВысота,см

df_hydro_ice['snezhnСт.покр.'].fillna(0,inplace=True)
df_hydro_ice['snezhnВысота,см'].fillna(0,inplace=True)

#%%
#check if there are 0 values in the precipitation column
df_hydro_ice[df_hydro_ice['osadkiСумма']==0]

#temporarily replace the nan values is -1 to see if they are continuous 
df_hydro_ice['osadkiСумма'] = df_hydro_ice['osadkiСумма'].replace(-1,np.nan)

#I wanted to check if the missing values only occur in summer for different years 
df_summer_2017 = df_hydro_ice[(df_hydro_ice['Дата']<'2021-09-01') & (df_hydro_ice['Дата']>'2021-06-01')]

# Plotting
plt.figure(figsize=(10, 6))  # Set figure size
plt.scatter(df_summer_2017['Дата'], df_summer_2017['osadkiСумма'], marker='o', linestyle='-', color='blue')

plt.title('Summer 2017 Osadki Summa')
plt.xlabel('Date')
plt.ylabel('Osadki Сумма')

#after some visual analysis, it does not seem like all the missing values correspond to no rain days
#which makes sense because there already were values with the value of 0 (which would be used if NaN values were supposed to be 0)

#since I am bored, I want to make my assumptions and approximations as reasonable as possible
#hence I will use st-GNN  to use the the precipitation from other neighboring geographies to fill in the missing values 
#it will be continued in the file "st-GNN_precipitation.py"

# %%
#the rest of the missing values are realted to pressue, which does not tend to vary much.
#hence, we fill the missing value with the mean value 
df_hydro_ice['pressureНа ур.станц.'].fillna(np.mean(df_hydro_ice['pressureНа ур.станц.']),inplace=True)
df_hydro_ice['pressureНа ур.моря'].fillna(np.mean(df_hydro_ice['pressureНа ур.моря']),inplace=True)
# %%
#the previous code caused errors due to the attribute pressureНа ур.моря having "-" value 
# we replace all the rows with the value "-" with nan value 
df_hydro_ice['pressureНа ур.моря'].replace('-', np.nan, inplace=True)
#turn all the values into float since the presence of "-" values turned the series into string type 
df_hydro_ice['pressureНа ур.моря'] = df_hydro_ice['pressureНа ур.моря'].astype(float)
#then we run the preivous code to fill in the gaps 
df_hydro_ice['pressureНа ур.моря'].fillna(np.mean(df_hydro_ice['pressureНа ур.моря']),inplace=True)


df_hydro_ice['pressureНа ур.станц.'].replace('-', np.nan, inplace=True)
#turn all the values into float since the presence of "-" values turned the series into string type 
df_hydro_ice['pressureНа ур.станц.'] = df_hydro_ice['pressureНа ур.станц.'].astype(float)
#then we run the preivous code to fill in the gaps 
df_hydro_ice['pressureНа ур.станц.'].fillna(np.mean(df_hydro_ice['pressureНа ур.станц.']),inplace=True)

# %%
#finally we check if there are any other misisng values 
df_hydro_ice.isnull().sum()
#we can see the precipiations is the only column with missing values that we will fill in later with st-GNN
# %%
#save the df with all the variables as a new csv file 
df_hydro_ice.to_csv("./data/ekidin_hydro_ice.csv", index =False, header = True)

# %%
#plot Ekidin percipation data 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv("./data/ekidin_hydro_ice.csv")
df[df['osadkiСумма'] == " "]

plt.figure(figsize=(10, 6))  # Set figure size
plt.plot(df['Дата'], np.where(df['osadkiСумма'].notnull(), 1, np.nan), '|', color='blue')

plt.title('Ekidin Precipitation Data')
plt.xlabel('Date')
plt.ylabel('Presence of Precipitation')

plt.show()

#%%
#we see that the missing data does not correpsond to only summer time 
#let us see if there are values of 0 
df[df['osadkiСумма'] == 0]
# %%
#turns out there are 0 values, hence, the missin values are not 0 
# Find the longest consecutive missing data time frame for osadkiСумма
missing_data = df[df['osadkiСумма'].isnull()]
missing_data['Дата'] = pd.to_datetime(missing_data['Дата'])
missing_data['diff'] = (missing_data['Дата'] - missing_data['Дата'].shift(1)).dt.days
missing_periods = missing_data.groupby((missing_data['diff'] != 1).cumsum()).size()

plt.hist(missing_periods, bins=10)

plt.xlabel('Length of Missing Periods')
plt.ylabel('Frequency')
plt.title('Distribution of Missing Period Lengths')
plt.show()


# %%
# Create a DataFrame with the histogram values
hist_values = pd.DataFrame({'Length of Missing Periods': missing_periods.values, 'Frequency': missing_periods.index})
hist_values = hist_values.reset_index(drop=True)
hist_values
# %%
#let us see if there is going to be a lot of values missing if we use other stations 
# Combine all files from st-gnn-other-locs into one DataFrame
import pandas as pd
import os

directory = "/Users/zhuldyzualikhankyzy/Documents/GitHub/flood_prediciton/data/st-gnn-other-locs"
df_combined = pd.DataFrame()

df_combined["Дата"] = pd.date_range(start='2007-01-01', end='2024-01-31', freq='D')
print(df_combined["Дата"])

#%%
# Loop through each file in the directory
for file in os.listdir(directory):
    if file.endswith(".xlsx"):  # Check if the file is an Excel file
        file_path = os.path.join(directory, file)
        df = pd.read_excel(file_path, skiprows=2)  # Read the file
        # Rename the 'Сумма' column to include the station name for uniqueness
        station_name = df["Станция"].iloc[0]  # Assuming the station name is constant per file
        df = df[["Дата", "Сумма"]]
        df.columns = ["Дата", station_name]
        # Merge or initialize the combined DataFrame
        df_combined = df_combined.merge(df, on="Дата", how="left")

# Check the combined DataFrame
df_combined

#%%
#add the values of ekidin 
df_ekidin = pd.read_csv("/Users/zhuldyzualikhankyzy/Documents/GitHub/flood_prediciton/data/ekidin/ekidin_osadki.csv",skiprows=2)
# Rename the 'Сумма' column to include the station name for uniqueness
station_name = df_ekidin["Станция"].iloc[0]  # Assuming the station name is constant per file
df_ekidin = df_ekidin[["Дата", "Сумма"]]
df_ekidin.columns = ["Дата", station_name]
df_ekidin['Дата'] = pd.to_datetime(df_ekidin['Дата'], errors='coerce')
# Convert 'Дата' to a string in the format 'YYYY-MM-DD', handling any NaT values which turn into NaN
df_ekidin['Дата'] = df_ekidin['Дата'].astype(str)

df_ekidin['Дата'] = pd.to_datetime(df_ekidin['Дата'], errors='coerce')
df_combined['Дата'] = pd.to_datetime(df_combined['Дата'], errors='coerce')

df_combined = df_combined.merge(df_ekidin, on="Дата", how="left")
df_combined.to_csv("./data/5locs_percip.csv", index =False, header = True)

#%%
#impute the values of one day missingness with three day average 
import pandas as pd
import numpy as np

def impute_small_gaps(df, max_gap_length=3):
    # Process each column separately
    for column in df.columns:
        # Skip the 'Дата' column or any non-numeric column
        if df[column].dtype == 'object' or column == 'Дата':
            continue
        
        # Detect consecutive NaNs and their indices
        nans = df[column].isna()
        fill_value = None
        
        for i in range(len(df)):
            # Start of NaN sequence
            if nans.iloc[i] and (i == 0 or not nans.iloc[i - 1]):
                start = i
            
            # End of NaN sequence
            if nans.iloc[i] and (i == len(df) - 1 or not nans.iloc[i + 1]):
                end = i
                if (end - start) <= max_gap_length:
                    # Indices for two days before and after the gap
                    before_indices = [idx for idx in range(start-2, start) if idx >= 0]
                    after_indices = [idx for idx in range(end+1, end+3) if idx < len(df)]
                    relevant_indices = before_indices + after_indices
                    
                    # Calculate mean excluding NaNs
                    relevant_values = df.iloc[relevant_indices][column].dropna()
                    if not relevant_values.empty:
                        fill_value = relevant_values.mean()
                    
                    # Fill the gap if a mean was calculable
                    if fill_value is not None:
                        df.loc[start:end+1, column] = df.loc[start:end+1, column].fillna(fill_value)
    
    return df

imputed_df = impute_small_gaps(df_combined, 3)
print(imputed_df)
#%%
#we want to find the longest connsequtive time frame without any missing data across all 5 locations 
def longest_non_nan_window(df, columns):
    max_length = 0
    current_length = 0
    start_date = None
    end_date = None
    longest_window = None

    # Iterate over the DataFrame rows
    for idx, row in df.iterrows():
        # Check if all columns in this row are not NaN
        if all([pd.notna(row[col]) for col in columns]):
            current_length += 1
            if current_length == 1:  # Mark the start of a new non-NaN window
                start_date = row['Дата']
            end_date = row['Дата']  # Update the end date of the current window

            # Update the longest window found so far
            if current_length > max_length:
                max_length = current_length
                longest_window = (start_date, end_date)
        else:
            current_length = 0  # Reset current window length

    return longest_window, max_length

columns_to_check = ['Екидин']

# Get the longest window
longest_window, max_length = longest_non_nan_window(imputed_df, columns_to_check)
# %%
##in this case, we unfortunaely cannot use st-GNN due to a very limited number of complete data
##we also cannot use oter temporal imputaiton models 
##to address this issue, I will use DNN that does not require contrinous data

##first I need to generate additonal variables to help DNN to account for seasonality 
df_combined['Дата'] = pd.to_datetime(df_combined['Дата'])
df_combined['Day'] = df_combined['Дата'].dt.day
df_combined['Month'] = df_combined['Дата'].dt.month
df_combined['Year'] = df_combined['Дата'].dt.year   
df_combined

#%%
df_combined = df_combined.dropna()
df_combined

#excellent, we have 867 complete rows to train our model
#it turns out it is a bad idea because we still do not have enough values for prediciton 

#%%
#let us instead see what weather values we can use to predict precipation 

#%%
#we need to drop the date column 
df_combined = df_combined.drop(columns = ['Дата'])


#%%
#import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.optimizers import Adam 
from joblib import dump,load
#%%
df = df_combined
##build the model architecture 
X = df[['Амангельды', 'Улытау', 'Тасты-Талды', 'Аркалык', 'Day', 'Month', 'Year']]
Y = df['Екидин']

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### 2. **Model Building**

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer and 1st hidden layer
model.add(Dense(32, activation='relu'))  # 2nd hidden layer
model.add(Dense(1, activation='linear'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#%%
#train the model 
model.fit(X_train, Y_train, epochs=100, batch_size=60, verbose=1)

#%%
#for some reason, running this command in the terminal did not work, but running it here resolved the error 
!pip3 install --user pandas openpyxl

# %%
#let us open the df with all meteo and hydro data 
df_hydro_ice = pd.read_csv("/Users/zhuldyzualikhankyzy/Documents/GitHub/flood_prediciton/data/ekidin_hydro_ice.csv")
df_hydro_ice.columns = ['station', 'date', 'average', 'min', 'soil_tempaverage', 'soil_tempmax', 'soil_tempmin', 'air_tempaverage', 'air_tempmax', 'air_tempmin', 'pressureat station level', 'pressureat sea level', 'dew_pointmin', 'snow_cover_depth', 'snow_height_cm', 'def_nasaverage', 'def_nasmax', 'precipitationtotal', 'windaverage', 'windmax from 8 terms', 'windabs max', 'part_pressaverage']
df_hydro_ice['date'] = pd.to_datetime(df_hydro_ice['date'], errors='coerce')
df_hydro_ice['date'] = df_hydro_ice['date'].astype(str)
df_water = pd.read_csv("/Users/zhuldyzualikhankyzy/Documents/GitHub/flood_prediciton/data/ekidin_water_level_discharge.csv")
df_total = df_hydro_ice.merge(df_water, on = "date", how = "inner")
df_total['date'] = pd.to_datetime(df_total['date'], errors='coerce')
df_total['day'] = df_total['date'].dt.day
df_total['month'] = df_total['date'].dt.month
df_total['year'] = df_total['date'].dt.year
df_total

#%% 
#replace the bad values with nan 
df_total['discharge'] = pd.to_numeric(df_total['discharge'], errors='coerce')
# %%
nan_count = df_total['discharge'].isnull().sum()
print("Number of NaN values:", nan_count)

# %%
zero_count = (df_total['discharge'] == 0).sum()
print("Number of 0 values:", zero_count)

#%%
#the previous observation confrims the general proceudre for water discharge records 
# "нб" stands for no or 0 discharge
df_total['discharge'].fillna(0, inplace=True)
df_total

#%%
#drop the symbol column 
df_total = df_total.drop(columns = ['symbol'])

#%%
#save the df with all the variables as a new csv file
df_total.to_csv("/Users/zhuldyzualikhankyzy/Documents/GitHub/flood_prediciton/data/ekidin_total.csv", index =False, header = True)
# %%
#load the dataframe
df = pd.read_csv("/Users/zhuldyzualikhankyzy/Documents/GitHub/flood_prediciton/data/ekidin_total.csv")
df = df.drop(["station","date","symbol"],axis=1)
#clean from white spaces 
df.replace('\*', '', regex=True, inplace=True)
df.replace(' ', '', regex=True, inplace=True)
df.replace({'-':np.nan},regex=False, inplace=True)
df = df.dropna()
df
# %%
#turn into numerical values 
df = df.apply(pd.to_numeric)
df
# %%
X = df.copy()
X = X.drop("precipitationtotal",axis=1)
Y = df['precipitationtotal']


# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### 2. **Model Building**

# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer and 1st hidden layer
model.add(Dense(32, activation='relu'))  # 2nd hidden layer
model.add(Dense(1, activation='linear'))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=100, batch_size=20, verbose=1)

# %%
#test the model 
y_pred = model.predict(X_test)
y_pred[y_pred<0]=0 
y_pred
# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot([x for x in range(len(Y_test))],Y_test.values,label = "ground",marker ='o')
plt.plot([x for x in range(len(Y_test))],y_pred,label = "pred",marker ='x')
plt.legend()
plt.grid(True)
plt.show()

# %%
#the predoicitons do not seem to be following the pattern 
#let's see what variables have more correlation with the percipitation column 
#first, let's drop the station_name 
import seaborn as sns 
df = pd.read_csv("/Users/zhuldyzualikhankyzy/Documents/GitHub/flood_prediciton/data/ekidin_total.csv")
df = df.drop(["station","date","symbol","station_code"],axis=1)
#clean from white spaces 
df.replace('\*', '', regex=True, inplace=True)
df.replace(' ', '', regex=True, inplace=True)
df.replace({'-':np.nan},regex=False, inplace=True)
df.to_csv("final_features.csv")
corr_matrix = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=False, cmap = "coolwarm",fmt=".2f")
plt.show()
# %%
filtered_corr_matrix = corr_matrix[(corr_matrix['precipitationtotal'] > 0.25) | (corr_matrix['precipitationtotal'] < -0.25)]
filtered_corr_matrix
#%%
#now we see the columns that have greater than 0.25 correlation or less that -0.25 correlation 
#let's retrain the model 
# correlated_percip_df = df.copy()
# correlated_percip_df = correlated_percip_df.dropna()
# correlated_percip_df = correlated_percip_df[['soil_tempmin','air_tempmin','pressureat station level','pressureat sea level','dew_pointmin','precipitationtotal','part_pressaverage','day','month','year']]

# X = correlated_percip_df[['soil_tempmin','air_tempmin','pressureat station level','pressureat sea level','dew_pointmin','part_pressaverage','day','month','year']]
# Y = correlated_percip_df['precipitationtotal']

df = df.dropna()
X = df.copy()
X = X.drop("precipitationtotal",axis=1)
Y = df['precipitationtotal']
# Y = np.log1p(Y)


# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
dump(scaler,'scaler.joblib')
X_test = scaler.transform(X_test)
# Y_train = Y_train.values.reshape(-1,1)
# Y_train = scaler.fit_transform(Y_train)
# Y_test = Y_test.values.reshape(-1,1)
# Y_test = scaler.transform(Y_test)

### 2. **Model Building**
# Build the neural network model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer and 1st hidden layer
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))  # 2nd hidden layer
model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))  # Output layer

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=200, batch_size=30, verbose=1)

#test the model 
y_pred = model.predict(X_test)
y_pred
# %%
#turns out that the removing columns except for the "station","date","symbol","station_code" does not affect the model's learnings 
#let's see the predictions 
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot([x for x in range(len(Y_test))],Y_test,label = "ground",marker ='o')
plt.plot([x for x in range(len(Y_test))],y_pred,label = "pred",marker ='x')
plt.legend()
plt.grid(True)
plt.show()

# Assuming `transformed_values` is your data transformed with log1p

# %%
#now we want to fill in the missing percipiation data using this DNN 
#first let's check how many nan values there are 
df = pd.read_csv("/Users/zhuldyzualikhankyzy/Documents/GitHub/flood_prediciton/final_features.csv",index_col=0)
#"-" in snow height means no snow because all the following values after are 0 and it also coincides with the flood event caused by rapid melt of snow 
df['snow_height_cm'].fillna(0,inplace=True)
df.isnull().sum()


#find the rows with NaN percip 
nan_rows = df['precipitationtotal'].isna()
X_miss = df.loc[nan_rows,['average', 'min', 'soil_tempaverage', 'soil_tempmax', 'soil_tempmin',
       'air_tempaverage', 'air_tempmax', 'air_tempmin',
       'pressureat station level', 'pressureat sea level', 'dew_pointmin',
       'snow_cover_depth', 'snow_height_cm', 'def_nasaverage', 'def_nasmax','windaverage', 'windmax from 8 terms',
       'windabs max', 'part_pressaverage', 'water_level', 'discharge', 'day',
       'month', 'year']]
scaler = load('scaler.joblib')
X_miss = scaler.fit_transform(X_miss)
filled_miss = model.predict(X_miss)
df.loc[nan_rows,'precipitationtotal'] = filled_miss

df.isnull().sum()

#save is csv 
df.to_csv("final_features_imputed.csv",index =False, header = True)


# %%
#component 1 
#regression model to predict the water levels
#load data 
import numpy as np 
import pandas as pd 

df = pd.read_csv("final_features_imputed.csv")
#we need two columns discharge and water level 
df['date'] = df['year'].astype(str) + "-" + df['month'].astype(str) + "-" + df['day'].astype(str)
df['date'] = pd.to_datetime(df['date'])
df = df[['date','discharge','water_level','month']]

df = df.set_index('date').sort_index()

df['water_level_1day'] = df['water_level'].shift()
df['water_level_2day'] = df['water_level'].shift(2)
df['water_level_3day'] = df['water_level'].shift(3)
df['water_level_4day'] = df['water_level'].shift(4)
df['water_level_5day'] = df['water_level'].shift(5)
df['water_level_6day'] = df['water_level'].shift(6)
df['water_level_7day'] = df['water_level'].shift(7)

df = df.dropna()
df

# %%
from sklearn.model_selection import train_test_split
df_train = df[:int(len(df)*0.8)]
df_test = df[int(len(df)*0.8):]

#%%
##!pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o --user
# %%
#now we begin using h2o 
import h2o
from h2o.automl import H2OAutoML
h2o.init()

h2o_frame = h2o.H2OFrame(df_train)
x = h2o_frame.columns
y = 'water_level'
x.remove(y)
# %%
h2o_automl = H2OAutoML(sort_metric='mse', max_runtime_secs=5*60, seed=42)
h2o_automl.train(x=x, y=y, training_frame=h2o_frame)
# %%
#retrive the best model and save it
best_model = h2o_automl.leader
model_path = h2o.save_model(model=best_model, path="./models/time_series_forecast/", force=True)

#%%
# #component 2
#multi-output regression model to predict the water levels based on meteo data 
#generate new features 
#found an error in naming columns, let's fix it
df = pd.read_csv("final_features_imputed.csv")
df.columns = ['humid_average', 'humid_min', 'soil_tempaverage', 'soil_tempmax', 'soil_tempmin',
       'air_tempaverage', 'air_tempmax', 'air_tempmin',
       'pressureat station level', 'pressureat sea level', 'dew_pointmin',
       'snow_cover_depth', 'snow_height_cm', 'def_nasaverage', 'def_nasmax',
       'precipitationtotal', 'windaverage', 'windmax from 8 terms',
       'windabs max', 'part_pressaverage', 'water_level', 'discharge', 'day',
       'month', 'year']
df.to_csv("final_features_imputed.csv",index =False, header = True)

#%%
##now let's aggregate the features 
#function for aggregation 
def days_agg(dataframe: pd.DataFrame, column_name: str, agg_func: str, days: int):

    d = {
        'mean': np.mean,
        'sum': np.sum,
        'std': np.std,
        'amplitude': lambda x: np.max(x) - np.min(x),
        'first_and_last': lambda x: x[0] - x[-1],
    }

    agg_f = d[agg_func]
    values = np.array(dataframe[column_name])
    result = []

    for i in range(0, len(values)):
        i_ = i - days
        if i_ < 0:
            i_ = 0

        result.append(agg_f(values[i_:i + 1]))

    return result

#funciton that calls aggregation at once for all columns 
def feature_aggregation(dataframe: pd.DataFrame):
    columns = dataframe.columns
    columns_drop = []

    if 'discharge' in columns:
        # Расход - среднее за 7 суток
        dataframe['discharge_mean7'] = days_agg(dataframe, 'discharge', 'mean', 7)
    if 'humid_average' in columns:
        # Целевая переменная - среднее и амплитуда за 7 и 3 суток
        dataframe['humid_average_ampl7'] = days_agg(dataframe, 'humid_average', 'amplitude', 7)
        dataframe['humid_average_mean3'] = days_agg(dataframe, 'humid_average', 'mean', 4)
    if 'snow_cover_depth' in columns:
        # Доля снежного покрова - амплитуда за 30 суток
        dataframe['snow_cover_depth_ampl30'] = days_agg(dataframe, 'snow_cover_depth', 'amplitude', 30)
    if 'snow_height_cm' in columns:
        # Высота снежного покрова
        dataframe['snow_height_cm_mean15'] = days_agg(dataframe, 'snow_height_cm', 'mean', 15)
        dataframe['snow_height_cm_diff30'] = days_agg(dataframe, 'snow_height_cm', 'first_and_last', 30)
    if 'precipitationtotal' in columns:
        # Сумма осадков за 20 суток
        dataframe['precipitation_sum20'] = days_agg(dataframe, 'precipitationtotal', 'sum', 20)
    

    dataframe.drop(columns_drop, axis=1, inplace=True)

    return dataframe

# %%
df = pd.read_csv("final_features_imputed.csv")
df = feature_aggregation(df)

#create lagging variables for the 7 day prediction 
for i in range(1, 8):
    df[f'target_day_{i}'] = df['water_level'].shift(-i)
    
#drop the first 20 days due to aggregation 
df = df[20:]
#drop the last 7 days because they don't have targets 
df = df[:-7]
# %%
#import the neccessary varaibles 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# %%
X = df.drop(['target_day_1', 'target_day_2', 'target_day_3', 'target_day_4', 'target_day_5', 'target_day_6', 'target_day_7'], axis=1)
Y = df[['target_day_1', 'target_day_2', 'target_day_3', 'target_day_4', 'target_day_5', 'target_day_6', 'target_day_7']]

X_train = X[:int(len(X)*0.8)]
X_test = X[int(len(X)*0.8):]
Y_train = Y[:int(len(Y)*0.8)]
Y_test = Y[int(len(Y)*0.8):]
# Create and train the multi-target regression model (ElasticNet)
multioutput_model = MultiOutputRegressor(
    ElasticNet(alpha=0.5, l1_ratio=0.5), n_jobs=7)
multioutput_model.fit(X_train, Y_train)

# Create and train the decision tree regressor model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, Y_train)

# Create and train the random forest regressor model
forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
forest_model.fit(X_train, Y_train)
# %%
#let's plot the prections 
# Make predictions
multioutput_pred = multioutput_model.predict(X_test)
tree_pred = tree_model.predict(X_test)
forest_pred = forest_model.predict(X_test)

# Calculate performance metrics for multioutput model
multioutput_mse = mean_squared_error(Y_test, multioutput_pred)
multioutput_mae = mean_absolute_error(Y_test, multioutput_pred)

# Calculate performance metrics for decision tree model
tree_mse = mean_squared_error(Y_test, tree_pred)
tree_mae = mean_absolute_error(Y_test, tree_pred)

# Calculate performance metrics for random forest model
forest_mse = mean_squared_error(Y_test, forest_pred)
forest_mae = mean_absolute_error(Y_test, forest_pred)

# Print the performance metrics
print("Multioutput Model - Mean Squared Error:", multioutput_mse)
print("Multioutput Model - Mean Absolute Error:", multioutput_mae)
print("Decision Tree Model - Mean Squared Error:", tree_mse)
print("Decision Tree Model - Mean Absolute Error:", tree_mae)
print("Random Forest Model - Mean Squared Error:", forest_mse)
print("Random Forest Model - Mean Absolute Error:", forest_mae)

# %%
#let's visualze the results 
# Create a comparative visualization
plt.figure(figsize=(10, 4))
models = ['Multioutput', 'Decision Tree', 'Random Forest']
mse_scores = [multioutput_mse, tree_mse, forest_mse]
mae_scores = [multioutput_mae, tree_mae, forest_mae]

# Plot Mean Squared Error (MSE)
plt.subplot(1, 2, 1)
plt.bar(models, mse_scores, color=['blue', 'green', 'purple'])
plt.xlabel('Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparative MSE Scores')

# Plot Mean Absolute Error (MAE)
plt.subplot(1, 2, 2)
plt.bar(models, mae_scores, color=['blue', 'green', 'purple'])
plt.xlabel('Models')
plt.ylabel('Mean Absolute Error(MAE)')
plt.title('Comparative MAE Scores')

plt.tight_layout()
plt.show()
# %%
#from the results of model training we can see that it's better to use MultiOutputRegressor model 
#let's compare the resutls of time-series forcasting and MultiOutputRegressor forecasting 

#floods usually happen in spring
#hence, we want to choose the February-Match windwow for out visualization 
import matplotlib.pyplot as plt
df['date'] = df['year'].astype(str) + "-" + df['month'].astype(str) + "-" + df['day'].astype(str)
df['date'] = pd.to_datetime(df['date'])
pltdf = df[(df['date']>='2020-3-20') & (df['date']<'2020-3-27')]
pltdf
plt.figure(figsize=(10,5))
plt.plot(pltdf['date'],pltdf['water_level'],label = "water level",marker ='o',)

plt.title("Change in water level recorded by Ekidin  station between February 1st and March 31st of 2020")
plt.legend()
plt.grid(False)
plt.show()

#%%
#from the plot, we see that this time period includes an event of flood 
# although our data is not labelled by flood events, we can see such an event from the graph and the news archive
#the flood happened on 2020-03-24 
#therefore, we use the date of 21 days around the event of flood

#get the prediciton of time series 
import numpy as np
model = h2o.load_model("/Users/zhuldyzualikhankyzy/Documents/GitHub/flood_prediciton/models/time_series_forecast/GBM_grid_1_AutoML_1_20240415_115845_model_64")
regression = pd.DataFrame()
regression['date'] = pd.date_range(start='2020-03-17',periods=21)
regression['date'] = pd.to_datetime(regression['date'])

df_reg = df.copy()
df_reg['water_level_1day'] = df_reg['water_level'].shift()
df_reg['water_level_2day'] = df_reg['water_level'].shift(2)
df_reg['water_level_3day'] = df_reg['water_level'].shift(3)
df_reg['water_level_4day'] = df_reg['water_level'].shift(4)
df_reg['water_level_5day'] = df_reg['water_level'].shift(5)
df_reg['water_level_6day'] = df_reg['water_level'].shift(6)
df_reg['water_level_7day'] = df_reg['water_level'].shift(7)
multimodel_X = regression.merge(df_reg,on="date",how='inner')
multimodel_X =multimodel_X[['discharge', 'month', 'water_level_1day', 'water_level_2day', 'water_level_3day', 'water_level_4day', 'water_level_5day', 'water_level_6day', 'water_level_7day']]
h2o.init()

h2o_frame = h2o.H2OFrame(multimodel_X)

pred_21days = model.predict(h2o_frame)


#%%
df['date'] = df['year'].astype(str) + "-" + df['month'].astype(str) + "-" + df['day'].astype(str)
df['date'] = pd.to_datetime(df['date'])
#get the predicitons of MultiOutputRegressor
#define the dates 
#since we want to make predciton for 7 days, we will chose the first 21 days of test set 
multimodel = pd.DataFrame()

#we will do in-sample prediciton for 21 days, (use 3 points that generate 7 precitions each)
multioutput_pred = []

start_dates = ['2020-03-17','2020-03-24','2020-03-31']

for start_date in start_dates: 
    multimodel['date'] = pd.date_range(start=start_date,periods=1)
    multimodel['date'] = pd.to_datetime(multimodel['date'])

    multimodel_X = multimodel.merge(df,on="date",how='inner')
    multimodel_X = multimodel_X.drop(['target_day_1', 'target_day_2', 'target_day_3', 'target_day_4', 'target_day_5', 'target_day_6', 'target_day_7','date'], axis=1)
    multimodel_X = multimodel_X.apply(lambda x: pd.to_numeric(x, errors='ignore'))

    multioutput_pred.append(multioutput_model.predict(multimodel_X.values.reshape(1,-1)))


#%%
#plot the data 
import matplotlib.pyplot as plt

start_date = '2020-02-01'
end_date = '2020-04-07'
# Filter the dataframe to include only the desired period
desired_period= df[(df['date'] >= start_date) & (df['date'] <= end_date)]


# Define the start and end dates of the desired period
start_date = '2020-03-18'
end_date = '2020-04-07'
# Filter the dataframe to include only the desired period
desired_period2= df[(df['date'] >= start_date) & (df['date'] <= end_date)]



multioutput_pred = np.concatenate([x.flatten() for x in multioutput_pred])


# Plot the data for the desired period
plt.figure(figsize=(10,5))
plt.plot(desired_period['date'], desired_period['water_level'], label='Historical Water Level', color='green')
plt.plot(desired_period2['date'], multioutput_pred, label='MultiOutputRegressor', color='blue')
plt.plot(desired_period2['date'], pred_21days['predict'].as_data_frame(), label='LinearRegression', color='red')

plt.title("Change in water level recorded by Ekidin station between February 1st and March 31st of 2020")
plt.legend()
plt.grid(False)
plt.show()

# %%
#plot the historical data of water level 
start_date = '2014-01-01'
end_date = '2021-12-31'
# Filter the dataframe to include only the desired period
desired_period2= df[(df['date'] >= start_date) & (df['date'] <= end_date)]

plt.figure(figsize=(10,5))
plt.plot(desired_period2['date'], desired_period2['water_level'], label='Historical Water Level', color='green')
plt.title("Historical data showing water levels recorded from Ekidin station between January 1st 2014 and December 31st of 2021")
plt.legend()
plt.grid(False)
plt.show()
# %%
