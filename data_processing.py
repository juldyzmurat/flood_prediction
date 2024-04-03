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
