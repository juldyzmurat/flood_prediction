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
