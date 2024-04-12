#%%
#change the order of columns 
import pandas as pd 
import os 

directory_path = "/home/zxu4/CSE_MSE_RXF131/cradle-members/mdle/zxu4/ss_weather/"

for filename in os.listdir(directory_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory_path, filename)
        df = pd.read_csv(file_path)
        new_column_order = ["dateTime",  "dateTime_adj", "GHI", "DIF", "GTI", "TEMP", "RH", "PWAT", "WS"]
        df = df[new_column_order]
        df.to_csv(file_path, index=False)


# %%
#combine weather and power into one tensor 
power = pd.read_csv("/home/zxu4/stGNN/21-pv-stgnn/data/zxu/rwb-s4p_stgae_imputation.csv")

# %%
import pandas as pd 
powercl = pd.read_csv("/home/zxu4/CSE_MSE_RXF131/staging/sdle-guest/ucf-sunsmart/s4p-gae/rwb-s4p_cleaned_scaled.csv")
# %%
import pandas as pd
#merge together the weather with power (cleaned, but not the imputed version)
weather= pd.read_csv("/home/zxu4/CSE_MSE_RXF131/cradle-members/mdle/zxu4/ss_weather/SS2001_weather.csv")
weather.columns  = ['dateTime','tmst','GHI','DIF','GTI','TEMP','RH','PWAT',	'WS']
#GHI is most commonly used to predict the power 
weather_time = pd.DataFrame(weather['tmst'])
power = pd.read_csv("/home/zxu4/CSE_MSE_RXF131/staging/sdle-guest/ucf-sunsmart/s4p-gae/rwb-s4p_cleaned.csv")
power['tmst'] = pd.to_datetime(power['tmst'])  # Convert to datetime object
power['tmst'] = power['tmst'].dt.strftime("%Y-%m-%d %H:%M:%S") 
merged_df = weather_time.merge(power, on='tmst', how='left')
merged_df

# %%
#this doesn't work 
import os 
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.automl import H2OAutoML
import matplotlib.pyplot as plt



weatherdir = "/home/zxu4/CSE_MSE_RXF131/cradle-members/mdle/zxu4/ss_weather/"
for column in merged_df.columns[1:2]: 
    colcode = column[1:5]
    weathercsv = pd.read_csv(os.path.join(weatherdir,"SS"+colcode+"_weather.csv"))
    weathercsv = weathercsv['GHI']
    pair = pd.concat([weathercsv, merged_df[column]], axis=1)
    pair.columns = ['GHI','power']

    #filter out the existent ones using the NaN 
    h2o.init()

    pair_for_pred = pair.dropna()

    plt.scatter([x for x in range(len(pair_for_pred))],pair_for_pred.power,alpha=0.025)
    plt.show()
    print(pair['power'].corr(pair['GHI']))

    pair_for_pred = h2o.H2OFrame(pair_for_pred)
    train, test = pair_for_pred.split_frame(ratios=[.8], seed=10)

    y = "power"
    x = ["GHI"]

    train[y] = train[y]
    test[y] = test[y]
    aml = H2OAutoML(max_runtime_secs = 30)
    aml.train(x = x, y = y, training_frame = train)
    print(aml.leaderboard)
    perf = aml.leader.model_performance(test)


# %%
#try GBR model 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


weatherdir = "/home/zxu4/CSE_MSE_RXF131/cradle-members/mdle/zxu4/ss_weather/"
for column in merged_df.columns[1:2]: 
    colcode = column[1:5]
    weathercsv = pd.read_csv(os.path.join(weatherdir,"SS"+colcode+"_weather.csv"))
    weathercsv = weathercsv[['dateTime_adj','GHI']]
    pair = pd.concat([weathercsv, merged_df[column]], axis=1)
    pair.columns = ['tmst','GHI','power']

    pair_for_pred = pair.dropna()

    X = pair_for_pred[['GHI']]
    y = pair_for_pred['power']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Gradient Boosting Regression model
    model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)
   
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Squared Error:", mae)



# %%
#this produces 30% of error
#to be further reduced by introdution of night 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


weatherdir = "/home/zxu4/CSE_MSE_RXF131/cradle-members/mdle/zxu4/ss_weather/"
for column in merged_df.columns[1:2]: 
    colcode = column[1:5]
    weathercsv = pd.read_csv(os.path.join(weatherdir,"SS"+colcode+"_weather.csv"))
    weathercsv = weathercsv[['dateTime_adj','GHI','DIF','GTI']]

    pair = pd.concat([weathercsv, merged_df[column]], axis=1)
    pair.columns = ['tmst','GHI','DIF','GTI','power']

    avrad = (pair['GHI'] + pair['DIF'] + pair['GTI']) / 3
    tolerance = 10
    within_tolerance = abs(avrad) <= tolerance
    pair = pair[~within_tolerance]


    pair['tmst'] = pd.to_datetime(pair['tmst'])
    pair['year'] = pair['tmst'].dt.year
    pair['month'] = pair['tmst'].dt.month
    pair['day'] = pair['tmst'].dt.day
    pair['hour'] = pair['tmst'].dt.hour
    pair['minute'] = pair['tmst'].dt.minute

    pair_for_pred = pair.dropna()
    print(pair_for_pred)

    X = pair_for_pred[['year','month','day','hour','minute','GHI','DIF','GTI']]
    y = pair_for_pred['power']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



    # Create a feedforward neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
        ])  

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    # Predict on the test set
    
    y_pred = model.predict(X_test)
    y_pred[y_pred < 0] = 0

    y_pred = y_pred.tolist()
    y_test = y_test.tolist()
 


    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)

    error_percentage = (mae / np.mean(y_test)) * 100
    print("Mean Absolute Error percentage:", error_percentage)

    
    #ydif = [abs((ypred - ytest) / ytest) for ypred, ytest in zip(ypred_list, ytest_list)]
    #print(np.mean(ydif))
    

# %%
#st-GAE based on irradiance adjacency matrix 
#input variables include only the irradiance  


#%%
# Model VS Data problem using Canada data 
#13 percent is our best 

#add solar time 
#try SLURM to find the optimun characterists 

import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

pv1_in1_power = pd.read_csv("/home/zxu4/CSE_MSE_RXF131/vuv-data/proj/CanadianSolar/imputed_data/alvkhk5-phys_imputation.csv")
columns_to_check = ["temp_isna", "poay_isna", "wspa_isna", "modt_isna", "idcp_isna"]
pv1_in1_power = pv1_in1_power[~(pv1_in1_power[columns_to_check] == 1).any(axis=1)]

pair = pv1_in1_power


pair['tmst'] = pd.to_datetime(pair['tmst'])
pair['year'] = pair['tmst'].dt.year
pair['month'] = pair['tmst'].dt.month
pair['day'] = pair['tmst'].dt.day
pair['hour'] = pair['tmst'].dt.hour
pair['minute'] = pair['tmst'].dt.minute

pair_for_pred = pair.dropna()
print(pair_for_pred)

X = pair_for_pred[['year','month','day','hour','minute','poay']]
y = pair_for_pred['idcp_sandia']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Create a feedforward neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(6,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
    ])  

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Predict on the test set

y_pred = model.predict(X_test)
y_pred[y_pred < 0] = 0

y_pred = y_pred.tolist()
y_test = y_test.tolist()



# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

error_percentage = (mae / np.mean(y_test)) * 100
print("Mean Absolute Error percentage:", error_percentage)








# %%
from itertools import combinations
from itertools import product
import pandas as pd


# Define possible values for each feature
rate = [0.1, 0.01, 0.001,0.0001]
epochs = [100,150,200,300]
batch = [4,8,16,32,64]

# Combine the possible values into a list of tuples representing all combinations
combinations_list = list(product(rate, epochs, batch))

# Flatten the list of tuples to get all ossible combinations
all_combinations = [list(combination) for combination in combinations_list]

# Print the list of all possible combinations
for combination in all_combinations:
    combination = formatted_combination = ' '.join([f'{x:.2f}' if isinstance(x, float) else str(x) for x in combination])
    
all_combinations

columns = ['rate', 'epochs', 'batch']

# Create a DataFrame
df = pd.DataFrame(all_combinations, columns=columns)
df.to_parquet("/home/zxu4/stGNN/21-pv-stgnn/scripts/slurm_stGNN/DL_regression.parquet")



# %%
