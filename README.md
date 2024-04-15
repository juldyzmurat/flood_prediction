# Flood Prediciton using meteo- and hydro- data in Ekidin station, Kostanay, Kazakhstan 

This project predicts water levels in Ekidin station, Kostanay Resion, Kazakhstan bases on meteo and hydro data from 2014-01-01 until 12-31-2020. 
The project was motivated by National Flood Crisis that has been taking place in Kazakhstan since March 29th, 2024. 

## Data

The raw data was pulled from The National Hydrometeorological Service of Kazakhstan data base 
* https://www.kazhydromet.kz/en/

## Packages

The key Python packages used in this project are:

**Data Manipulation and Analysis**:
- pandas
- numpy
- matplotlib
- seaborn

**Machine Learning**:
- sklearn.datasets
- sklearn.multioutput.MultiOutputRegressor
- sklearn.linear_model.ElasticNet
- sklearn.tree.DecisionTreeRegressor
- sklearn.ensemble.RandomForestRegressor

**Model Evaluation**:
- sklearn.metrics.mean_squared_error
- sklearn.metrics.mean_absolute_error

**Preprocessing**:
- sklearn.preprocessing.StandardScaler
- sklearn.preprocessing.RobustScaler
- sklearn.model_selection.train_test_split

**Model Training**:
- tensorflow
- tensorflow.keras.models.Sequential
- tensorflow.keras.layers.Dense
- tensorflow.keras.layers.Dropout
- tensorflow.keras.optimizers.Adam

**Miscellaneous Utils**:
- joblib.dump
- joblib.load

### Data Wrangling

1. **Join**: Hydrology and Meteo features were joined
2. **Imputation**: Hydro data had very little missingness that was imputed through finding the average of neighboring dates or filled in with 0 based on the nature of missing point. Meteo data was mostly imputed in the same fashion. Precipitation data had large periods of missingness that was imputed using a Deep Learning Model trained on non-temporal weather attributes. 
4. **Feature generation**: Additonal features were generated through aggregation functions and sliding windows 
5. **Train Test Split**: 80-20 stratified split based on continous period to form training and testing sets 

## Modeling

Two main ML approaches were tested in this project:

1. **Time Series Forecast**: 
   - **Overview**: The project explored the use of AutoML tools, with a particular focus on the H2O platform. The H2O's Gradient Boosting Machine (GBM) was utilized to forecast future values in a time series setting.
   - **Objective**: The model aimed to predict the water levels for the next day using historical data.
   - **Results**: The model's performance was evaluated based on standard metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE).
   - **Visualization**:
     ![Time Series Forecast Results](figures/autoML) <!-- Replace 'path/to/timeseries_image.png' with the actual path to the image file. -->

2. **Multi-output regression**:
   - **Overview**: This method involved the application of three common multi-output regression models: the Multioutput Regressor, Decision Tree, and Random Forest models.
   - **Objective**: The aim was to predict water levels for the next seven days.
   - **Model Comparison**: The models were compared based on their MAE and MSE scores to determine the most effective approach.
   - **Selected Model**: Based on the comparative analysis, the Multioutput Regressor model was selected for the final predictions.
   - **Visualization**:
     ![Multi-output Regression Results](path/to/multioutput_image.png) <!-- Replace 'path/to/multioutput_image.png' with the actual path to the image file. -->

Hyperparameter tuning and cross validation were utilized.

## Results  

Multiple models were evaluated based on accuracy on a held-out EY test set consisting of Sentinel-1 SAR data over unknown crop types.

The best performing model was a **random forest classifier trained on engineered relative vegetation index (RVI) features**, which achieved **85% accuracy** at predicting crop types on the EY test data.

The next best model, a 1D CNN operating directly on the VV/VH time series, scored 10 percentage points lower at 75% accuracy.

The high performance of the RF+RVI approach can be attributed to:

* The derived polarization ratios captured in RVI provide meaningful geospatial features
* Ensemble modeling reduces overfitting  compared to deep CNNs
* Interpretability of RFs allows analysis of important variables  

Given the significant jump in accuracy on external industry data, the RF+RVI approach shows good generalization ability even with limited training samples. This model has been saved for real-world crop type mapping applications.

Future iterations could incorporate recent hyperparameter optimization techniques such as Bayesian hyperband to further improve predictive power.

## Status 
Project is: complete

## Contact 
* zxu4@case.edu
* ktn37@case.edu

## Acknowledgements 
* Data was taken from EY Challenge 2023 and Microsoft Planetary Hub 
