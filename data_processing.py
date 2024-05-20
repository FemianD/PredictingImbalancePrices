import csv
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import TimeSeriesSplit

#load data
data_tennet = pd.read_csv('bidpriceladder.csv', delimiter=';')

data_prices = pd.read_csv('tennet_prices.csv', delimiter=';')

data_production = pd.read_csv('production.csv', delimiter=';')

#Independent variable
def custom_function(row):
    if row['Regulation_state'] == 1:
        return row['upward']
    if row['Regulation_state'] == -1:
        return row ['downward']
    if row ['Regulation_state'] == 2:
        return (row['upward'] + row['downward']) / 2
    else:
        return row['take_from_system']
    
data_prices['imbalance_price'] = data_prices.apply(custom_function, axis=1)

# Add IV to final dataset
df = data_tennet
df['imbalance_price'] = data_prices['imbalance_price']
df['gas_production'] = data_production['Fossil Gas  - Actual Aggregated [MW]']
df['solar_production'] = data_production['Solar  - Actual Aggregated [MW]']
df['wind_production'] = data_production['Wind Offshore  - Actual Aggregated [MW]'] + data_production['Wind Onshore  - Actual Aggregated [MW]']

print(df)

df.to_csv('final_dataset.csv', index=False)

