import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error as mse

sns.set_style("darkgrid")

def set_features():
    # Creating this as a method so it's easier to fold it away
    return ['latitude',
            'longitude',
            'year',
            'week_no',
#            'SulphurDioxide_SO2_column_number_density',
#            'SulphurDioxide_SO2_column_number_density_amf',
#            'SulphurDioxide_SO2_slant_column_number_density',
#            'SulphurDioxide_cloud_fraction',
#            'SulphurDioxide_sensor_azimuth_angle',
#            'SulphurDioxide_sensor_zenith_angle',
#            'SulphurDioxide_solar_azimuth_angle',
#            'SulphurDioxide_solar_zenith_angle',
#            'SulphurDioxide_SO2_column_number_density_15km',
#            'CarbonMonoxide_CO_column_number_density',
#            'CarbonMonoxide_H2O_column_number_density',
#            'CarbonMonoxide_cloud_height',
#            'CarbonMonoxide_sensor_altitude',
#            'CarbonMonoxide_sensor_azimuth_angle',
#            'CarbonMonoxide_sensor_zenith_angle',
#            'CarbonMonoxide_solar_azimuth_angle',
#            'CarbonMonoxide_solar_zenith_angle',
#            'NitrogenDioxide_NO2_column_number_density',
#            'NitrogenDioxide_tropospheric_NO2_column_number_density',
#            'NitrogenDioxide_stratospheric_NO2_column_number_density',
#            'NitrogenDioxide_NO2_slant_column_number_density',
#            'NitrogenDioxide_tropopause_pressure',
#            'NitrogenDioxide_absorbing_aerosol_index',
#            'NitrogenDioxide_cloud_fraction',
#            'NitrogenDioxide_sensor_altitude',
#            'NitrogenDioxide_sensor_azimuth_angle',
#            'NitrogenDioxide_sensor_zenith_angle',
#            'NitrogenDioxide_solar_azimuth_angle',
#            'NitrogenDioxide_solar_zenith_angle',
#            'Formaldehyde_tropospheric_HCHO_column_number_density',
#            'Formaldehyde_tropospheric_HCHO_column_number_density_amf',
#            'Formaldehyde_HCHO_slant_column_number_density',
#            'Formaldehyde_cloud_fraction',
#            'Formaldehyde_solar_zenith_angle',
#            'Formaldehyde_solar_azimuth_angle',
#            'Formaldehyde_sensor_zenith_angle',
#            'Formaldehyde_sensor_azimuth_angle',
#            'UvAerosolIndex_absorbing_aerosol_index',
#            'UvAerosolIndex_sensor_altitude',
#            'UvAerosolIndex_sensor_azimuth_angle',
#            'UvAerosolIndex_sensor_zenith_angle',
#            'UvAerosolIndex_solar_azimuth_angle',
#            'UvAerosolIndex_solar_zenith_angle',
#            'Ozone_O3_column_number_density',
#            'Ozone_O3_column_number_density_amf',
#            'Ozone_O3_slant_column_number_density',
#            'Ozone_O3_effective_temperature',
#            'Ozone_cloud_fraction',
#            'Ozone_sensor_azimuth_angle',
#            'Ozone_sensor_zenith_angle',
#            'Ozone_solar_azimuth_angle',
#            'Ozone_solar_zenith_angle',
#            'Cloud_cloud_fraction',
#            'Cloud_cloud_top_pressure',
#            'Cloud_cloud_top_height',
#            'Cloud_cloud_base_pressure',
#            'Cloud_cloud_base_height',
#            'Cloud_cloud_optical_depth',
#            'Cloud_surface_albedo',
#            'Cloud_sensor_azimuth_angle',
#            'Cloud_sensor_zenith_angle',
#            'Cloud_solar_azimuth_angle',
#            'Cloud_solar_zenith_angle',
    ]

features = set_features()
labels = ["emission"]

train = pd.read_csv("./data/train.csv", usecols=features+labels)
train = train.dropna().sample(frac=0.1)
test = pd.read_csv("./data/test.csv", usecols=features)
test = test.dropna().sample(frac=0.1)

y = train[labels]

train_test = [train, test]
for idx, df in enumerate(train_test):
    # Adding 'month_no' column
    df['date'] = pd.to_datetime('2021' + df['week_no'].astype(str) + '0', format='%Y%W%w')
    df['month_no'] = df['date'].dt.month
    df.drop(columns=['date'], inplace=True)
    
    # Adding 'corona' active period column    
    df['corona'] = (df.year == 2020) & (df.month_no > 2)
    
    # Keeping 'latitude', 'longitude', 'week_no', 'month_no', 'corona' features only.
    train_test[idx] = df[['latitude', 'longitude', 'week_no', 'month_no', 'corona']]

train, test = train_test

train_x, test_x, train_y, test_y = train_test_split(train, y, test_size=0.2, random_state=42)

covid_train_x = train_x[train_x["corona"]]
not_covid_train_x = train_x[train_x["corona"].isin([False])]
sns.kdeplot(covid_train_x, x=train_y.values.squeeze())

model = RandomForestRegressor(n_estimators=100, random_state=42, verbose=1)
ensemble = BaggingRegressor(base_estimator=model, n_estimators=5, random_state=42, verbose=1)
ensemble.fit(train_x, train_y)

preds = ensemble.predict(test_x)
print(np.sqrt(mse(test_y, preds)))
