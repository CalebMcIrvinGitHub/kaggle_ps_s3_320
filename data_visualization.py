import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse

sns.set_style("darkgrid")

def set_features():
    # Creating this as a method so it's easier to fold it away
    return [#'latitude',
            #'longitude',
            #'year',
            #'week_no',
            'SulphurDioxide_SO2_column_number_density',
            'SulphurDioxide_SO2_column_number_density_amf',
            'SulphurDioxide_SO2_slant_column_number_density',
            'SulphurDioxide_cloud_fraction',
            'SulphurDioxide_sensor_azimuth_angle',
            'SulphurDioxide_sensor_zenith_angle',
            'SulphurDioxide_solar_azimuth_angle',
            'SulphurDioxide_solar_zenith_angle',
            'SulphurDioxide_SO2_column_number_density_15km',
            'CarbonMonoxide_CO_column_number_density',
            'CarbonMonoxide_H2O_column_number_density',
            'CarbonMonoxide_cloud_height',
            'CarbonMonoxide_sensor_altitude',
            'CarbonMonoxide_sensor_azimuth_angle',
            'CarbonMonoxide_sensor_zenith_angle',
            'CarbonMonoxide_solar_azimuth_angle',
            'CarbonMonoxide_solar_zenith_angle',
            'NitrogenDioxide_NO2_column_number_density',
            'NitrogenDioxide_tropospheric_NO2_column_number_density',
            'NitrogenDioxide_stratospheric_NO2_column_number_density',
            'NitrogenDioxide_NO2_slant_column_number_density',
            'NitrogenDioxide_tropopause_pressure',
            'NitrogenDioxide_absorbing_aerosol_index',
            'NitrogenDioxide_cloud_fraction',
            'NitrogenDioxide_sensor_altitude',
            'NitrogenDioxide_sensor_azimuth_angle',
            'NitrogenDioxide_sensor_zenith_angle',
            'NitrogenDioxide_solar_azimuth_angle',
            'NitrogenDioxide_solar_zenith_angle',
            'Formaldehyde_tropospheric_HCHO_column_number_density',
            'Formaldehyde_tropospheric_HCHO_column_number_density_amf',
            'Formaldehyde_HCHO_slant_column_number_density',
            'Formaldehyde_cloud_fraction',
            'Formaldehyde_solar_zenith_angle',
            'Formaldehyde_solar_azimuth_angle',
            'Formaldehyde_sensor_zenith_angle',
            'Formaldehyde_sensor_azimuth_angle',
            'UvAerosolIndex_absorbing_aerosol_index',
            'UvAerosolIndex_sensor_altitude',
            'UvAerosolIndex_sensor_azimuth_angle',
            'UvAerosolIndex_sensor_zenith_angle',
            'UvAerosolIndex_solar_azimuth_angle',
            'UvAerosolIndex_solar_zenith_angle',
            'Ozone_O3_column_number_density',
            'Ozone_O3_column_number_density_amf',
            'Ozone_O3_slant_column_number_density',
            'Ozone_O3_effective_temperature',
            'Ozone_cloud_fraction',
            'Ozone_sensor_azimuth_angle',
            'Ozone_sensor_zenith_angle',
            'Ozone_solar_azimuth_angle',
            'Ozone_solar_zenith_angle',
            'Cloud_cloud_fraction',
            'Cloud_cloud_top_pressure',
            'Cloud_cloud_top_height',
            'Cloud_cloud_base_pressure',
            'Cloud_cloud_base_height',
            'Cloud_cloud_optical_depth',
            'Cloud_surface_albedo',
            'Cloud_sensor_azimuth_angle',
            'Cloud_sensor_zenith_angle',
            'Cloud_solar_azimuth_angle',
            'Cloud_solar_zenith_angle',
    ]

features = set_features()
labels = ["emission"]

#cols = pd.read_csv("./data/train.csv", nrows=1)
#dropcols = ["ID_LAT_LON_YEAR_WEEK"]
#cols = [col for col in cols if col not in dropcols]
#[(col, train.loc[:, col].dropna().shape) for col in cols]

train = pd.read_csv("./data/train.csv", usecols=features+labels)
train = train.dropna().sample(frac=0.1)

x = train[features]
#x = stats.zscore(x, axis=0)

y = train[labels]

#test = SelectKBest(score_func=f_regression, k=50)
#fit = test.fit(x, y)
#features = fit.transform(x)

#sns.pairplot(train.sample(100))

#model = LinearRegression().fit(x, y)
#model = SVR(verbose=True).fit(x, y)

fold = KFold(n_splits=5, shuffle=True, random_state=None)
for idx, (train_idx, test_idx) in enumerate(fold.split(x)):
    train_data, test_data = x.iloc[train_idx], x.iloc[test_idx] 
    train_labels, test_labels = y.iloc[train_idx], y.iloc[test_idx]
    rf = RandomForestRegressor(n_estimators = 5, max_depth = 100, verbose = 0)
    rf.fit(train_data, train_labels)

    preds = rf.predict(test_data)
    score = np.sqrt(mse(preds, test_labels.values.reshape(-1)))
    print(score)


#print(model.score(x, y))

print("Done plotting")
