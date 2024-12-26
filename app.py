import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

dataset1 = pd.read_csv('https://raw.githubusercontent.com/chidaksh/CosmosocClub/master/Parsec2023/user1_data.csv', delimiter=',')
dataset2 = pd.read_csv('https://raw.githubusercontent.com/chidaksh/CosmosocClub/master/Parsec2023/user2_data.csv', delimiter=',')
df = pd.concat([dataset1, dataset2], ignore_index=True)

df['ttime'] = pd.to_datetime(df['ttime'])
df['ttime'] = df['ttime'].apply(lambda x: x.timestamp())

x = df.drop(['sm'], axis=1).iloc[:, 0:7].values  
y = df[['sm']].values.ravel()  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


imputer = SimpleImputer(missing_values=np.nan, strategy='median')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

imputer_y = SimpleImputer(missing_values=np.nan, strategy='mean')
y_train = imputer_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test = imputer_y.transform(y_test.reshape(-1, 1)).ravel()

scaler = StandardScaler()
x_train = scaler.fit_transform(pd.DataFrame(x_train))
x_test = scaler.transform(pd.DataFrame(x_test))


regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x_train, y_train)

crop_data = pd.read_csv('Crop_recommendation.csv')
X_crop = crop_data.drop('label', axis=1)
y_crop = crop_data['label']

X_features = X_crop[['temperature', 'humidity', 'ph', 'rainfall']]
y_n, y_p, y_k = X_crop['N'], X_crop['P'], X_crop['K']

scaler_crop_full = StandardScaler()
scaler_crop_full.fit(X_crop)


scaler_npk_features = StandardScaler()
scaler_npk_features.fit(X_features)

n_regressor = RandomForestRegressor(n_estimators=20, random_state=0)
p_regressor = RandomForestRegressor(n_estimators=20, random_state=0)
k_regressor = RandomForestRegressor(n_estimators=20, random_state=0)

n_regressor.fit(scaler_npk_features.transform(X_features), y_n)
p_regressor.fit(scaler_npk_features.transform(X_features), y_p)
k_regressor.fit(scaler_npk_features.transform(X_features), y_k)


crop_classifier = RandomForestClassifier(n_estimators=20, random_state=0)
crop_classifier.fit(scaler_crop_full.transform(X_crop), y_crop)

def sensor_to_vwc(sensor_value, conversion_factor):
    return sensor_value * conversion_factor

def provide_recommendations(n, p, k, vwc, crop):
    print("\n--- Soil and Fertilizer Recommendations ---")

    if n < 240:
        print(f"Nitrogen (N) is low: {n:.2f} kg/ha. Add urea or ammonium sulfate fertilizers.")
    elif 240 <= n <= 480:
        print(f"Nitrogen (N) is adequate: {n:.2f} kg/ha.")
    else:
        print(f"Nitrogen (N) is high: {n:.2f} kg/ha. Avoid adding nitrogen fertilizers.")
    
  
    if p < 11:
        print(f"Phosphorus (P) is low: {p:.2f} kg/ha. Use rock phosphate or diammonium phosphate.")
    elif 11 <= p <= 22:
        print(f"Phosphorus (P) is adequate: {p:.2f} kg/ha.")
    else:
        print(f"Phosphorus (P) is high: {p:.2f} kg/ha. Avoid adding phosphorus fertilizers.")
   
    if k < 110:
        print(f"Potassium (K) is low: {k:.2f} kg/ha. Use muriate of potash or sulfate of potash.")
    elif 110 <= k <= 280:
        print(f"Potassium (K) is adequate: {k:.2f} kg/ha.")
    else:
        print(f"Potassium (K) is high: {k:.2f} kg/ha. Avoid adding potassium fertilizers.")
    

    print("\nOrganic Matter Recommendation:")
    if n < 240 or k < 110:
        print("Consider adding organic matter like compost or manure to improve soil fertility.")

    print("\nIrrigation Recommendation:")
    if vwc < 20:
        print(f"Volumetric Water Content (VWC) is low: {vwc:.2f}. Irrigation is required.")
    elif 20 <= vwc <= 40:
        print(f"Volumetric Water Content (VWC) is optimal: {vwc:.2f}. Irrigation is not needed currently.")
    else:
        print(f"Volumetric Water Content (VWC) is high: {vwc:.2f}. Avoid overwatering.")

    print(f"\nRecommended Crop: {crop}\n")


def predict_moisture_and_crop():
    print("Please enter the following details for soil moisture prediction:")

    ttime = input("Enter the timestamp (YYYY-MM-DD HH:MM:SS format): ")
    pm1 = float(input("Enter the value for PM1 (Particle Matter 1): "))
    pm2 = float(input("Enter the value for PM2 (Particle Matter 2): "))
    pm3 = float(input("Enter the value for PM3 (Particle Matter 3): "))
    am = float(input("Enter the value for AM (Ambient Moisture): "))
    st = float(input("Enter the value for ST (Soil Temperature): "))
    lum = float(input("Enter the value for LUM (Luminosity): "))
    temperature = float(input("Enter the temperature in °C: "))
    humidity = float(input("Enter the humidity percentage: "))
    ph = float(input("Enter the soil pH level: "))
    rainfall = float(input("Enter the rainfall in mm: "))

    try:
        ttime = pd.to_datetime(ttime)
        ttime = ttime.timestamp()
    except ValueError:
        print("Error parsing the timestamp. Please ensure it is in 'YYYY-MM-DD HH:MM:SS' format.")
        return


    df_input = pd.DataFrame({
        'ttime': [ttime],
        'pm1': [pm1],
        'pm2': [pm2],
        'pm3': [pm3],
        'am': [am],
        'st': [st],
        'lum': [lum]
    })

    x_input = scaler.transform(df_input)
    y_pred_input = regressor.predict(x_input)

    conversion_factor = 12.04
    vwc = sensor_to_vwc(y_pred_input[0], conversion_factor)

    print(f"\nPredicted Moisture Content (Sensor Value): {y_pred_input[0]:.2f}")
    print(f"Predicted Volumetric Water Content (VWC): {vwc:.2f}")

    npk_features = scaler_npk_features.transform([[temperature, humidity, ph, rainfall]])
    predicted_n = n_regressor.predict(npk_features)[0]
    predicted_p = p_regressor.predict(npk_features)[0]
    predicted_k = k_regressor.predict(npk_features)[0]

    print(f"\nPredicted N (Nitrogen): {predicted_n:.2f}")
    print(f"Predicted P (Phosphorus): {predicted_p:.2f}")
    print(f"Predicted K (Potassium): {predicted_k:.2f}")


    crop_input = scaler_crop_full.transform([[predicted_n, predicted_p, predicted_k, temperature, humidity, ph, rainfall]])
    recommended_crop = crop_classifier.predict(crop_input)[0]

    provide_recommendations(predicted_n, predicted_p, predicted_k, vwc, recommended_crop)

predict_moisture_and_crop()
