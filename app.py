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











# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import numpy as np
# import datetime as dt
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# # Initialize Flask app
# app = Flask(__name__)

# # Load and preprocess soil moisture dataset
# dataset1 = pd.read_csv('https://raw.githubusercontent.com/chidaksh/CosmosocClub/master/Parsec2023/user1_data.csv', delimiter=',')
# dataset2 = pd.read_csv('https://raw.githubusercontent.com/chidaksh/CosmosocClub/master/Parsec2023/user2_data.csv', delimiter=',')
# df = pd.concat([dataset1, dataset2], ignore_index=True)

# df['ttime'] = pd.to_datetime(df['ttime'])
# df['ttime'] = df['ttime'].apply(lambda x: x.timestamp())

# x = df.drop(['sm'], axis=1).iloc[:, 0:7].values
# y = df[['sm']].values.ravel()

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# # Handle missing values
# imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# x_train = imputer.fit_transform(x_train)
# x_test = imputer.transform(x_test)

# imputer_y = SimpleImputer(missing_values=np.nan, strategy='mean')
# y_train = imputer_y.fit_transform(y_train.reshape(-1, 1)).ravel()
# y_test = imputer_y.transform(y_test.reshape(-1, 1)).ravel()

# # Scale features
# scaler = StandardScaler()
# x_train = scaler.fit_transform(pd.DataFrame(x_train))
# x_test = scaler.transform(pd.DataFrame(x_test))

# # Train soil moisture model
# regressor = RandomForestRegressor(n_estimators=20, random_state=0)
# regressor.fit(x_train, y_train)

# # Load crop recommendation dataset
# crop_data = pd.read_csv('Crop_recommendation.csv')
# X_crop = crop_data.drop('label', axis=1)
# y_crop = crop_data['label']

# # Separate NPK prediction features
# X_features = X_crop[['temperature', 'humidity', 'ph', 'rainfall']]
# y_n, y_p, y_k = X_crop['N'], X_crop['P'], X_crop['K']

# # Scale features for crop classification
# scaler_crop_full = StandardScaler()
# scaler_crop_full.fit(X_crop)

# # Scale features for NPK prediction
# scaler_npk_features = StandardScaler()
# scaler_npk_features.fit(X_features)

# # Train NPK predictors
# n_regressor = RandomForestRegressor(n_estimators=20, random_state=0)
# p_regressor = RandomForestRegressor(n_estimators=20, random_state=0)
# k_regressor = RandomForestRegressor(n_estimators=20, random_state=0)

# n_regressor.fit(scaler_npk_features.transform(X_features), y_n)
# p_regressor.fit(scaler_npk_features.transform(X_features), y_p)
# k_regressor.fit(scaler_npk_features.transform(X_features), y_k)

# # Train crop recommendation classifier
# crop_classifier = RandomForestClassifier(n_estimators=20, random_state=0)
# crop_classifier.fit(scaler_crop_full.transform(X_crop), y_crop)

# # Helper function for VWC conversion
# def sensor_to_vwc(sensor_value, conversion_factor):
#     return sensor_value * conversion_factor

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json

#     # Extract inputs
#     ttime = data.get('ttime')
#     pm1 = float(data.get('pm1'))
#     pm2 = float(data.get('pm2'))
#     pm3 = float(data.get('pm3'))
#     am = float(data.get('am'))
#     st = float(data.get('st'))
#     lum = float(data.get('lum'))
#     temperature = float(data.get('temperature'))
#     humidity = float(data.get('humidity'))
#     ph = float(data.get('ph'))
#     rainfall = float(data.get('rainfall'))

#     try:
#         ttime = pd.to_datetime(ttime)
#         ttime = ttime.timestamp()
#     except ValueError:
#         return jsonify({'error': 'Invalid timestamp format. Please use "YYYY-MM-DD HH:MM:SS".'})

#     # Predict soil moisture
#     df_input = pd.DataFrame({
#         'ttime': [ttime],
#         'pm1': [pm1],
#         'pm2': [pm2],
#         'pm3': [pm3],
#         'am': [am],
#         'st': [st],
#         'lum': [lum]
#     })

#     x_input = scaler.transform(df_input)
#     y_pred_input = regressor.predict(x_input)

#     conversion_factor = 12.04
#     vwc = sensor_to_vwc(y_pred_input[0], conversion_factor)

#     # Predict NPK values
#     npk_features = scaler_npk_features.transform([[temperature, humidity, ph, rainfall]])
#     predicted_n = n_regressor.predict(npk_features)[0]
#     predicted_p = p_regressor.predict(npk_features)[0]
#     predicted_k = k_regressor.predict(npk_features)[0]

#     # Recommend crops
#     crop_input = scaler_crop_full.transform([[predicted_n, predicted_p, predicted_k, temperature, humidity, ph, rainfall]])
#     recommended_crop = crop_classifier.predict(crop_input)[0]

#     return jsonify({
#         'moisture_sensor': y_pred_input[0],
#         'vwc': vwc,
#         'predicted_n': predicted_n,
#         'predicted_p': predicted_p,
#         'predicted_k': predicted_k,
#         'recommended_crop': recommended_crop
#     })

# if __name__ == '__main__':
#     app.run(debug=True)
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

# # Load and preprocess datasets
# dataset1 = pd.read_csv('https://raw.githubusercontent.com/chidaksh/CosmosocClub/master/Parsec2023/user1_data.csv')
# dataset2 = pd.read_csv('https://raw.githubusercontent.com/chidaksh/CosmosocClub/master/Parsec2023/user2_data.csv')
# df = pd.concat([dataset1, dataset2], ignore_index=True)

# df['ttime'] = pd.to_datetime(df['ttime']).apply(lambda x: x.timestamp())
# x = df.drop(['sm'], axis=1).iloc[:, 0:7].values
# y = df[['sm']].values.ravel()

# # Train-test split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# # Impute missing values
# imputer = SimpleImputer(strategy='median')
# x_train = imputer.fit_transform(x_train)
# x_test = imputer.transform(x_test)

# imputer_y = SimpleImputer(strategy='mean')
# y_train = imputer_y.fit_transform(y_train.reshape(-1, 1)).ravel()
# y_test = imputer_y.transform(y_test.reshape(-1, 1)).ravel()

# # Standardize features
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# # Regression Model
# regressor = RandomForestRegressor(n_estimators=20, random_state=0)
# regressor.fit(x_train, y_train)

# # Predict and evaluate regression
# y_pred_train = regressor.predict(x_train)
# y_pred_test = regressor.predict(x_test)

# # Feature Importance for Regression
# feature_importances_reg = regressor.feature_importances_

# # Plot 1: Actual vs. Predicted (Train and Test)
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Test Data')
# plt.scatter(y_train, y_pred_train, color='orange', alpha=0.6, label='Train Data')
# plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Ideal Fit')
# plt.title('Actual vs Predicted Soil Moisture')
# plt.xlabel('Actual Soil Moisture')
# plt.ylabel('Predicted Soil Moisture')
# plt.legend()
# plt.grid()
# plt.show()

# # Plot 2: Residual Distribution
# residuals = y_test - y_pred_test
# plt.figure(figsize=(8, 5))
# sns.histplot(residuals, kde=True, color='purple', bins=20)
# plt.title('Residuals Distribution')
# plt.xlabel('Residuals')
# plt.ylabel('Frequency')
# plt.grid()
# plt.show()

# # Plot 3: Heatmap of Correlations
# plt.figure(figsize=(10, 8))
# correlation_matrix = df.corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()

# # Plot 4: Pair Plot of Physical Parameters
# sns.pairplot(df.drop(['sm'], axis=1).iloc[:, :7])
# plt.suptitle('Pair Plot of Physical Parameters', y=1.02)
# plt.show()

# # Plot 5: Feature Importances for Regression
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(feature_importances_reg)), feature_importances_reg, tick_label=df.columns[:7], color='skyblue')
# plt.title('Feature Importance (Regression)')
# plt.ylabel('Importance')
# plt.xlabel('Features')
# plt.grid()
# plt.show()

# # Classification Model
# crop_data = pd.read_csv('Crop_recommendation.csv')
# X_crop = crop_data.drop('label', axis=1)
# y_crop = crop_data['label']

# scaler_crop = StandardScaler()
# X_crop_scaled = scaler_crop.fit_transform(X_crop)

# crop_classifier = RandomForestClassifier(n_estimators=20, random_state=0)
# crop_classifier.fit(X_crop_scaled, y_crop)


# y_crop_pred = crop_classifier.predict(X_crop_scaled)

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_crop, y_crop_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=crop_data['label'].unique(),
#             yticklabels=crop_data['label'].unique())
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # Classification Report
# print("Classification Report:")
# print(classification_report(y_crop, y_crop_pred))

# # Plot 6: Actual vs Predicted Crop Labels
# plt.figure(figsize=(10, 6))
# plt.hist([y_crop, y_crop_pred], bins=len(y_crop.unique()), alpha=0.7, label=['Actual', 'Predicted'], color=['green', 'blue'])
# plt.title('Actual vs Predicted Crop Labels')
# plt.xlabel('Crop Labels')
# plt.ylabel('Frequency')
# plt.legend()
# plt.grid()
# plt.show()
