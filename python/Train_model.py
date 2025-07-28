import pandas as pd
import numpy as np
import joblib
import os
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Flatten, Concatenate
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error

# ==== Load and Prepare Data ====
df = pd.read_excel("Commodities_price.xlsx")

# Clean and format
df['District Name'] = df['District Name'].astype(str).str.strip().str.title()
df['Crop'] = df['Crop'].astype(str).str.strip().str.title()
df = df.dropna(subset=['Modal Price', 'Crop', 'District Name'])

df['Price Date'] = pd.to_datetime(df['Price Date'], errors='coerce')
df = df.dropna(subset=['Price Date'])

# Create Harvest Month column (month of price date)
df['Harvest Month'] = df['Price Date'].dt.month_name().str[:3]

# ==== Encoding ====
crop_encoder = LabelEncoder()
district_encoder = LabelEncoder()
harvest_month_encoder = LabelEncoder()

df['Crop_encoded'] = crop_encoder.fit_transform(df['Crop'])
df['District_encoded'] = district_encoder.fit_transform(df['District Name'])
df['Harvest_month_encoded'] = harvest_month_encoder.fit_transform(df['Harvest Month'])

# ==== Smoothing & Scaling ====
df['Smooth Price'] = df.groupby(['Crop', 'District Name'])['Modal Price']\
                       .transform(lambda x: x.rolling(3, min_periods=1).mean())

scaler = MinMaxScaler()
df['Scaled Price'] = scaler.fit_transform(df[['Smooth Price']])

# ==== Create Sequences ====
sequence_length = 6
X_price, X_crop, X_district, X_harvest_month, y = [], [], [], [], []

for (crop, district), group in df.groupby(['Crop', 'District Name']):
    group = group.sort_values('Price Date')
    prices = group['Scaled Price'].values
    month_vals = group['Harvest_month_encoded'].values

    if len(prices) < sequence_length + 1:
        continue

    crop_val = crop_encoder.transform([crop])[0]
    dist_val = district_encoder.transform([district])[0]

    for i in range(len(prices) - sequence_length):
        X_price.append(prices[i:i + sequence_length])
        y.append(prices[i + sequence_length])
        X_crop.append(crop_val)
        X_district.append(dist_val)
        X_harvest_month.append(month_vals[i + sequence_length])

# ==== Convert to Arrays ====
X_price = np.array(X_price).reshape((-1, sequence_length, 1))
X_crop = np.array(X_crop)
X_district = np.array(X_district)
X_harvest_month = np.array(X_harvest_month)
y = np.array(y)

# ==== Build LSTM Model ====
input_price = Input(shape=(sequence_length, 1), name='price_sequence')
input_crop = Input(shape=(1,), name='crop_input')
input_dist = Input(shape=(1,), name='district_input')
input_month = Input(shape=(1,), name='harvest_month_input')

# Embeddings
crop_embed = Embedding(input_dim=len(crop_encoder.classes_), output_dim=4)(input_crop)
dist_embed = Embedding(input_dim=len(district_encoder.classes_), output_dim=4)(input_dist)
month_embed = Embedding(input_dim=len(harvest_month_encoder.classes_), output_dim=3)(input_month)

crop_flat = Flatten()(crop_embed)
dist_flat = Flatten()(dist_embed)
month_flat = Flatten()(month_embed)

# LSTM for price sequence
lstm_out = LSTM(64)(input_price)

# Combine all
merged = Concatenate()([lstm_out, crop_flat, dist_flat, month_flat])
dense_out = Dense(32, activation='relu')(merged)
output = Dense(1)(dense_out)

model = Model(inputs=[input_price, input_crop, input_dist, input_month], outputs=output)
model.compile(optimizer='adam', loss=MeanSquaredError())

# ==== Train Model ====
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    [X_price, X_crop, X_district, X_harvest_month],
    y,
    epochs=50,
    batch_size=4,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ==== Save Assets ====
os.makedirs("models", exist_ok=True)
model.save("models/unified_lstm_model.h5")
joblib.dump(scaler, "models/price_scaler.pkl")
joblib.dump(crop_encoder, "models/crop_encoder.pkl")
joblib.dump(district_encoder, "models/district_encoder.pkl")
joblib.dump(harvest_month_encoder, "models/harvest_month_encoder.pkl")

# ==== Evaluate Model ====
y_pred = model.predict([X_price, X_crop, X_district, X_harvest_month])
y_pred_act = scaler.inverse_transform(y_pred)
y_true_act = scaler.inverse_transform(y.reshape(-1, 1))

mse = mean_squared_error(y_true_act, y_pred_act)
rmse = root_mean_squared_error(y_true_act, y_pred_act)
mae = mean_absolute_error(y_true_act, y_pred_act)
r2 = r2_score(y_true_act, y_pred_act)

print("\n📊 Model Evaluation:")
print(f"✅ MSE  : {mse:.2f}")
print(f"✅ RMSE : {rmse:.2f}")
print(f"✅ MAE  : {mae:.2f}")
print(f"✅ R²   : {r2:.4f}")
print("\n🎉 Model trained and saved successfully.")
