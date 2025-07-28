# KRISHI-JNAN
*Krishi-Jñāna* is a smart agricultural web platform designed to empower farmers by recommending the most suitable crops based on their local conditions and predicting expected market prices at harvest time. This two-stage system blends soil science and data-driven forecasting to improve farm decisions and increase profitability.

Key Features

✅ 1. Unified LSTM Price Predictor
A single LSTM-based model predicts modal prices for all crops across all districts, trained on historical market data.

✅ 2. Contextual Crop Recommendation
Recommends the best crops based on:

District

Available soil types

Sowing month
Uses a curated database of crop–soil compatibility and maturity durations.

✅ 3. Predicts Only for Recommended Crops
For each recommended crop:

Calculates harvest month

Retrieves last 6 modal prices

Predicts harvest-time price using the trained LSTM model

# How It Works
 
User Input:
District, soil type, and sowing month are collected via terminal.

Recommendation:
Filters crops suited for the soil and sowing month using a knowledge base.

Price Forecasting:
For each valid crop:

Calculates harvest month

Fetches last 6 modal prices (before harvest)

Runs prediction via LSTM model

Output:
Displays crop names with predicted modal prices at harvest.


├── models/    
   ├── unified_lstm_model.h5
   
   ├── crop_encoder.pkl
   
   ├── district_encoder.pkl
   
   └── price_scaler.pkl

├── data/ 
   ├── Commodities_price.xlsx
   
   ├── Crop.xlsx
   
   └── District_Mapping.xlsx

├── scripts/                            
   ├── train_model.py  
   
   ├── predict_prices.py
   
   └── crop_recommendation.py         
                             
├── README.md                         
├── requirements.txt

# requirements
streamlit==1.35.0

pandas

numpy

tensorflow==2.14.0

scikit-learn==1.3.2

openpyxl==3.1.2

joblib==1.3.2

python-dateutil==2.9.0

Tap on 'Open in Streamlikt' button to use the web application

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://krishi-jnan-mxvlhad4zmbyyb89js8twa.streamlit.app/)
## Author
Kirana BV

kiranaish15@gmail.com

linkedin.com/in/kirana-BV0106



