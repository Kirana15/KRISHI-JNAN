# KRISHI-JNAN
*Krishi-Jñāna* is a smart agricultural web platform designed to empower farmers by recommending the most suitable crops based on their local conditions *and* predicting expected *market prices at harvest time*. This two-stage system blends soil science and data-driven forecasting to improve farm decisions and increase profitability.

1. Unified LSTM-based Price Prediction Model
  A single deep learning model built using LSTM (Long Short-Term Memory) network predicts modal prices for all crops across all districts.

2. Intelligent Crop Recommendation
  Recommends the most suitable crops based on:
  District
  Soil type available in that district
  Sowing month
The recommendation logic comes from a curated knowledge base with maturity periods and soil compatibility.

3. Predicts Prices Only for Recommended Crops
  For each recommended crop, the system:
  Estimates harvest month from maturity duration ,gathers the last 6 modal prices (up to harvest month), Predicts expected modal price using the trained LSTM
  This ensures predictions are practical and contextually accurate.

4. Easy-to-Use CLI Interface
  Users can interactively input their district, soil type, and sowing month.
  The tool shows a list of crops with predicted harvest-time prices.

# How It Works

1️ User Input
  The user is asked to provide three inputs via a terminal prompt:
  District – e.g., Bengaluru Urban
  Soil Type – Based on available soil types in that district
  Sowing Month – e.g., Jul
2️ Crop Recommendation
  Once the inputs are given:
  The system fetches all soil types available for the given district using get_soil_types_for_district.
   Using recommend_crops, it filters crops that:
   Grow well in that soil, suitable for the sowing month, are recognized in the training data.
   Only recommended crops go through to the prediction stage.

4️ Price Prediction Using LSTM
For each recommended crop:
These prices, along with crop and district information, are passed into the trained LSTM model.

The model returns a predicted modal price at harvest time.

5️ Output Display
The system prints a list of recommended crops with
