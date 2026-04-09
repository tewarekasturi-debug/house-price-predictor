import pandas as pd
from xgboost import XGBRegressor
import pickle
data = pd.read_csv("data.csv")
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X, y)
pickle.dump(model, open("model.pkl", "wb"))
print("XGBoost model trained and saved!")