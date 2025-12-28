

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

df = pd.read_csv("/content/stores_sales_forecasting.csv", encoding='latin1')

df.columns = df.columns.str.lower()
df.rename(columns={'order date': 'date'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])


df = df.sort_values('date').reset_index(drop=True)

df['sales'] = df['sales'].interpolate(method='linear')


Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df['sales'] = np.clip(df['sales'], lower, upper)

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)

df['lag_1'] = df['sales'].shift(1)
df['lag_7'] = df['sales'].shift(7)
df['lag_14'] = df['sales'].shift(14)

df.dropna(inplace=True)


train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

features_to_drop = [
    'date', 'sales', 'order id', 'ship date', 'ship mode', 'customer id',
    'customer name', 'segment', 'country', 'city', 'state', 'region',
    'product id', 'category', 'sub-category', 'product name'
]

X_train = train.drop(features_to_drop, axis=1, errors='ignore')
y_train = train['sales']

X_test = test.drop(features_to_drop, axis=1, errors='ignore')
y_test = test['sales']


model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("Model Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")


plt.figure(figsize=(12, 6))
plt.plot(test['date'], y_test.values, label='Actual Sales', color='blue')
plt.plot(test['date'], predictions, label='Predicted Sales', color='red')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual vs Predicted Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()