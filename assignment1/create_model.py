import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Load data
try:
    df = pd.read_csv('assignment1/athletes.csv')
except FileNotFoundError:
    try:
        df = pd.read_csv('athletes.csv')
    except FileNotFoundError:
        raise FileNotFoundError("The file 'athletes.csv' was not found in the current directory.")
print("READY TO GO")
# Drop rows with missing target
df = df.dropna(subset=['total_lift'])

# Separate features and target
y = df['total_lift']
X = df.drop(columns=['total_lift', 'snatch', 'deadlift', 'backsq', 'candj'])
optional_drop_cols = set(['train', 'affiliate', 'team', 'name'])
if X.columns.intersection(optional_drop_cols).any():
    X = X.drop(columns=optional_drop_cols)


# Fill missing values for categorical columns with mode
cat_cols = ['gender']
for col in cat_cols:
    mode = X[col].mode()[0]
    # print(mode, col)
    X[col] = X[col].fillna(mode)

# Fill missing values for numeric columns with mean
num_cols = ['age', 'height', 'weight']
X[num_cols] = X[num_cols].fillna(X[num_cols].mean())

# Handle categorical variables (one-hot encoding)
dummies = pd.get_dummies(X[cat_cols], drop_first=True)
X = X.drop(columns=cat_cols)
X = pd.concat([X, dummies], axis=1)

attributes = [*num_cols, *dummies.columns.tolist()]
X = X[attributes]

print("Filled missing values and encoded categoricals")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=24)
print("Train/test split completed")
print(X.shape, y.shape)

# Train MLPRegressor model
model = MLPRegressor(hidden_layer_sizes=(100,50,25,), max_iter=1000,
                     random_state=24, batch_size=32)
model.fit(X_train, y_train)
print("Model training completed")

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'R^2 score: {r2:.3f}')
print(f'RMSE: {rmse:.3f}')
