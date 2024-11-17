import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
dataset = pd.read_csv("car_price.csv")
# Check for missing values
print(dataset.isnull().sum())
print(dataset.info())
# Encode categorical variables
le = LabelEncoder()
dataset['Car_Name'] = le.fit_transform(dataset['Car_Name'])
dataset['Fuel_Type'] = le.fit_transform(dataset['Fuel_Type'])
dataset['Selling_type'] = le.fit_transform(dataset['Selling_type'])
dataset['Transmission'] = le.fit_transform(dataset['Transmission'])
# Features and target variable
X = dataset[["Car_Name", "Year", "Present_Price", "Driven_kms", "Fuel_Type", "Selling_type", "Transmission", "Owner"]]
Y = dataset["Selling_Price"]
# Visualize data
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Driven_kms", y="Selling_Price", data=dataset)
plt.title("Driven Kilometers vs Selling Price")
plt.show()
# Correlation heatmap
correlation = dataset.corr()
sns.heatmap(correlation, annot=True)
plt.show()
# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the model
model = RandomForestRegressor()
model.fit(X_train, Y_train)
# Make predictions
predictions = model.predict(X_test)
# Evaluate the model
rmse = mean_squared_error(Y_test, predictions, squared=False)
r2 = r2_score(Y_test, predictions)
print(f"RMSE: {rmse}, R-squared: {r2}")
new_data = {
    "Car_Name": "city",
    "Year": 2019,
    "Present_Price": 10.0,
    "Driven_kms": 5000,
    "Fuel_Type": "Petrol",
    "Selling_type": "Dealer",
    "Transmission": "Manual",
    "Owner": 0
}
new_data_encoded = [
    le.transform([new_data["Car_Name"]])[0],
    new_data["Year"],
    new_data["Present_Price"],
    new_data["Driven_kms"],
    le.transform([new_data["Fuel_Type"]])[0],
    le.transform([new_data["Selling_type"]])[0],
    le.transform([new_data["Transmission"]])[0],
    new_data["Owner"]
]
new_data_standardized = scaler.transform([new_data_encoded])
predicted_price = model.predict(new_data_standardized)
print("Predicted Selling Price:", predicted_price[0])


