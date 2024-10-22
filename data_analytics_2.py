# First, let's load the dataset again and perform the analyses as described in the report.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

# Load the dataset
file_path = './dataset.csv'
data = pd.read_csv(file_path)

# Display basic info about the dataset
data.info()

# Descriptive statistics
desc_stats = data.describe()

# Select only numerical columns for correlation calculation
numeric_data = data.select_dtypes(include='number')

# Correlation matrix
corr = numeric_data.corr()

# Display the correlation matrix
print(corr)

# Plot the correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Select features for regression
X = data[['App Sessions', 'Distance Travelled (km)']]
y = data['Calories Burned']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and calculate mean squared error (MSE)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Print the regression coefficients and MSE
regression_results = {
    "MSE": mse,
    "Coefficients": model.coef_,
    "Intercept": model.intercept_
}

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Calories Burned')
plt.ylabel('Predicted Calories Burned')
plt.title('Actual vs Predicted Calories Burned')
plt.show()

# Apply KMeans clustering on 'App Sessions' and 'Distance Travelled'
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Plot the clustering results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='App Sessions', y='Distance Travelled (km)', hue='Cluster', data=data, palette='Set1')
plt.title('User Clustering based on App Sessions and Distance Travelled')
plt.show()

desc_stats, regression_results
