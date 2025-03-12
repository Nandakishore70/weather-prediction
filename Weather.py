#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


import pickle

# Define your trained model (replace with your actual model variable)
trained_model = rf_model  # Example: Change rf_model to the actual model variable

# Save the model as a pickle file
with open("weather_model.pkl", "wb") as file:
    pickle.dump(trained_model, file)

print("Model successfully saved as weather_model.pkl")


# In[21]:


weather = pd.read_csv("/kaggle/input/indianweatherrepository/IndianWeatherRepository.csv")


# In[22]:


weather.head()


# In[23]:


weather.info()


# In[24]:


weather.describe()


# In[25]:


weather['humidity']


# In[26]:


weather['wind_direction']


# In[27]:


weather['wind_direction'].nunique()


# In[28]:


weather['wind_direction'].value_counts()


# In[29]:


weather['condition_text'].nunique()


# In[30]:


weather['condition_text'].value_counts()


# In[31]:


weather_num = weather.select_dtypes(include=[np.number])
weather_num.head()


# In[32]:


weather_non_num = weather.select_dtypes(exclude=[np.number])
weather_non_num.head()


# In[33]:


weather_non_num['moon_phase'].nunique()


# In[34]:


weather_non_num['moon_phase'].value_counts()


# In[35]:


weather_num.drop(columns=["temperature_fahrenheit", "feels_like_fahrenheit",
                          "precip_in", "pressure_in", "visibility_miles",
                          "last_updated_epoch"], inplace=True)


# In[36]:


weather_non_num.drop(columns=[
    "country", "location_name", "region", "timezone",
    "last_updated", "sunrise", "sunset",
    "moonrise", "moonset"
], inplace=True)


# In[37]:


import matplotlib.pyplot as plt

weather.hist(bins=50,figsize=(24,24))
plt.show()


# In[38]:


weather.plot(kind="scatter",x="longitude",y="latitude",grid=True)
plt.show()


# In[39]:


weather['humidity'].hist(bins=200, figsize=(8, 6))
plt.show()


# In[40]:


for col in weather_num:
    plt.figure()  # Create a new figure for each histogram
    weather[col].hist(bins=50, edgecolor='black')
    # plt.title(col)
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[41]:


corr_matrix = weather.corr(numeric_only=True)


# In[42]:


corr_matrix["humidity"].sort_values(ascending=False)


# In[43]:


from pandas.plotting import scatter_matrix

p_attributes = ["humidity","cloud","longitude","precip_mm","feels_like_celsius"]

scatter_matrix(weather[p_attributes],figsize=(12,12))
plt.show()


# In[44]:


n_attributes = ["humidity","latitude","air_quality_PM2.5","air_quality_PM10","gust_kph","pressure_mb","air_quality_Ozone","visibility_km"]
scatter_matrix(weather[n_attributes],figsize=(18,18))
plt.show()


# In[45]:


weather.plot(kind="scatter",x="cloud",y="humidity",alpha=0.1,grid=True)
plt.show()


# In[46]:


weather["cloud"].corr(weather["humidity"], method="spearman")


# In[47]:


weather.plot.hexbin(x="cloud", y="humidity", gridsize=50, cmap="Blues")
plt.show()


# In[48]:


import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

# Extract relevant columns
x = weather["cloud"]
y = weather["humidity"]

# Apply Lowess smoothing
lowess_result = lowess(y, x, frac=0.3)  # Adjust frac for more or less smoothing

# Plot the scatter points
plt.figure(figsize=(10, 7))
plt.scatter(x, y, alpha=0.1, color="orange", s=5)  # Scatter plot

# Plot the Lowess regression line
plt.plot(lowess_result[:, 0], lowess_result[:, 1], color="darkorange", linewidth=2.5)

# Labels and title
plt.xlabel("Cloud Cover (%)")
plt.ylabel("Humidity (%)")
plt.title("Lowess Regression: Humidity vs Cloud")
plt.show()


# In[49]:


# Apply Meteorological-Based Binning for cloud in weather_num
bins = [0, 10, 30, 60, 90, 100]  # Corrected bin boundaries
labels = ["Clear", "Partly Cloudy", "Mostly Cloudy", "Overcast", "Full Cover"]

# Create the cloud category column in weather_num
weather_num["cloud_category"] = pd.cut(
    weather_num["cloud"], bins=bins, labels=labels, include_lowest=True
)

# Display the distribution of categories
cloud_category_counts = weather_num["cloud_category"].value_counts()

# Plot the distribution of cloud categories
plt.figure(figsize=(6, 4))
cloud_category_counts.plot(kind="bar")
plt.xlabel("Cloud Cover Category")
plt.ylabel("Frequency")
plt.title("Distribution of Cloud Cover Categories")
plt.xticks(rotation=45)
plt.show()

# Show category counts
cloud_category_counts


# In[50]:


# Define Meteorological-Based Binning
bins = [0, 10, 30, 60, 90, 100]  # Bin boundaries
labels = [1, 2, 3, 4, 5]  # Use numeric labels instead of strings

# Apply binning and label encoding
weather_num["cloud_category"] = pd.cut(
    weather_num["cloud"], bins=bins, labels=labels, include_lowest=True
).astype(int)  # Convert to integers

# Display the distribution of categories
cloud_category_counts = weather_num["cloud_category"].value_counts().sort_index()

# Plot the distribution of cloud categories
plt.figure(figsize=(6, 4))
cloud_category_counts.plot(kind="bar")
plt.xlabel("Cloud Cover Category (Encoded)")
plt.ylabel("Frequency")
plt.title("Distribution of Cloud Cover Categories")
plt.xticks(rotation=0)  # Keep labels upright
plt.show()

# Display the dataframe using standard Pandas print
print(cloud_category_counts.to_frame())

# Or, if using Jupyter Notebook, display it as a table
import pandas as pd
from IPython.display import display

display(cloud_category_counts.to_frame())


# In[51]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
weather_num["longitude_scaled"] = scaler.fit_transform(weather_num[["longitude"]])


# In[52]:


# Create a new figure with properly assigned axes
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Original longitude distribution
weather_num["longitude"].hist(ax=axs[0], bins=50, color="blue", alpha=0.7)
axs[0].set_xlabel("Longitude")
axs[0].set_ylabel("Frequency")
axs[0].set_title("Original Longitude Distribution")

# Standardized longitude distribution
weather_num["longitude_scaled"].hist(ax=axs[1], bins=50, color="orange", alpha=0.7)
axs[1].set_xlabel("Standardized Longitude")
axs[1].set_title("Z-score Standardized Longitude")

plt.tight_layout()
plt.show()


# In[53]:


# Apply log1p transformation for precip_mm
weather_num["precip_mm_log"] = np.log1p(weather_num["precip_mm"])

# Visualizing before and after Log Transformation
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Original precip_mm distribution
weather_num["precip_mm"].hist(ax=axs[0], bins=50, color="blue", alpha=0.7)
axs[0].set_xlabel("Precipitation (mm)")
axs[0].set_ylabel("Frequency")
axs[0].set_title("Original Precipitation Distribution")

# Log-transformed precip_mm distribution
weather_num["precip_mm_log"].hist(ax=axs[1], bins=50, color="orange", alpha=0.7)
axs[1].set_xlabel("Log of Precipitation (mm)")
axs[1].set_title("Log-Transformed Precipitation Distribution")

plt.tight_layout()
plt.show()


# In[54]:


weather['precip_mm'].max()


# In[55]:


weather['precip_mm'].value_counts()


# In[56]:


# Convert to binary: 1 if precipitation > 0, else 0
weather_num["precip_binary"] = weather_num["precip_mm"].apply(lambda x: 1 if x > 0 else 0)

fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

# Original precipitation distribution
weather_num["precip_mm"].hist(ax=axs[0], bins=50)
axs[0].set_xlabel("precip_mm")
axs[0].set_ylabel("Frequency")
axs[0].set_title("Original Precipitation Data")

# Binary (Rain vs No Rain) distribution
weather_num["precip_binary"].value_counts().plot(kind="bar", ax=axs[1])
axs[1].set_xlabel("Precipitation Binary (0 = No Rain, 1 = Rain)")
axs[1].set_ylabel("Frequency")
axs[1].set_xticks([0, 1])
axs[1].set_xticklabels(["No Rain", "Rain"])
axs[1].set_title("Binary Encoding (Rain vs No Rain)")

plt.show()


# In[57]:


weather_num[["precip_mm", "precip_binary"]].head()


# In[58]:


weather.plot(kind="scatter",x="visibility_km",y="humidity",alpha=0.1,grid=True)
plt.show()


# In[59]:


# Apply log1p transformation for visibility_km
weather_num["visibility_km_log"] = np.log1p(weather_num["visibility_km"])

# Visualizing before and after Log Transformation
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Original visibility_km distribution
weather_num["visibility_km"].hist(ax=axs[0], bins=50, color="blue", alpha=0.7)
axs[0].set_xlabel("Visibility (km)")
axs[0].set_ylabel("Frequency")
axs[0].set_title("Original Visibility Distribution")

# Log-transformed visibility_km distribution
weather_num["visibility_km_log"].hist(ax=axs[1], bins=50, color="orange", alpha=0.7)
axs[1].set_xlabel("Log of Visibility (km)")
axs[1].set_title("Log-Transformed Visibility Distribution")

plt.tight_layout()
plt.show()


# In[60]:


# # Create bins with intervals of 1: [0, 1, 2, ..., 10]
# bins = np.arange(0, 11, 1)  # This creates the array: [0, 1, 2, ..., 10]

# # Use pd.cut to categorize the 'visibility_km' column into these bins
# # labels=False returns integer bin indices (0 for [0,1], 1 for (1,2], etc.)
# weather['visibility_bin'] = pd.cut(weather['visibility_km'], bins=bins, labels=False, include_lowest=True)

# # Plot the histogram (bar chart) of the binned data
# weather['visibility_bin'].value_counts().sort_index().plot.bar(rot=0, grid=True)
# plt.xlabel("Visibility Category (Bin Index)")
# plt.ylabel("Frequency")
# plt.title("Histogram of Visibility Bins (0-10, interval 1)")
# plt.show()


# In[61]:


weather.plot(kind="scatter",x="air_quality_Ozone",y="humidity",alpha=0.2,grid=True)
plt.show()


# In[62]:


weather['air_quality_Ozone'].max()


# In[63]:


# Apply log1p transformation for air_quality_Ozone
weather_num["air_quality_Ozone_log"] = np.log1p(weather_num["air_quality_Ozone"])

# Visualizing before and after Log Transformation
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Original air_quality_Ozone distribution
weather_num["air_quality_Ozone"].hist(ax=axs[0], bins=50, color="blue", alpha=0.7)
axs[0].set_xlabel("Air Quality Ozone")
axs[0].set_ylabel("Frequency")
axs[0].set_title("Original Ozone Distribution")

# Log-transformed air_quality_Ozone distribution
weather_num["air_quality_Ozone_log"].hist(ax=axs[1], bins=50, color="orange", alpha=0.7)
axs[1].set_xlabel("Log of Air Quality Ozone")
axs[1].set_title("Log-Transformed Ozone Distribution")

plt.tight_layout()
plt.show()


# In[64]:


weather.plot(kind="scatter",x="pressure_mb",y="humidity",alpha=0.2,grid=True)
plt.show()


# In[65]:


# Apply StandardScaler (Z-score Normalization) for pressure_mb
scaler = StandardScaler()
weather_num["pressure_mb_scaled"] = scaler.fit_transform(weather_num[["pressure_mb"]])

# Create a new figure with properly assigned axes
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# Original pressure_mb distribution
weather_num["pressure_mb"].hist(ax=axs[0], bins=50, color="blue", alpha=0.7)
axs[0].set_xlabel("Pressure (mb)")
axs[0].set_ylabel("Frequency")
axs[0].set_title("Original Pressure Distribution")

# Standardized pressure_mb distribution
weather_num["pressure_mb_scaled"].hist(ax=axs[1], bins=50, color="orange", alpha=0.7)
axs[1].set_xlabel("Standardized Pressure (mb)")
axs[1].set_title("Z-score Standardized Pressure")

plt.tight_layout()
plt.show()


# In[66]:


weather.plot(kind="scatter",x="wind_kph",y="humidity",alpha=0.2,grid=True)
plt.show()


# In[67]:


weather.plot(kind="scatter",x="gust_kph",y="humidity",alpha=0.2,grid=True)
plt.show()


# In[68]:


# Apply log1p transformation for wind_kph and gust_kph
weather_num["wind_kph_log"] = np.log1p(weather_num["wind_kph"])
weather_num["gust_kph_log"] = np.log1p(weather_num["gust_kph"])

# Visualizing before and after Log Transformation
fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharey='row')

# Original wind_kph distribution
weather_num["wind_kph"].hist(ax=axs[0, 0], bins=50, color="blue", alpha=0.7)
axs[0, 0].set_xlabel("Wind Speed (kph)")
axs[0, 0].set_ylabel("Frequency")
axs[0, 0].set_title("Original Wind Speed Distribution")

# Log-transformed wind_kph distribution
weather_num["wind_kph_log"].hist(ax=axs[0, 1], bins=50, color="orange", alpha=0.7)
axs[0, 1].set_xlabel("Log of Wind Speed (kph)")
axs[0, 1].set_title("Log-Transformed Wind Speed")

# Original gust_kph distribution
weather_num["gust_kph"].hist(ax=axs[1, 0], bins=50, color="blue", alpha=0.7)
axs[1, 0].set_xlabel("Gust Speed (kph)")
axs[1, 0].set_ylabel("Frequency")
axs[1, 0].set_title("Original Gust Speed Distribution")

# Log-transformed gust_kph distribution
weather_num["gust_kph_log"].hist(ax=axs[1, 1], bins=50, color="orange", alpha=0.7)
axs[1, 1].set_xlabel("Log of Gust Speed (kph)")
axs[1, 1].set_title("Log-Transformed Gust Speed")

plt.tight_layout()
plt.show()


# In[69]:


weather_num.head()


# In[70]:


weather_num.info()


# In[71]:


weather_num.drop(columns=["wind_mph", "gust_mph"], inplace=True)


# In[72]:


weather_num.info()


# In[73]:


weather_non_num.head()


# In[74]:


weather_non_num.nunique()


# In[75]:


weather_non_num.value_counts()


# In[76]:


weather_non_num['condition_text'].nunique()


# In[77]:


weather_non_num['condition_text'].value_counts()


# In[78]:


# Convert condition_text to lowercase and strip extra spaces
weather_non_num["condition_text"] = weather_non_num["condition_text"].str.lower().str.strip()

# Display updated value counts of condition_text
condition_text_counts = weather_non_num["condition_text"].value_counts()


# In[79]:


weather_non_num["condition_text"].value_counts()


# In[80]:


# Define a mapping to group similar weather conditions into broader categories
condition_mapping = {
    "patchy rain possible": "rain",
    "light rain": "rain",
    "moderate rain": "rain",
    "heavy rain": "rain",
    "patchy snow possible": "snow",
    "light snow": "snow",
    "moderate snow": "snow",
    "heavy snow": "snow",
    "patchy sleet possible": "sleet",
    "light sleet": "sleet",
    "moderate sleet": "sleet",
    "heavy sleet": "sleet",
    "mist": "fog",
    "fog": "fog",
    "freezing fog": "fog",
    "sunny": "clear",
    "clear": "clear",
    "partly cloudy": "cloudy",
    "cloudy": "cloudy",
    "overcast": "cloudy",
    "thundery outbreaks possible": "thunderstorm",
    "patchy light drizzle": "rain",
    "light drizzle": "rain",
    "moderate or heavy drizzle": "rain",
    "patchy light rain with thunder": "thunderstorm",
    "moderate or heavy rain with thunder": "thunderstorm",
    "patchy moderate snow": "snow",
    "patchy heavy snow": "snow",
    "moderate or heavy snow showers": "snow",
    "ice pellets": "sleet",
    "light rain shower": "rain",
    "moderate or heavy rain shower": "rain",
    "torrential rain shower": "rain",
    "light sleet showers": "sleet",
    "moderate or heavy sleet showers": "sleet",
    "light snow showers": "snow",
    "moderate or heavy snow showers": "snow",
    "blowing snow": "snow",
    "blizzard": "snow"
}

# Apply mapping to simplify the condition_text column
weather_non_num["condition_text"] = weather_non_num["condition_text"].replace(condition_mapping)

# Display updated value counts after merging similar categories
condition_text_counts_updated = weather_non_num["condition_text"].value_counts()

# Show the counts in a standard way
print(condition_text_counts_updated)


# In[81]:


weather_non_num['condition_text'].nunique()


# In[82]:


weather_non_num.nunique()


# In[83]:


from sklearn.preprocessing import LabelEncoder

# Apply Label Encoding to condition_text
label_encoder = LabelEncoder()
weather_non_num["condition_text_encoded"] = label_encoder.fit_transform(weather_non_num["condition_text"])

# Save the mapping of encoded values for reference
condition_text_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Drop the original condition_text column if no longer needed
weather_non_num.drop(columns=["condition_text"], inplace=True)

# Display the first few rows to confirm encoding
weather_non_num.head()


# In[84]:


# Define an ordered mapping for wind directions
wind_direction_order = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]

# Create a dictionary mapping wind directions to ordinal values
wind_direction_mapping = {direction: idx for idx, direction in enumerate(wind_direction_order)}

# Apply ordinal encoding
weather_non_num["wind_direction_encoded"] = weather_non_num["wind_direction"].map(wind_direction_mapping)

# Drop the original wind_direction column if no longer needed
weather_non_num.drop(columns=["wind_direction"], inplace=True)

# Display the first few rows to confirm encoding
weather_non_num.head()


# In[85]:


# Apply One-Hot Encoding to moon_phase using pandas get_dummies
weather_non_num = pd.get_dummies(weather_non_num, columns=["moon_phase"], prefix="moon")


# In[86]:


weather_non_num.head()


# In[87]:


weather_num.head()


# In[88]:


weather_num.info()


# In[89]:


weather_num.drop(columns=[
    "longitude", "precip_mm", "visibility_km", "air_quality_Ozone", "pressure_mb", "wind_kph", "gust_kph"
], inplace=True)


# In[90]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = ["air_quality_Carbon_Monoxide", "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide",
                   "air_quality_PM2.5", "air_quality_PM10", "wind_degree", "uv_index",
                   "moon_illumination", "temperature_celsius","feels_like_celsius", "humidity", "cloud","latitude","air_quality_us-epa-index","air_quality_gb-defra-index"]

weather_num[scaled_features] = scaler.fit_transform(weather_num[scaled_features])


# In[91]:


weather_num.head()


# In[92]:


weather_num.info()


# In[93]:


# Merge numerical and categorical datasets
weather_final = pd.concat([weather_num, weather_non_num], axis=1)

# Confirm the merge
print("Final dataset shape:", weather_final.shape)
weather_final.head()


# In[94]:


weather_final.to_csv("/content/weather_final.csv", index=False)


# In[95]:


# from google.colab import files
# files.download("/content/weather_final.csv")


# In[96]:


weather_final.info()


# In[97]:


from sklearn.model_selection import train_test_split

# Define features (X) and target variable (y)
X = weather_final.drop(columns=["humidity"])  # Features
y = weather_final["humidity"]  # Target variable

# Perform train-test split with stratification on cloud_category
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=weather_final["cloud_category"], random_state=42
)

# Confirm split sizes
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[98]:


X_test['cloud_category'].value_counts() / len(X_test)


# In[99]:


print(0.541716 + 0.181741 + 0.119992 + 0.114413 + 0.042139)


# In[100]:


# from sklearn.ensemble import RandomForestRegressor
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # Train a simple RandomForest model for feature importance
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Extract feature importances
# feature_importance = rf_model.feature_importances_
# features = X_train.columns

# # Create a DataFrame for visualization
# importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
# importance_df = importance_df.sort_values(by='Importance', ascending=False)

# # Plot feature importance
# plt.figure(figsize=(10, 6))
# plt.barh(importance_df['Feature'], importance_df['Importance'], color='royalblue')
# plt.xlabel("Feature Importance Score")
# plt.ylabel("Features")
# plt.title("Feature Importance using RandomForest")
# plt.gca().invert_yaxis()  # Highest importance at top
# plt.show()


# In[101]:


# # Display top 10 most important features
# importance_df.head(10)


# In[102]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions
y_train_pred = lr_model.predict(X_train)
y_test_pred = lr_model.predict(X_test)

# Compute Mean Squared Error (Loss)
train_loss = mean_squared_error(y_train, y_train_pred)
test_loss = mean_squared_error(y_test, y_test_pred)

# Compute R² score as accuracy
train_acc = r2_score(y_train, y_train_pred) * 100  # Convert to percentage
test_acc = r2_score(y_test, y_test_pred) * 100  # Convert to percentage

# Print results
print(f"Train loss: {train_loss:.5f}, Train Acc: {train_acc:.2f}%")
print(f"Test loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")


# In[103]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Define a range of tree depths to evaluate
max_depths = range(1, 21)  # Trying depths from 1 to 20
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Iterate over different tree depths
for depth in max_depths:
    # Initialize Decision Tree Regressor with the current depth
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate Mean Squared Error (Loss)
    train_loss = mean_squared_error(y_train, y_train_pred)
    test_loss = mean_squared_error(y_test, y_test_pred)

    # Calculate R² Score (Accuracy)
    train_acc = r2_score(y_train, y_train_pred) * 100  # Convert to percentage
    test_acc = r2_score(y_test, y_test_pred) * 100  # Convert to percentage

    # Store results
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    # Print results per depth
    print(f"Depth: {depth} | Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")


# In[104]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100,  # Number of trees in the forest
                                 max_depth=None,  # No depth limit (can cause overfitting)
                                 random_state=42,
                                 n_jobs=-1)  # Use all available CPU cores

# Train the model
rf_model.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculate MSE Loss
train_loss = mean_squared_error(y_train, y_train_pred)
test_loss = mean_squared_error(y_test, y_test_pred)

# Calculate Accuracy (R² Score)
train_acc = r2_score(y_train, y_train_pred) * 100  # Convert to percentage
test_acc = r2_score(y_test, y_test_pred) * 100  # Convert to percentage

# Print results
print(f"Train loss: {train_loss:.5f}, Train Acc: {train_acc:.2f}%")
print(f"Test loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")


# In[105]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100,  # Number of boosting stages
                                     learning_rate=0.1,  # Step size shrinkage
                                     max_depth=3,  # Limits the depth of trees
                                     random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = gb_model.predict(X_train)
y_test_pred = gb_model.predict(X_test)

# Calculate MSE Loss
train_loss = mean_squared_error(y_train, y_train_pred)
test_loss = mean_squared_error(y_test, y_test_pred)

# Calculate Accuracy (R² Score)
train_acc = r2_score(y_train, y_train_pred) * 100  # Convert to percentage
test_acc = r2_score(y_test, y_test_pred) * 100  # Convert to percentage

# Print results
print(f"Train loss: {train_loss:.5f}, Train Acc: {train_acc:.2f}%")
print(f"Test loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")


# In[106]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100,   # Number of trees (boosting rounds)
                             learning_rate=0.1,  # Step size shrinkage
                             max_depth=3,        # Maximum depth of a tree
                             objective="reg:squarederror",  # Use squared error loss
                             random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on training and test sets
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# Calculate MSE Loss
train_loss = mean_squared_error(y_train, y_train_pred)
test_loss = mean_squared_error(y_test, y_test_pred)

# Calculate Accuracy (R² Score)
train_acc = r2_score(y_train, y_train_pred) * 100  # Convert to percentage
test_acc = r2_score(y_test, y_test_pred) * 100  # Convert to percentage

# Print results
print(f"Train loss: {train_loss:.5f}, Train Acc: {train_acc:.2f}%")
print(f"Test loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")


# # **CNN**

# In[109]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import numpy as np

# Normalize the data (CNNs need normalized inputs)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for Conv1D: (samples, time_steps, features)
X_train_cnn = np.expand_dims(X_train_scaled, axis=2)
X_test_cnn = np.expand_dims(X_test_scaled, axis=2)

# Build the CNN model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    BatchNormalization(),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test),
                    epochs=100, batch_size=32, verbose=1)

# Evaluate the model
train_loss, train_mae = model.evaluate(X_train_cnn, y_train, verbose=0)
test_loss, test_mae = model.evaluate(X_test_cnn, y_test, verbose=0)

print(f"Train loss: {train_loss:.5f}, Train MAE: {train_mae:.2f}")
print(f"Test loss: {test_loss:.5f}, Test MAE: {test_mae:.2f}")

