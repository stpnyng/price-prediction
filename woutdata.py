import os

#Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional, TimeDistributed
from tensorflow.keras.layers import MaxPooling1D, Flatten
from tensorflow.keras.regularizers import L2
from sklearn.metrics import explained_variance_score, r2_score, max_error
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import cifar10
import seaborn as sns
import io
import matplotlib.pyplot as plt
from scipy.signal import stft, butter, sosfilt
from sklearn.preprocessing import MinMaxScaler
import pywt
from tensorflow.keras import layers, models
from scipy.signal import butter, filtfilt
from scipy.signal import cwt, morlet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from io import StringIO

# Set styles for plotting
sns.set_style("whitegrid")
plt.style.use("ggplot")

# Bitcoin and Ethereum data

# Convert the data into DataFrames
bitcoin_df = pd.read_csv(io.StringIO(bitcoin_data))
ethereum_df = pd.read_csv(io.StringIO(ethereum_data))

# Convert 'date' column to datetime type for better plotting
bitcoin_df['date'] = pd.to_datetime(bitcoin_df['date'])
ethereum_df['date'] = pd.to_datetime(ethereum_df['date'])

# Plotting Bitcoin closing prices
plt.figure(figsize=(10,6))
plt.plot(bitcoin_df['date'], bitcoin_df['close'], label="Bitcoin", color="orange")
plt.title("Bitcoin Closing Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.show()

# Plotting Ethereum closing prices
plt.figure(figsize=(10,6))
plt.plot(ethereum_df['date'], ethereum_df['close'], label="Ethereum", color="blue")
plt.title("Ethereum Closing Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.show()

# Function to plot the original and transformed data
def plot_wavelet_results(original, cA, cD, title):
    plt.figure(figsize=(14, 8))

    # Plot original data
    plt.subplot(3, 1, 1)
    plt.plot(original, label=f'{title} Original Data')
    plt.title(f'{title} Original Data')
    plt.legend()

    # Plot DWT approximation coefficients
    plt.subplot(3, 1, 2)
    plt.plot(cA, label=f'{title} DWT Approximation Coefficients')
    plt.title(f'{title} DWT Approximation Coefficients')
    plt.legend()

    # Plot DWT detail coefficients
    plt.subplot(3, 1, 3)
    plt.plot(cD, label=f'{title} DWT Detail Coefficients')
    plt.title(f'{title} DWT Detail Coefficients')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Perform Discrete Wavelet Transform (DWT) on Bitcoin closing prices
bitcoin_close = bitcoin_df['close'].values
cA_btc, cD_btc = pywt.dwt(bitcoin_close, 'db1')  # Using Daubechies wavelet

# Plot Bitcoin wavelet results
plot_wavelet_results(bitcoin_close, cA_btc, cD_btc, "Bitcoin")

# Perform Discrete Wavelet Transform (DWT) on Ethereum closing prices
ethereum_close = ethereum_df['close'].values
cA_eth, cD_eth = pywt.dwt(ethereum_close, 'db1')  # Using Daubechies wavelet

# Plot Ethereum wavelet results
plot_wavelet_results(ethereum_close, cA_eth, cD_eth, "Ethereum")

# Load Bitcoin and Ethereum data into DataFrames
bitcoin_df = pd.read_csv(StringIO(bitcoin_data))
ethereum_df = pd.read_csv(StringIO(ethereum_data))

# Prepare the data by selecting the 'close' prices for both Bitcoin and Ethereum
bitcoin_close = bitcoin_df['close'].values
ethereum_close = ethereum_df['close'].values

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
bitcoin_close_scaled = scaler.fit_transform(bitcoin_close.reshape(-1, 1)).flatten()
ethereum_close_scaled = scaler.fit_transform(ethereum_close.reshape(-1, 1)).flatten()

# DWT (Discrete Wavelet Transform)
wavelet = 'db4'  # Daubechies wavelet of order 4

# Perform DWT on the Bitcoin data
coeffs_bitcoin = pywt.wavedec(bitcoin_close_scaled, wavelet, level=2)
cA_bitcoin, cD_bitcoin = coeffs_bitcoin[0], coeffs_bitcoin[1]

# Perform DWT on the Ethereum data
coeffs_ethereum = pywt.wavedec(ethereum_close_scaled, wavelet, level=2)
cA_ethereum, cD_ethereum = coeffs_ethereum[0], coeffs_ethereum[1]

# Plot original vs DWT-transformed data for Bitcoin and Ethereum
plot_wavelet_results(bitcoin_close_scaled, cA_bitcoin, cD_bitcoin, "Bitcoin")
plot_wavelet_results(ethereum_close_scaled, cA_ethereum, cD_ethereum, "Ethereum")


#Bandpass Filter
def apply_wavelet_bandpass_filter(coeffs, low_level, high_level):
    filtered_coeffs = coeffs.copy()
    for i in range(len(filtered_coeffs)):
        if i < low_level or i > high_level:
            filtered_coeffs[i] = np.zeros_like(filtered_coeffs[i])  # Zero out unwanted coefficients
    return filtered_coeffs

# Apply bandpass filter to Bitcoin and Ethereum
filtered_coeffs_bitcoin = apply_wavelet_bandpass_filter(coeffs_bitcoin, 1, 2)  # Keep levels 1 and 2 (bandpass)
filtered_coeffs_ethereum = apply_wavelet_bandpass_filter(coeffs_ethereum, 1, 2)  # Keep levels 1 and 2 (bandpass)

# Reconstruct the filtered signal
filtered_bitcoin = pywt.waverec(filtered_coeffs_bitcoin, wavelet)
filtered_ethereum = pywt.waverec(filtered_coeffs_ethereum, wavelet)

# Plot the original and bandpass filtered signals
plt.figure(figsize=(14, 8))

# Plot original Bitcoin data
plt.subplot(2, 1, 1)
plt.plot(bitcoin_close_scaled, label='Original Bitcoin Data')
plt.plot(filtered_bitcoin, label='Bandpass Filtered Bitcoin Data', linestyle='--')
plt.title('Bitcoin: Original vs Bandpass Filtered')
plt.legend()

# Plot original Ethereum data
plt.subplot(2, 1, 2)
plt.plot(ethereum_close_scaled, label='Original Ethereum Data')
plt.plot(filtered_ethereum, label='Bandpass Filtered Ethereum Data', linestyle='--')
plt.title('Ethereum: Original vs Bandpass Filtered')
plt.legend()

plt.tight_layout()
plt.show()

# Define function to create sequences of inputs and outputs
def create_sequences(data, window_size):
    X, Y = [], []
    for i in range(1, len(data) - window_size - 1, 1):
        temp = []
        temp2 = []
        for j in range(window_size):
            temp.append(data[i + j])
        temp2.append(data[i + window_size])
        X.append(np.array(temp).reshape(100, 1))
        Y.append(np.array(temp2).reshape(1, 1))
    return np.array(X), np.array(Y)

# Define function to process the data, train, and evaluate model for Bitcoin/Ethereum
def process_and_train(df, crypto_name):
    # Normalize the 'close' column
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['close']])

    # Create sequences of inputs and outputs
    window_size = 100
    X, Y = create_sequences(df_scaled, window_size)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

    # Reshape the data for the CNN-LSTM model
    train_X = x_train.reshape(x_train.shape[0], 1, 100, 1)
    test_X = x_test.reshape(x_test.shape[0], 1, 100, 1)

    # Create the Sequential model
    model = tf.keras.Sequential()

    # CNN layers
    model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', input_shape=(None, 100, 1))))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Conv1D(128, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Conv1D(64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(2)))
    model.add(TimeDistributed(Flatten()))

    # LSTM layers
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(100, return_sequences=False)))
    model.add(Dropout(0.5))

    # Final dense layer
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mse', 'mae']
    )

    # Train the model
    history = model.fit(train_X, np.array(y_train), validation_data=(test_X, np.array(y_test)), epochs=40, batch_size=40, verbose=1, shuffle=True)

    # Plot the loss, mse, and mae over epochs
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{crypto_name} - Loss Over Epochs')
    plt.show()

    plt.plot(history.history['mse'], label='Train MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.title(f'{crypto_name} - MSE Over Epochs')
    plt.show()

    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.title(f'{crypto_name} - MAE Over Epochs')
    plt.show()

    # Evaluate the model
    print(f"Evaluating {crypto_name} model...")
    model.evaluate(test_X, np.array(y_test))

    # Predict the test set
    yhat_probs = model.predict(test_X, verbose=0)
    yhat_probs = yhat_probs[:, 0]  # Flatten to 1D array

    # Calculate explained variance, R2 score, and max error
    var = explained_variance_score(np.array(y_test).reshape(-1, 1), yhat_probs)
    print(f'Explained Variance ({crypto_name}): {var}')

    r2 = r2_score(np.array(y_test).reshape(-1, 1), yhat_probs)
    print(f'R2 Score ({crypto_name}): {r2}')

    max_err = max_error(np.array(y_test).reshape(-1, 1), yhat_probs)
    print(f'Max Error ({crypto_name}): {max_err}')

    # Inverse transform the predictions and true values for comparison
    predicted = scaler.inverse_transform(model.predict(test_X))
    test_label = scaler.inverse_transform(np.array(y_test).reshape(-1, 1))
    predicted = np.array(predicted[:, 0]).reshape(-1, 1)

    # Plot the real vs predicted stock prices
    plt.plot(predicted, color='green', label='Predicted Stock Price')
    plt.plot(test_label, color='red', label='Real Stock Price')
    plt.title(f'{crypto_name} Stock Price Prediction')
    plt.xlabel('Day')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

    # Save the model
    model.save(f'{crypto_name}_model.h5')
    return model

# Process and train for both Bitcoin and Ethereum

# For Bitcoin
process_and_train(bitcoin_df, "Bitcoin")

# For Ethereum
process_and_train(ethereum_df, "Ethereum")

#Accuracy
# Step 1: Load and preprocess the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the images to a range of [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Optionally split off some of the training set to use for validation
X_valid = X_train[:5000]  # First 5000 samples for validation
y_valid = y_train[:5000]

X_train = X_train[5000:]  # Remaining samples for training
y_train = y_train[5000:]

# Step 2: Define the model architecture
model = models.Sequential()

# Example layers for a basic Convolutional Neural Network (CNN)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # CIFAR-10 has 10 classes

# Step 3: Compile the model with accuracy metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use for integer labels
              metrics=['accuracy'])

# Step 4: Print the model summary
model.summary()

# Step 5: Train the model
history = model.fit(X_train, y_train, 
                    epochs=1,  # Change this for more epochs
                    validation_data=(X_valid, y_valid))

# Step 6: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

##TRY
# Split the data into training and testing sets
# Assuming Bitcoin and Ethereum closing prices are the targets for testing
bitcoin_prices = bitcoin_df['close'].values
ethereum_prices = ethereum_df['close'].values

# Define the test size (e.g., 20% for testing)
test_size = 0.2

# Create indices for splitting
bitcoin_train, bitcoin_test = train_test_split(bitcoin_prices, test_size=test_size, random_state=42, shuffle=False)
ethereum_train, ethereum_test = train_test_split(ethereum_prices, test_size=test_size, random_state=42, shuffle=False)

# Display the split sizes
print(f"Bitcoin Training Size: {len(bitcoin_train)}, Testing Size: {len(bitcoin_test)}")
print(f"Ethereum Training Size: {len(ethereum_train)}, Testing Size: {len(ethereum_test)}")

# Optionally visualize the splits
plt.figure(figsize=(10, 6))

# Plot Bitcoin training and testing splits
plt.plot(range(len(bitcoin_train)), bitcoin_train, label="Bitcoin Training", color="orange")
plt.plot(range(len(bitcoin_train), len(bitcoin_train) + len(bitcoin_test)), bitcoin_test, label="Bitcoin Testing", color="red")
plt.title("Bitcoin Data Split")
plt.xlabel("Index")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.show()

# Plot Ethereum training and testing splits
plt.figure(figsize=(10, 6))
plt.plot(range(len(ethereum_train)), ethereum_train, label="Ethereum Training", color="blue")
plt.plot(range(len(ethereum_train), len(ethereum_train) + len(ethereum_test)), ethereum_test, label="Ethereum Testing", color="green")
plt.title("Ethereum Data Split")
plt.xlabel("Index")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.show()

####TRYYYYYY

# Import required libraries for wavelet transforms
import pywt

# Function to perform Discrete Wavelet Transform
def apply_dwt(data, wavelet='db1'):
    # Perform single-level DWT
    cA, cD = pywt.dwt(data, wavelet)
    return cA, cD

# Apply DWT to Bitcoin closing prices
bitcoin_close = bitcoin_df['close'].values
cA_btc, cD_btc = apply_dwt(bitcoin_close)

# Apply DWT to Ethereum closing prices
ethereum_close = ethereum_df['close'].values
cA_eth, cD_eth = apply_dwt(ethereum_close)

# Split DWT coefficients into training and testing sets
test_size = 0.2

# Split approximation coefficients (cA) for Bitcoin
cA_btc_train, cA_btc_test = train_test_split(cA_btc, test_size=test_size, random_state=42, shuffle=False)
# Split detail coefficients (cD) for Bitcoin
cD_btc_train, cD_btc_test = train_test_split(cD_btc, test_size=test_size, random_state=42, shuffle=False)

# Split approximation coefficients (cA) for Ethereum
cA_eth_train, cA_eth_test = train_test_split(cA_eth, test_size=test_size, random_state=42, shuffle=False)
# Split detail coefficients (cD) for Ethereum
cD_eth_train, cD_eth_test = train_test_split(cD_eth, test_size=test_size, random_state=42, shuffle=False)

# Display the sizes of splits
print("Bitcoin Approximation Coefficients:")
print(f"Training Size: {len(cA_btc_train)}, Testing Size: {len(cA_btc_test)}")
print("Bitcoin Detail Coefficients:")
print(f"Training Size: {len(cD_btc_train)}, Testing Size: {len(cD_btc_test)}")

print("Ethereum Approximation Coefficients:")
print(f"Training Size: {len(cA_eth_train)}, Testing Size: {len(cA_eth_test)}")
print("Ethereum Detail Coefficients:")
print(f"Training Size: {len(cD_eth_train)}, Testing Size: {len(cD_eth_test)}")

# Visualization of DWT splits
plt.figure(figsize=(10, 6))

# Plot Bitcoin DWT Approximation Coefficients Training and Testing Splits
plt.plot(range(len(cA_btc_train)), cA_btc_train, label="Bitcoin cA Training", color="orange")
plt.plot(range(len(cA_btc_train), len(cA_btc_train) + len(cA_btc_test)), cA_btc_test, label="Bitcoin cA Testing", color="red")
plt.title("Bitcoin Approximation Coefficients Split")
plt.xlabel("Index")
plt.ylabel("cA Value")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))

# Plot Ethereum DWT Approximation Coefficients Training and Testing Splits
plt.plot(range(len(cA_eth_train)), cA_eth_train, label="Ethereum cA Training", color="blue")
plt.plot(range(len(cA_eth_train), len(cA_eth_train) + len(cA_eth_test)), cA_eth_test, label="Ethereum cA Testing", color="green")
plt.title("Ethereum Approximation Coefficients Split")
plt.xlabel("Index")
plt.ylabel("cA Value")
plt.legend()
plt.show()


####TRIAL
test_data_bitcoin = bitcoin_df[-273:]
test_data_ethereum = ethereum_df[-273:]

# You can now use these DataFrames for analysis or plotting
print(test_data_bitcoin)
print(test_data_ethereum)

##############################
# Generate predicted prices ranging from 1 to 273
predicted_bitcoin_price = list(range(1, 274))  # Replace with Bitcoin predictions
predicted_ethereum_price = list(range(1, 274))  # Replace with Ethereum predictions

# Assuming the test datasets also have the same size (273 entries)
# Create dummy test datasets for Bitcoin and Ethereum
bitcoin_test_data = pd.DataFrame({
    'close': np.random.randint(20000, 30000, 273),  # Replace with actual Bitcoin test values
    'date': pd.date_range(start='2023-01-01', periods=273)  # Replace with Bitcoin test dates
})

ethereum_test_data = pd.DataFrame({
    'close': np.random.randint(1000, 2000, 273),  # Replace with actual Ethereum test values
    'date': pd.date_range(start='2023-01-01', periods=273)  # Replace with Ethereum test dates
})

# Extract real stock prices
real_bitcoin_price = bitcoin_test_data['close'].values
real_ethereum_price = ethereum_test_data['close'].values

# Plot Bitcoin real and predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(real_bitcoin_price, label='Bitcoin Real Price', color='blue')
plt.plot(predicted_bitcoin_price, label='Bitcoin Predicted Price', color='orange')
plt.title("Bitcoin Stock Price Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.show()

# Plot Ethereum real and predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(real_ethereum_price, label='Ethereum Real Price', color='green')
plt.plot(predicted_ethereum_price, label='Ethereum Predicted Price', color='red')
plt.title("Ethereum Stock Price Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price (USD)")
plt.legend()
plt.show()

# Print additional information about the test data and predictions for Bitcoin
print("Bitcoin Testing Data Size:", len(real_bitcoin_price))
print("Bitcoin Prediction Output Size:", len(predicted_bitcoin_price))
if 'date' in bitcoin_test_data:
    bitcoin_test_data_dates = bitcoin_test_data['date'].values
    print("Bitcoin Testing Dates:", bitcoin_test_data_dates)

# Print additional information about the test data and predictions for Ethereum
print("Ethereum Testing Data Size:", len(real_ethereum_price))
print("Ethereum Prediction Output Size:", len(predicted_ethereum_price))
if 'date' in ethereum_test_data:
    ethereum_test_data_dates = ethereum_test_data['date'].values
    print("Ethereum Testing Dates:", ethereum_test_data_dates)


#####$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Function to preprocess data
def preprocess_data(data, lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['close']])
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])
    X = np.array(X).reshape(-1, lookback, 1)  # Ensure the input shape is (batch_size, time_steps, features)
    return X, np.array(y), scaler

# Generate dummy datasets for Bitcoin and Ethereum
np.random.seed(42)
bitcoin_data = pd.DataFrame({
    'close': np.random.randint(30000, 50000, 300),
    'date': pd.date_range(start='2023-01-01', periods=300)
})

# Preprocess Bitcoin data
lookback = 60
X_btc, y_btc, btc_scaler = preprocess_data(bitcoin_data, lookback)

# Split data into train and test sets
split = int(len(X_btc) * 0.8)
X_btc_train, X_btc_test = X_btc[:split], X_btc[split:]
y_btc_train, y_btc_test = y_btc[:split], y_btc[split:]

# Define model architecture
def build_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(2))
    # Removed Flatten() to preserve the 3D shape required by LSTM layers
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(100, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    return model

# Train Bitcoin model
btc_model = build_model((lookback, 1))  # Correct input shape for Conv1D
history_btc = btc_model.fit(
    X_btc_train, y_btc_train,
    validation_data=(X_btc_test, y_btc_test),
    epochs=40, batch_size=32, verbose=1, shuffle=True
)

# Plot training history
plt.plot(history_btc.history['loss'], label='Train Loss')
plt.plot(history_btc.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Bitcoin - Loss Over Epochs')
plt.show()

# Make predictions
predicted_btc = btc_model.predict(X_btc_test)
predicted_btc = btc_scaler.inverse_transform(predicted_btc.reshape(-1, 1))
real_btc = btc_scaler.inverse_transform(y_btc_test.reshape(-1, 1))

# Plot real vs predicted Bitcoin prices
plt.plot(real_btc, label='Real Bitcoin Price', color='red')
plt.plot(predicted_btc, label='Predicted Bitcoin Price', color='green')
plt.title('Bitcoin Stock Price Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
