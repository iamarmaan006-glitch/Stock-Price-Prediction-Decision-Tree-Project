import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import datetime
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Decision Tree Classifier for Stock Price Prediction")
st.markdown("""
This application demonstrates how a Decision Tree Classifier can be used to predict stock price movements.
The model predicts whether the stock price will go **UP** (1) or **DOWN** (0) based on various technical indicators.
""")

# Function to generate synthetic stock data
@st.cache_data
def generate_stock_data(n_days=1000, start_date="2020-01-01"):
    """Generate synthetic stock data with technical indicators"""
    
    np.random.seed(42)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    # Generate stock price with random walk and some trend
    returns = np.random.normal(0.001, 0.02, n_days)
    price = 100 * np.exp(np.cumsum(returns))
    
    # Calculate technical indicators
    df = pd.DataFrame({
        'Date': dates,
        'Open': price * (1 + np.random.normal(0, 0.01, n_days)),
        'High': price * (1 + np.random.normal(0.02, 0.01, n_days)),
        'Low': price * (1 - np.random.normal(0.02, 0.01, n_days)),
        'Close': price,
        'Volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    # Ensure High is always highest and Low is always lowest
    df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
    df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Create target variable (1 if price goes up next day, 0 otherwise)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Remove last row (no target)
    df = df[:-1]
    
    return df

def calculate_technical_indicators(df):
    """Calculate technical indicators for stock analysis"""
    
    # Moving averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Price relative to moving averages
    df['Price_to_MA5'] = df['Close'] / df['MA5']
    df['Price_to_MA20'] = df['Close'] / df['MA20']
    df['Price_to_MA50'] = df['Close'] / df['MA50']
    
    # Returns
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_20d'] = df['Close'].pct_change(20)
    
    # Volatility (20-day rolling standard deviation of returns)
    df['Volatility'] = df['Return_1d'].rolling(window=20).std()
    
    # Volume indicators
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA5']
    
    # RSI (Relative Strength Index)
    df['RSI'] = calculate_rsi(df['Close'])
    
    # MACD
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Price range
    df['Day_Range'] = (df['High'] - df['Low']) / df['Close']
    
    return df

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Sidebar
st.sidebar.header("Model Parameters")

# Data generation parameters
n_days = st.sidebar.slider("Number of trading days", min_value=500, max_value=2000, value=1000, step=100)
test_size = st.sidebar.slider("Test size (%)", min_value=10, max_value=40, value=20, step=5) / 100

# Decision Tree parameters
max_depth = st.sidebar.slider("Max Depth", min_value=2, max_value=20, value=5, step=1)
min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=20, value=5, step=1)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=1, max_value=10, value=2, step=1)
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])

# Generate or load data
df = generate_stock_data(n_days=n_days)

# Display data overview
st.header("📊 Data Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Days", len(df))
col2.metric("Features", len(df.columns) - 3)  # Exclude Date, Close, Target
col3.metric("Avg Close Price", f"${df['Close'].mean():.2f}")
col4.metric("Target Distribution", f"{df['Target'].mean()*100:.1f}% Up")

# Show sample data
with st.expander("View Sample Data"):
    st.dataframe(df.head(10))

# Data visualization
st.header("📈 Stock Price Visualization")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Stock price with moving averages
axes[0, 0].plot(df['Date'], df['Close'], label='Close Price', linewidth=1)
axes[0, 0].plot(df['Date'], df['MA20'], label='20-day MA', linewidth=1)
axes[0, 0].plot(df['Date'], df['MA50'], label='50-day MA', linewidth=1)
axes[0, 0].set_title('Stock Price with Moving Averages')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Volume
axes[0, 1].bar(df['Date'], df['Volume'], alpha=0.7, width=0.8)
axes[0, 1].set_title('Trading Volume')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Volume')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: RSI
axes[1, 0].plot(df['Date'], df['RSI'], linewidth=1)
axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.5)
axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.5)
axes[1, 0].set_title('RSI (Relative Strength Index)')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('RSI')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Bollinger Bands
axes[1, 1].plot(df['Date'], df['Close'], label='Close Price', linewidth=1)
axes[1, 1].fill_between(df['Date'], df['BB_Upper'], df['BB_Lower'], alpha=0.3)
axes[1, 1].set_title('Bollinger Bands')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].legend(['Close Price', 'Bollinger Bands'])
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
st.pyplot(fig)

# Feature selection
st.header("🎯 Model Training")

# Select features for training
feature_columns = ['Price_to_MA5', 'Price_to_MA20', 'Price_to_MA50', 
                   'Return_1d', 'Return_5d', 'Return_20d',
                   'Volatility', 'Volume_Ratio', 'RSI', 'MACD', 
                   'BB_Position', 'Day_Range']

# Check if all features exist
available_features = [col for col in feature_columns if col in df.columns]
X = df[available_features].dropna()
y = df.loc[X.index, 'Target']

# Display feature information
st.subheader("Features Used")
st.write(f"Total features: {len(available_features)}")
st.write("Feature list:", available_features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Decision Tree model
st.subheader("Model Training")
with st.spinner("Training Decision Tree Classifier..."):
    dt_classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        random_state=42
    )
    dt_classifier.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = dt_classifier.predict(X_train_scaled)
    y_pred_test = dt_classifier.predict(X_test_scaled)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

# Display results
col1, col2 = st.columns(2)
col1.metric("Training Accuracy", f"{train_accuracy:.2%}")
col2.metric("Testing Accuracy", f"{test_accuracy:.2%}")

# Confusion Matrix
st.subheader("Confusion Matrix")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training confusion matrix
cm_train = confusion_matrix(y_train, y_pred_train)
sns.heatmap(cm_train, annot=True, fmt='d', ax=axes[0], cmap='Blues')
axes[0].set_title('Training Set')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Testing confusion matrix
cm_test = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm_test, annot=True, fmt='d', ax=axes[1], cmap='Blues')
axes[1].set_title('Testing Set')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
st.pyplot(fig)

# Classification Report
st.subheader("Classification Report (Test Set)")
report = classification_report(y_test, y_pred_test, target_names=['Down', 'Up'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# Feature Importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': dt_classifier.feature_importances_
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(feature_importance['feature'], feature_importance['importance'])
ax.set_xlabel('Importance')
ax.set_title('Feature Importance in Decision Tree')
ax.invert_yaxis()

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, feature_importance['importance'])):
    ax.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
            ha='left', va='center', fontweight='bold')

plt.tight_layout()
st.pyplot(fig)

# Visualize Decision Tree
st.subheader("Decision Tree Visualization")
st.markdown("Note: Tree visualization is limited to depth 3 for clarity")

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_classifier, max_depth=3, feature_names=available_features, 
          class_names=['Down', 'Up'], filled=True, rounded=True, 
          fontsize=10, ax=ax)
plt.title("Decision Tree Structure (Limited to depth 3)")
st.pyplot(fig)

# Prediction on latest data
st.header("🔮 Latest Prediction")
latest_features = X.iloc[-1:].values
latest_features_scaled = scaler.transform(latest_features)
prediction = dt_classifier.predict(latest_features_scaled)[0]
probability = dt_classifier.predict_proba(latest_features_scaled)[0]

col1, col2, col3 = st.columns(3)
col1.metric("Latest Close Price", f"${df['Close'].iloc[-1]:.2f}")
col2.metric("Prediction", "📈 UP" if prediction == 1 else "📉 DOWN")
col3.metric("Confidence", f"{max(probability)*100:.1f}%")

# Probability breakdown
st.subheader("Prediction Probability")
prob_df = pd.DataFrame({
    'Direction': ['Down', 'Up'],
    'Probability': probability
})
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(prob_df['Direction'], prob_df['Probability'], color=['red', 'green'])
ax.set_ylabel('Probability')
ax.set_title('Prediction Probability Distribution')
ax.set_ylim([0, 1])

# Add percentage labels
for bar, prob in zip(bars, probability):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{prob:.1%}', ha='center', va='bottom')

st.pyplot(fig)

# Model parameters summary
st.sidebar.header("Model Summary")
st.sidebar.write(f"**Training samples:** {len(X_train)}")
st.sidebar.write(f"**Testing samples:** {len(X_test)}")
st.sidebar.write(f"**Tree depth:** {dt_classifier.get_depth()}")
st.sidebar.write(f"**Number of leaves:** {dt_classifier.get_n_leaves()}")

# Download model predictions
if st.button("Download Predictions"):
    predictions_df = df.loc[X.index, ['Date', 'Close', 'Target']].copy()
    predictions_df['Predicted'] = dt_classifier.predict(X)
    predictions_df['Prediction_Probability_Down'] = dt_classifier.predict_proba(X)[:, 0]
    predictions_df['Prediction_Probability_Up'] = dt_classifier.predict_proba(X)[:, 1]
    
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="stock_predictions.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
**Note:** This is a demonstration project using synthetic data. The predictions should not be used for real trading decisions.
The model shows how Decision Tree classifiers can be applied to financial time series data for educational purposes.
""")