import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset"""
    df = pd.read_csv(filepath)
    
    # Check which dataset we're loading based on columns
    if 'Status (Y)' in df.columns:  # ann_stock_market_data.csv
        df.columns = ['property_value', 'debt_value', 'status']
        # Convert status to binary (assuming 1=Healthy, 0=At-Risk)
        y = df['status'].values
    elif 'status' in df.columns:  # stock_market_ann_dataset.csv
        # Convert status to binary (Healthy=1, others=0)
        df['status'] = df['status'].apply(lambda x: 1 if x == 'Healthy' else 0)
        y = df['status'].values
        # Select only property and debt values for consistency
        df = df[['property_value', 'debt_value', 'status']]
    
    X = df[['property_value', 'debt_value']].values
    
    return X, y

def build_model(input_shape):
    """Build and compile the ANN model"""
    model = Sequential([
        Dense(64, input_dim=input_shape, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Load data (choose one dataset)
    X, y = load_and_preprocess_data("ann_stock_market_data.csv")
    # X, y = load_and_preprocess_data("stock_market_ann_dataset.csv")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    joblib.dump(scaler, "scaler.pkl")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    
    # Build model
    model = build_model(X_train.shape[1])
    
    # Train with early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test Accuracy: {accuracy:.2%}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.show()
    
    # Save the trained model
    model.save("ann_stock_model.h5")
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
