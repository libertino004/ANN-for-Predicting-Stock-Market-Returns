import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    print("TensorFlow is not installed. Please install it with: pip install tensorflow")
    exit()

def load_data(filepath):
    """Load and preprocess the dataset"""
    df = pd.read_csv(filepath)
    
    # Handle different file formats
    if 'Status (Y)' in df.columns:  # ann_stock_market_data.csv
        df.columns = ['property_value', 'debt_value', 'status']
    elif 'status' in df.columns:  # stock_market_ann_dataset.csv
        df = df[['property_value', 'debt_value', 'status']]
        df['status'] = df['status'].apply(lambda x: 1 if x == 'Healthy' else 0)
    else:
        raise ValueError("Unknown file format")
    
    X = df[['property_value', 'debt_value']].values
    y = df['status'].values
    
    return X, y

def build_model(input_shape):
    """Build and compile the ANN model"""
    model = Sequential([
        Dense(64, input_dim=input_shape, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def main():
    try:
        # Load data
        X, y = load_data("ann_stock_market_data.csv")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, "scaler.pkl")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42)
        
        # Build and train model
        model = build_model(X_train.shape[1])
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=16,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=10)],
            verbose=1
        )
        
        # Evaluate
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
        
        # Save model
        model.save("ann_stock_model.h5")
        print("Model trained and saved successfully.")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
