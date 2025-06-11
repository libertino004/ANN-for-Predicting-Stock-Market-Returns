import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Check and import visualization packages with error handling
try:
    import matplotlib.pyplot as plt
    CAN_PLOT = True
except ImportError:
    print("Warning: matplotlib not found. Plotting disabled.")
    CAN_PLOT = False

# Check and import TensorFlow with error handling
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    print("Warning: TensorFlow not found. Falling back to scikit-learn MLPClassifier.")
    from sklearn.neural_network import MLPClassifier
    HAS_TENSORFLOW = False

def load_data(filepath):
    """Load and preprocess the dataset"""
    try:
        df = pd.read_csv(filepath)
        
        # Handle different file formats
        if 'Status (Y)' in df.columns:  # ann_stock_market_data.csv
            df.columns = ['property_value', 'debt_value', 'status']
        elif 'status' in df.columns:  # stock_market_ann_dataset.csv
            df = df[['property_value', 'debt_value', 'status']]
            df['status'] = df['status'].apply(lambda x: 1 if x == 'Healthy' else 0)
        else:
            raise ValueError("Unknown file format")
        
        return df[['property_value', 'debt_value']].values, df['status'].values
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None

def build_model(input_shape=None):
    """Build and return appropriate model based on available packages"""
    if HAS_TENSORFLOW:
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
    else:
        model = MLPClassifier(hidden_layer_sizes=(64, 32),
                            activation='relu',
                            solver='adam',
                            random_state=42,
                            max_iter=200)
    return model

def main():
    # Load data
    X, y = load_data("ann_stock_market_data.csv")
    if X is None or y is None:
        return
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    
    # Build model
    model = build_model(input_shape=X_train.shape[1] if HAS_TENSORFLOW else None)
    
    # Train model
    if HAS_TENSORFLOW:
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=16,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=10)],
            verbose=1
        )
        
        if CAN_PLOT:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.legend()
            plt.savefig('training_history.png')
            plt.close()
    else:
        model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    if HAS_TENSORFLOW:
        y_pred = (y_pred > 0.5).astype(int)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    
    # Save model
    if HAS_TENSORFLOW:
        model.save("ann_stock_model.h5")
    else:
        joblib.dump(model, "mlp_classifier.pkl")
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    main()
