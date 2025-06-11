import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import sys

# Dependency check and import with fallbacks
try:
    import matplotlib.pyplot as plt
    CAN_PLOT = True
except ImportError:
    CAN_PLOT = False
    print("Note: matplotlib not available - plotting disabled")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Note: TensorFlow not available - falling back to scikit-learn")
    from sklearn.neural_network import MLPClassifier

def load_data(filepath):
    """Robust data loading function"""
    try:
        df = pd.read_csv(filepath)
        
        # Handle different dataset formats
        if 'Status (Y)' in df.columns:  # First dataset format
            df.columns = ['property', 'debt', 'status']
        elif 'status' in df.columns:    # Second dataset format
            df = df[['property_value', 'debt_value', 'status']]
            df.columns = ['property', 'debt', 'status']
            df['status'] = df['status'].apply(lambda x: 1 if x == 'Healthy' else 0)
        else:
            raise ValueError("Unrecognized dataset format")
            
        return df[['property', 'debt']].values, df['status'].values
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        sys.exit(1)

def train_and_save_model(X_train, y_train):
    """Train model with available framework"""
    if TF_AVAILABLE:
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(0.001),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_split=0.2,
            verbose=0
        )
        
        if CAN_PLOT:
            try:
                plt.figure(figsize=(10, 5))
                plt.plot(history.history['accuracy'], label='Train Accuracy')
                plt.plot(history.history['val_accuracy'], label='Val Accuracy')
                plt.legend()
                plt.savefig('training_history.png')
                plt.close()
            except:
                print("Could not generate training plot")
                
        model.save('model.h5')
        return model
        
    else:  # Fallback to scikit-learn
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, 'model.pkl')
        return model

def main():
    # Load and prepare data
    X, y = load_data("ann_stock_market_data.csv")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_and_save_model(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    if TF_AVAILABLE:
        y_pred = (y_pred > 0.5).astype(int)
    
    print(f"\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print(f"Model saved successfully")

if __name__ == "__main__":
    main()
