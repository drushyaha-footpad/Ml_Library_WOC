import numpy as np
import pandas as pd
from ml_library.logistic_regression import LogisticRegression

def train_and_predict():
    print("Loading data...")
    # Load data
    train_df = pd.read_csv('data/binary/train_binary.csv')
    test_df = pd.read_csv('data/binary/test_binary.csv')

    # Prepare training data
    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    # Prepare test data
    X_test = test_df.values

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Initialize and train model
    print("Training Logistic Regression model...")
    # Initialize and train model
    print("Training Logistic Regression model...")
    # Using default parameters.
    
    # Feature Scaling
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    model = LogisticRegression(learning_rate=0.01, n_iterations=2000)
    model.fit(X_train_scaled, y_train)
    print("Training complete.")

    # Predict
    print("Predicting on test set...")
    predictions = model.predict(X_test_scaled)

    # Create submission file
    submission = pd.DataFrame({
        'Id': range(len(predictions)),
        'Prediction': predictions
    })

    print("Saving submission.csv...")
    submission.to_csv('submission.csv', index=False)
    print("Done!")

if __name__ == "__main__":
    train_and_predict()
