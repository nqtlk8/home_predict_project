# Evaluation Linear Regression Model
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
def evaluate_model():
    # Load validation data
    val_df = pd.read_csv('data/processed/val_cleaned.csv')
    
    # Split features and target
    X_val = val_df.drop('SalePrice', axis=1)
    y_val = val_df['SalePrice']
    
    # Load trained model
    model = joblib.load('models/linear_regression_model.pkl')
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Evaluate model
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"Model Evaluation on Validation Set:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2 ): {r2:.4f}")

if __name__ == "__main__":
    print("Starting model evaluation...")
    evaluate_model()