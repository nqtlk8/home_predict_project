# Train Linear Regression Model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os # Thêm thư viện os để xử lý đường dẫn an toàn hơn

# Đặt tên cột mục tiêu vào một biến để dễ quản lý
TARGET_COLUMN = 'SalePrice'


def train_model():
    # Load training data
    # Đảm bảo đường dẫn này là chính xác
    file_path = 'data/processed/train_cleaned.csv' 
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file tại '{file_path}'. Vui lòng kiểm tra lại đường dẫn.")
        return

    train_df = pd.read_csv(file_path)
    
    # === Identify column types ===
    
    # Lỗi đã được sửa tại đây: Loại bỏ cột TARGET_COLUMN khỏi danh sách cột số.
    all_num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # LOẠI BỎ CỘT MỤC TIÊU KHỎI TẬP FEATURES
    num_cols = [col for col in all_num_cols if col != TARGET_COLUMN]
    
    # Các cột phân loại vẫn lấy từ DataFrame gốc
    cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numeric features used: {num_cols}")
    print(f"Categorical features used: {cat_cols}")
    
    # === Preprocessing (Không thay đổi) ===
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols), # Chỉ sử dụng num_cols KHÔNG có 'SalePrice'
            ('cat', categorical_transformer, cat_cols)
        ]
    )
    
    # Split features and target (Không thay đổi)
    X = train_df.drop(TARGET_COLUMN, axis=1)
    y = train_df[TARGET_COLUMN]
    
    # Create pipeline (Không thay đổi)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Train model
    model.fit(X, y)
    
    # Save model
    model_output_dir = 'models'
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, os.path.join(model_output_dir, 'linear_regression_model.pkl'))
    print(f"Model trained and saved as '{os.path.join(model_output_dir, 'linear_regression_model.pkl')}'") 

if __name__ == "__main__":
    from pathlib import Path
    print("Starting model training...")
    train_model()