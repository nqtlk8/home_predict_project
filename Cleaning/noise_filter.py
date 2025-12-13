import pandas as pd
import os

def handle_noise():
    # 1. Cấu hình đường dẫn
    input_path = os.path.join('data', 'processed', 'data_nodup.csv') # Hoặc đổi thành 'data/processed/data_nodup.csv' nếu muốn chạy nối tiếp
    output_folder = os.path.join('data', 'processed')
    output_path = os.path.join(output_folder, 'data_filled.csv')

    print(f"--- Đang bắt đầu xử lý nhiễu (Missing Values) ---")

    # 2. Kiểm tra file đầu vào
    if not os.path.exists(input_path):
        print(f"❌ Lỗi: Không tìm thấy file tại '{input_path}'")
        return

    df = pd.read_csv(input_path)

    # 3. Định nghĩa chiến lược xử lý
    # Nhóm 1: Thay thế bằng "None" (cho các đặc tính không có thật)
    cols_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu',
                      'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                      'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    
    # Nhóm 2: Thay thế bằng Median (cho các biến số)
    cols_fill_median = ['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                        'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt', 
                        'GarageCars', 'GarageArea']
    
    # Nhóm 3: Thay thế bằng Mode (cho các biến phân loại còn lại)
    cols_fill_mode = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 
                      'KitchenQual', 'Functional', 'SaleType', 'Electrical']

    # 4. Thực hiện xử lý
    print("Đang điền giá trị khuyết...")
    
    # Fill None
    for col in cols_fill_none:
        if col in df.columns:
            df[col] = df[col].fillna('None')
            
    # Fill Median
    for col in cols_fill_median:
        if col in df.columns:
            # Chỉ tính median nếu cột là dạng số
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())

    # Fill Mode
    for col in cols_fill_mode:
        if col in df.columns:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])

    # 5. Lưu kết quả
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Đã xử lý xong nhiễu.")
    print(f"File sạch được lưu tại: {output_path}")

if __name__ == "__main__":
    handle_noise()