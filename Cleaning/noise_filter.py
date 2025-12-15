import pandas as pd
import os

def handle_outliers():
    # 1. Cấu hình đường dẫn
    # Đọc từ file đã điền Missing Values (kết quả của file code trước đó)
    input_path = os.path.join('data', 'processed', 'data_nodup.csv')
    output_folder = os.path.join('data', 'processed')
    output_path = os.path.join(output_folder, 'data_clean.csv')

    print(f"--- Đang bắt đầu xử lý Nhiễu (Outliers) ---")

    # 2. Kiểm tra file đầu vào
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file '{input_path}'")
        print("Cần chạy file xử lý remove_duplicate trước để tạo ra file này.")
        return

    df = pd.read_csv(input_path)
    print(f"Dữ liệu đầu vào: {df.shape}")

    # 3. Chọn các cột số liên tục cần xử lý ngoại lai
    # Lưu ý: Không nên xử lý trên tất cả các cột số, vì một số cột là mã hóa (ví dụ MSSubClass, OverallQual)
    cols_to_process = [
        'LotFrontage',  # Mặt tiền
        'LotArea',      # Diện tích đất
        'MasVnrArea',   # Diện tích ốp lát
        'BsmtFinSF1',   # Diện tích hầm hoàn thiện
        'TotalBsmtSF',  # Tổng diện tích hầm
        '1stFlrSF',     # Diện tích tầng 1
        'GrLivArea',    # Diện tích ở trên mặt đất
        'GarageArea',   # Diện tích Gara
        'MiscVal'       # Giá trị các tính năng phụ
    ]

    count_changed = 0

    # 4. Thực hiện Capping (Phương pháp IQR)
    print("\nChi tiết xử lý từng cột:")
    for col in cols_to_process:
        if col in df.columns:
            # Tính toán IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Thiết lập biên (Dùng 3.0 * IQR để chỉ lọc những nhiễu cực đoan nhất)
            lower_bound = Q1 - 3.0 * IQR
            upper_bound = Q3 + 3.0 * IQR
            
            # Đếm số lượng ngoại lai trước khi xử lý
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers_count > 0:
                print(f"   - Cột '{col}': Phát hiện {outliers_count} giá trị ngoại lai -> Đang ép về ngưỡng.")
                # Dùng hàm clip để cắt ngọn (ép giá trị về biên trên/dưới)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                count_changed += 1

    # 5. Lưu kết quả cuối cùng
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nĐã xử lý xong Outliers.")
    print(f"Đã can thiệp xử lý trên {count_changed} cột.")
    print(f"File được lưu tại: {output_path}")

if __name__ == "__main__":
    handle_outliers()