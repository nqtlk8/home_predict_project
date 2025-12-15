import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def handle_outliers_manual():
    # 1. Cấu hình đường dẫn
    # Đọc từ file đã xử lý Missing Values
    input_path = os.path.join('data', 'processed', 'filled_data.csv')
    output_folder = os.path.join('data', 'processed')
    output_path = os.path.join(output_folder, 'data_clean.csv')

    print(f"--- Đang thực hiện xử lý Outlier thủ công (Manual Removal) ---")

    if not os.path.exists(input_path):
        print(f"❌ Lỗi: Không tìm thấy file '{input_path}'")
        return

    df = pd.read_csv(input_path)
    original_len = len(df)
    print(f"Số lượng bản ghi ban đầu: {original_len}")

    # =========================================================================
    # QUAN TRỌNG: CHỈ XỬ LÝ TRÊN TẬP TRAIN (CÓ CỘT SalePrice)
    # Tập Test chúng ta không được phép xóa dòng!
    # =========================================================================
    
    if 'SalePrice' in df.columns:
        print("🔍 Phát hiện tập TRAIN -> Áp dụng luật loại bỏ Outlier của Ames Housing.")
        
        # 1. Vẽ biểu đồ trước khi xóa để thấy rõ 4 ngoại lai 2 tốt và 2 xấu
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=df, x='GrLivArea', y='SalePrice')
        plt.title("Trước khi xóa Outliers")
        plt.show()

        # 2. Định nghĩa Outlier theo tác giả Dean De Cock:
        # "Những căn nhà có diện tích GrLivArea > 4000 nhưng SalePrice < 300,000"
        # Đây là những trường hợp dị biệt (nhà rất to nhưng giá rẻ bất thường)
        
        outlier_condition = (df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)
        num_outliers = outlier_condition.sum()
        
        print(f"👉 Phát hiện {num_outliers} căn nhà 'khổng lồ' nhưng giá rẻ (Nhiễu thực sự).")
        
        # 3. Thực hiện xóa
        df_clean = df[~outlier_condition]
        
        # 4. Kiểm tra một số cột khác (Optional)
        # Có thể lọc thêm các trường hợp GarageArea hoặc TotalBsmtSF quá lớn bất thường
        # Nhưng GrLivArea là quan trọng nhất.
        
    else:
        print("⚠️ Đây là tập TEST (không có SalePrice).")
        print("👉 KHÔNG ĐƯỢC XÓA DÒNG. Sẽ giữ nguyên dữ liệu.")
        # Đối với tập test, nếu có giá trị quá lớn gây lỗi, ta chỉ nên Clip nhẹ
        # Ví dụ: Clip GrLivArea về 5000 (nếu có cái nào to hơn thế) để tránh lỗi tính toán
        # Nhưng thường thì để nguyên cũng được.
        df_clean = df

    # =========================================================================
    
    # Lưu kết quả
    rows_removed = original_len - len(df_clean)
    print(f"✅ Đã loại bỏ: {rows_removed} dòng.")
    print(f"📉 Số lượng bản ghi còn lại: {len(df_clean)}")
    
    os.makedirs(output_folder, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"💾 File sạch được lưu tại: {output_path}")

if __name__ == "__main__":
    handle_outliers_manual()