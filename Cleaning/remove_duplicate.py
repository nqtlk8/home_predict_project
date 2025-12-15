import pandas as pd
import os

def remove_duplicates():
    # 1. Cấu hình đường dẫn
    input_path = os.path.join('data', 'processed', 'filled_data.csv')
    output_folder = os.path.join('data', 'processed')
    output_path = os.path.join(output_folder, 'data_nodup.csv')

    print(f"Đang bắt đầu xử lý trùng lặp")
    
    # 2. Kiểm tra file đầu vào
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file tại '{input_path}'")

        return

    # 3. Đọc dữ liệu
    df = pd.read_csv(input_path)
    print(f"Dữ liệu gốc: {df.shape[0]} dòng, {df.shape[1]} cột.")

    # 4. Xử lý trùng lặp
    # Lấy danh sách cột để kiểm tra trùng (loại bỏ 'Id' vì Id luôn duy nhất)
    cols_to_check = [c for c in df.columns if c != 'Id']
    
    # Kiểm tra trùng lặp dựa trên nội dung (bỏ qua Id)
    duplicates_count = df.duplicated(subset=cols_to_check).sum()
    
    if duplicates_count > 0:
        print(f"Phát hiện {duplicates_count} dòng trùng nội dung (khác Id). Đang loại bỏ...")
        df_clean = df.drop_duplicates(subset=cols_to_check, keep='first')
    else:
        print("Không phát hiện dòng trùng lặp nào.")
        df_clean = df

    # 5. Lưu kết quả
    # Tạo thư mục 'data/processed' nếu chưa có
    os.makedirs(output_folder, exist_ok=True)
    
    df_clean.to_csv(output_path, index=False)
    print(f"Đã lưu file kết quả tại: {output_path}")
    print(f"Số dòng còn lại: {len(df_clean)}")

if __name__ == "__main__":
    remove_duplicates()