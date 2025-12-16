# clean_pipeline.py

"""
Run the full data cleaning pipeline.
Steps:
1) Load raw data.
2) Handle missing values using logic defined in handle_missing.py.
3) Remove duplicate rows using logic defined in remove_duplicate.py.
4) (Optional) Noise/Outlier detection and filtering.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

# GIẢ ĐỊNH: Các hàm này nằm trong các file tương ứng trong thư mục Cleaning
# Vì chúng ta không thể truy cập các file khác của bạn, tôi sẽ tạo một giả định.
# Bạn cần đảm bảo các lệnh import sau hoạt động trong môi trường của bạn:
from handle_missing import handle_missing 
from remove_duplicate import remove_duplicate_rows
# from noise_filter import detect_outliers_and_apply_rules # Giả định cho bước 3


def ensure_input_file(file_path: Path) -> None:
    """Kiểm tra xem file đầu vào có tồn tại không."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing required input data at '{file_path}'. "
            f"Please ensure the raw data file is present."
        )


def main() -> None:
    """Thực thi pipeline làm sạch dữ liệu."""
    
    # Định nghĩa các đường dẫn
    RAW_DATA_PATH = Path('data/raw/train.csv')
    CLEANED_DIR = Path('data/cleaned')
    
    # Đảm bảo thư mục đầu ra tồn tại
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Kiểm tra file đầu vào
    ensure_input_file(RAW_DATA_PATH)
    
    print("=== START: Data Cleaning Pipeline ===")

    # Load the dataset
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Loaded raw data: {RAW_DATA_PATH} ({len(df)} rows)")

    
    # --- Step 1: Handle Missing Values ---
    print("\n--- Step 1: Handling Missing Values ---")
    
    # Giả định handle_missing(df) trả về DataFrame đã điền đầy đủ
    df_filled = handle_missing(df)
    
    MISSING_HANDLED_PATH = CLEANED_DIR / 'missing_handled_data.csv'
    df_filled.to_csv(MISSING_HANDLED_PATH, index=False)
    print(f"Missing values handled and saved to: {MISSING_HANDLED_PATH}")

    
    # --- Step 2: Remove Duplicate Rows ---
    print("\n--- Step 2: Removing Duplicate Rows ---")
    
    # Giả định remove_duplicate_rows(df) trả về DataFrame đã lọc
    df_no_duplicates = remove_duplicate_rows(df_filled)
    
    duplicates_removed = len(df_filled) - len(df_no_duplicates)
    print(f"Removed {duplicates_removed} duplicate row(s).")
    
    
    # --- Step 3 (Optional): Noise/Outlier Filtering ---
    # Giữ nguyên logic của bạn nhưng chưa thực thi
    # print("\n--- Step 3: Detecting and Filtering Outliers/Noise ---")
    # try:
    #     # Giả định detect_outliers_and_apply_rules(df) trả về DataFrame đã lọc outlier
    #     df_final_cleaned = detect_outliers_and_apply_rules(df_no_duplicates)
    # except NameError:
    #     print("Skipping Outlier/Noise step (function not imported).")
    df_final_cleaned = df_no_duplicates # Tạm thời bỏ qua bước này

    
    # --- Final Save ---
    CLEANED_DATA_PATH = CLEANED_DIR / 'cleaned_data.csv'
    df_final_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"\nFinal cleaned data saved to: {CLEANED_DATA_PATH} ({len(df_final_cleaned)} rows)")
    
    print("\n✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()