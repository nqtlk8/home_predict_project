import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def perform_eda():
    # 1. Cấu hình
    # LƯU Ý: Phải dùng file TRAIN (có cột SalePrice) để phân tích
    input_path = os.path.join('data', 'raw', 'train.csv')
    output_folder = os.path.join('reports', 'figures')
    
    # Tạo thư mục lưu ảnh nếu chưa có
    os.makedirs(output_folder, exist_ok=True)
    
    print("--- Đang đọc dữ liệu TRAIN để phân tích ---")
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy file '{input_path}'")

        return
        
    df = pd.read_csv(input_path)
    
    # Chỉ lấy các cột số để tính toán tương quan
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # ========================================================
    # 1. VẼ HISTOGRAM GIÁ NHÀ 
    # ========================================================

    plt.figure(figsize=(10, 6))
    sns.histplot(df['SalePrice'], kde=True, color='navy', bins=30)
    plt.title('Phân phối Giá nhà (SalePrice Distribution)')
    plt.xlabel('Giá nhà ($)')
    plt.ylabel('Số lượng')
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(os.path.join(output_folder, "1_histogram_price.png"))
    plt.close()

    # ========================================================
    # 2. VẼ HEATMAP 
    # ========================================================

    plt.figure(figsize=(12, 10))
    
    # Tìm 10 yếu tố ảnh hưởng mạnh nhất đến giá nhà
    k = 10 
    cols = numeric_df.corr().nlargest(k, 'SalePrice')['SalePrice'].index
    cm = numeric_df[cols].corr()
    
    sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, cmap='RdBu')
    plt.title('Ma trận tương quan (Top 10 yếu tố ảnh hưởng giá)')
    plt.savefig(os.path.join(output_folder, "2_heatmap_correlation.png"))
    plt.close()

    # ========================================================
    # 3. VẼ PAIRPLOT 
    # ========================================================

    # Chỉ chọn 4 biến quan trọng nhất để vẽ (vẽ hết sẽ treo máy)
    important_cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']
    
    sns.pairplot(df[important_cols], height=2.5, kind='scatter', diag_kind='kde')
    plt.savefig(os.path.join(output_folder, "3_pairplot.png"))
    plt.close()

    print(f"3 biểu đồ đã được lưu tại thư mục '{output_folder}'.")

if __name__ == "__main__":
    perform_eda()