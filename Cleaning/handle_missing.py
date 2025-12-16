import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_csv("data/raw/train.csv")


def kmeans_impute_masvnr(df):
    """
    Điền giá trị thiếu (NaN) cho MasVnrArea (Mean) và MasVnrType (Mode) 
    bằng K-Means Imputation, sử dụng OverallQual, YearBuilt, và ExterQual 
    làm các đặc trưng phân nhóm.
    """
    df_processed = df.copy()

    # 1. Mã hóa ExterQual sang số
    exter_qual_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    df_processed['ExterQual_Num'] = df_processed['ExterQual'].map(exter_qual_mapping)

    features = ['OverallQual', 'YearBuilt', 'ExterQual_Num']
    target_area = 'MasVnrArea'
    target_type = 'MasVnrType'

    # --- 2. K-Means Clustering: Fit trên tập dữ liệu NON-MISSING của MasVnrArea ---
    
    missing_area_mask = df_processed[target_area].isnull()
    df_non_missing_area = df_processed[~missing_area_mask].copy()

    # Scaling Features
    scaler = StandardScaler()
    X_non_missing_area_scaled = scaler.fit_transform(df_non_missing_area[features])

    # Fit K-Means
    K = 4 
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    df_non_missing_area['Cluster'] = kmeans.fit_predict(X_non_missing_area_scaled)

    # --- 3. Imputation cho MasVnrArea (8 NaNs) ---

    cluster_means_area = df_non_missing_area.groupby('Cluster')[target_area].mean()
    df_missing_area = df_processed[missing_area_mask].copy()
    X_missing_area_scaled = scaler.transform(df_missing_area[features])
    df_missing_area['Cluster'] = kmeans.predict(X_missing_area_scaled)
    
    # Điền giá trị MasVnrArea
    for cluster_id in range(K):
        mean_area = cluster_means_area.get(cluster_id, 0) 
        df_missing_area.loc[df_missing_area['Cluster'] == cluster_id, target_area] = mean_area
    df_processed.loc[missing_area_mask, target_area] = df_missing_area[target_area]

    # --- 4. Imputation cho MasVnrType (13 NaNs) ---
    
    missing_type_mask = df_processed[target_type].isnull()
    df_missing_type = df_processed[missing_type_mask].copy()

    # Tính Mode MasVnrType cho mỗi Cluster
    def get_mode(series):
        mode = series.mode()
        return mode.iloc[0] if not mode.empty else 'None'

    cluster_modes_type = df_non_missing_area.groupby('Cluster')[target_type].agg(get_mode)
    
    # Predict Cluster cho 13 dòng thiếu MasVnrType
    X_missing_type_scaled = scaler.transform(df_missing_type[features])
    df_missing_type['Cluster'] = kmeans.predict(X_missing_type_scaled)
    
    # Điền giá trị MasVnrType
    for cluster_id in range(K):
        mode_type = cluster_modes_type.get(cluster_id, 'None')
        df_missing_type.loc[df_missing_type['Cluster'] == cluster_id, target_type] = mode_type
        
    df_processed.loc[missing_type_mask, target_type] = df_missing_type[target_type]
    
    # --- 5. Tổng hợp dữ liệu và Kiểm tra ---
    df_processed.drop(columns=['ExterQual_Num'], inplace=True, errors='ignore')
    
    return df_processed

def compute_lotfrontage_r2(df):
    results = []
    for nbh in df['Neighborhood'].unique():
        sub = df[df['Neighborhood'] == nbh][['LotArea', 'LotFrontage']].dropna()
        if len(sub) > 2:
            X = np.log1p(sub['LotArea']).values.reshape(-1, 1)
            y = sub['LotFrontage'].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
        else:
            r2 = 0  # nếu dữ liệu ít quá
        results.append({'Neighborhood': nbh, 'R2': r2})
    return pd.DataFrame(results)


# === 1. Điền LotFrontage ===
def hybrid_fill_lotfrontage(df, eval_cv, r2_threshold=0.5):
    """
    Điền LotFrontage:
     - Nếu R² >= r2_threshold → dùng LinearRegression(LotArea)
     - Ngược lại → median theo Neighborhood
    """
    df = df.copy()
    neighborhoods = eval_cv['Neighborhood'].unique()
    for nbh in neighborhoods:
        r2 = eval_cv.loc[eval_cv['Neighborhood'] == nbh, 'R2'].values[0]
        mask = df['Neighborhood'] == nbh
        missing_mask = mask & df['LotFrontage'].isnull()
        if not missing_mask.any():
            continue

        if r2 >= r2_threshold:
            sub = df.loc[mask, ['LotArea', 'LotFrontage']].dropna()
            if len(sub) < 2:
                continue
            X = np.log1p(sub['LotArea']).values.reshape(-1, 1)
            y = sub['LotFrontage'].values
            model = LinearRegression().fit(X, y)
            X_missing = np.log1p(df.loc[missing_mask, 'LotArea']).values.reshape(-1, 1)
            df.loc[missing_mask, 'LotFrontage'] = model.predict(X_missing)
        else:
            med = df.loc[mask, 'LotFrontage'].median()
            df.loc[missing_mask, 'LotFrontage'] = med
    return df

def handle_missing(df):
    # Fence null nghĩa là không có hàng rào
    df['Fence'] = df['Fence'].fillna('No Fence')

    # nếu miscval = 0 thì miscfeature = 'No Misc'
    df['MiscFeature'] = df.apply(lambda row: 'No Misc' if row['MiscVal'] == 0 else row['MiscFeature'], axis=1)

    # Nếu Fireplaces = 0 thì FireplaceQu = 'No Fireplace'
    df['FireplaceQu'] = df.apply(lambda row: 'No Fireplace' if row['Fireplaces'] == 0 else row['FireplaceQu'], axis=1)

    # Nếu poolarea = 0 thì poolqc = 'No Pool'
    df['PoolQC'] = df.apply(lambda row: 'No Pool' if row['PoolArea'] == 0 else row['PoolQC'], axis=1)

    # Nếu GarageCars và GarageArea = 0 thì fill null bằng 'No Garage'
    df['GarageType'] = df.apply(lambda row: 'No Garage' if row['GarageCars'] == 0 and row['GarageArea'] == 0 else row['GarageType'], axis=1)
    df['GarageFinish'] = df.apply(lambda row: 'No Garage' if row['GarageCars'] == 0 and row['GarageArea'] == 0 else row['GarageFinish'], axis=1)
    df['GarageQual'] = df.apply(lambda row: 'No Garage' if row['GarageCars'] == 0 and row['GarageArea'] == 0 else row['GarageQual'], axis=1)
    df['GarageCond'] = df.apply(lambda row: 'No Garage' if row['GarageCars'] == 0 and row['GarageArea'] == 0 else row['GarageCond'], axis=1)
    df['GarageYrBlt'] = df.apply(lambda row: 0 if row['GarageCars'] == 0 and row['GarageArea'] == 0 else row['GarageYrBlt'], axis=1)
    

    # Các cột basement nếu có bsmtqual null thì fill null bằng 'No Basement'
    basement_cols = [col for col in df.columns if 'Bsmt' in col]
    for col in basement_cols:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('No Basement')
            else:
                df[col] = df[col].fillna(0)

    # Nếu MasVnrArea = 0 thì MasVnrType = 'No Masonry Veneer'
    df['MasVnrType'] = df.apply(lambda row: 'No Masonry Veneer' if row['MasVnrArea'] == 0 else row['MasVnrType'], axis=1)

    # Điền mode cho Electrical để giữ nguyên phân phối
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

    # Fill null cho Allowment bằng 'No Alley'
    df['Alley'] = df['Alley'].fillna('No Alley')
    
    # K-Means Imputation cho MasVnrArea và MasVnrType
    df = kmeans_impute_masvnr(df)

    # Linear Regression + Median Imputation cho LotFrontage
    eval_cv = compute_lotfrontage_r2(df)
    df = hybrid_fill_lotfrontage(df, eval_cv, r2_threshold=0.5)
    return df