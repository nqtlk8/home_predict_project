
# Remove duplicate rows based on all columns
def remove_duplicate_rows(df):
    initial_shape = df.shape
    df_cleaned = df.drop_duplicates()
    final_shape = df_cleaned.shape
    print(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows.")
    return df_cleaned
