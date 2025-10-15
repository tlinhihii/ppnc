import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("Dataset.xlsx", sheet_name=0)
print("Đọc dữ liệu thành công! Số dòng:", df.shape[0])

# KIỂM TRA VÀ XÓA GIÁ TRỊ THIẾU

print(df.isnull().sum())
before = df.shape[0]
df = df.dropna()
after = df.shape[0]
print(f"Đã xóa {before - after} dòng bị thiếu dữ liệu.")

# ĐỌC NGƯỠNG OUTLIER TỪ FILE EDA.xlsx

eda = pd.read_excel("EDA.xlsx", sheet_name=0)
eda = eda[['Variable', 'Lower bound', 'Upper bound']].dropna()
eda_bounds = dict(zip(eda['Variable'], zip(eda['Lower bound'], eda['Upper bound'])))

print("Đã lấy được ngưỡng:")
for var, (low, high) in eda_bounds.items():
    print(f"   {var}: lower={low}, upper={high}")

# ÁP DỤNG XỬ LÝ OUTLIER THEO FILE EDA

for col, (lower, upper) in eda_bounds.items():
    if col in df.columns:
        df[col] = np.where(df[col] < lower, lower,
                   np.where(df[col] > upper, upper, df[col]))

print("Đã xử lý outlier theo ngưỡng từ file EDA.xlsx.")

# CHUYỂN ĐỔI KIỂU DỮ LIỆU

df['ID'] = df['ID'].astype(str)
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
df['experience_c'] = df['experience_c'].astype(str).astype('category')
df['Default'] = df['Default'].astype(int)
print("Đã chuyển đổi kiểu dữ liệu phù hợp.")

# LÀM SẠCH TEXT & XÓA TRÙNG LẶP

df = df.drop_duplicates()
for col in ['emp_length', 'purpose', 'home_ownership_n', 'Title']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()

df['experience_c'] = df['experience_c'].replace({
    'ft': 'full time', 'fulltime': 'full time', 'full-time': 'full time'
})
print("Đã làm sạch dữ liệu text.")

# CHUẨN HÓA DỮ LIỆU SỐ (STANDARDIZATION)

num_cols = list(eda_bounds.keys())  # chỉ chuẩn hóa các cột có trong file EDA
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("Đã chuẩn hóa dữ liệu số.")

# XUẤT FILE KẾT QUẢ

df.to_excel("cleaned_dataset_from_EDA.xlsx", index=False)
print("File kết quả: cleaned_dataset_from_EDA.xlsx")
