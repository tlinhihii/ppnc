import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Đọc file 
df = pd.read_excel("Dataset.xlsx", sheet_name=0)

print("Đọc dữ liệu thành công!")
print("Kích thước dữ liệu:", df.shape)
print("5 dòng đầu tiên:")
print(df.head())

# KIỂM TRA GIÁ TRỊ THIẾU

print("Số giá trị thiếu (NaN) theo từng cột:")
print(df.isnull().sum())

# Vì mỗi cột chỉ có 1-2 giá trị thiếu → xóa luôn các dòng có NaN
print("Đang xóa các dòng bị thiếu dữ liệu...")
before = df.shape[0]
df = df.dropna()
after = df.shape[0]
print(f"Đã xóa {before - after} dòng thiếu dữ liệu. Còn lại {after} bản ghi hợp lệ.")

# PHÁT HIỆN & XỬ LÝ OUTLIER (THEO IQR)

print("Đang xử lý outlier bằng phương pháp IQR...")

def winsorize_series(s):
    Q1 = s.quantile(0.25)
    Q3 = s.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    print(f"{s.name}: lower={lower:.2f}, upper={upper:.2f}")
    return s.clip(lower=lower, upper=upper)

num_cols = ['revenue', 'dti_n', 'loan_amnt', 'fico_n']
for c in num_cols:
    df[c] = winsorize_series(df[c])

print("Đã xử lý outlier (Winsorization).")


# CHUYỂN ĐỔI KIỂU DỮ LIỆU

print("Đang chuyển đổi kiểu dữ liệu...")

# Cột ID là định danh -> chuyển sang dạng chuỗi (string)
df['ID'] = df['ID'].astype(str)

# Cột ngày tháng
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')

# Cột phân loại
df['experience_c'] = df['experience_c'].astype(str)
df['Default'] = df['Default'].astype(int)

print("Đã chuyển đổi kiểu dữ liệu phù hợp.")
print(df.dtypes)

# LÀM SẠCH TEXT & XÓA TRÙNG LẶP

print("Đang làm sạch dữ liệu dạng text & xóa trùng lặp...")

# Xóa dòng trùng lặp
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"Đã xóa {before - after} bản ghi trùng lặp.")

# Chuẩn hóa chữ thường và loại khoảng trắng dư
for col in ['emp_length', 'purpose', 'home_ownership_n', 'Title']:
    df[col] = df[col].astype(str).str.lower().str.strip()

# Chuẩn hóa giá trị trong experience_c
df['experience_c'] = df['experience_c'].replace({
    'ft': 'full time',
    'fulltime': 'full time',
    'full-time': 'full time'
})

print("Đã làm sạch dữ liệu text.")

# CHUẨN HÓA DỮ LIỆU SỐ (STANDARDIZATION)

print("Đang chuẩn hóa dữ liệu numeric (mean≈0, std≈1)...")

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("Dữ liệu numeric đã được chuẩn hóa.")
print(df[num_cols].describe().round(2))

# XUẤT FILE KẾT QUẢ
df.to_excel("cleaned_dataset.xlsx", index=False)
print("File kết quả: cleaned_dataset.xlsx")
