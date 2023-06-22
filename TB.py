import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Membaca dataset
data = pd.read_csv('Data Historis PTBA.csv')
# Mengubah kolom tanggal menjadi tipe data datetime
data['Tanggal'] = pd.to_datetime(data['Tanggal'], format='%d/%m/%Y')
print(data)
data.shape
data.isnull().sum()
# Memisahkan (tanggal) dan target (harga penutupan)
X = data['Tanggal'].values.astype(np.int64) // 10**9
y = data['Terakhir'].values.astype(np.int64)
# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Membuat objek model regresi linier
model = LinearRegression()
# Melatih model menggunakan data latih
model.fit(X_train.reshape(-1, 1), y_train)
# Memprediksi harga penutupan menggunakan data uji
y_pred = model.predict(X_test.reshape(-1, 1))
# Menghitung nilai Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
terakhir_price = data['Terakhir'].iloc[-1]
future_harga = [terakhir_price * (1 + i/100) for i in range(1, 11)]

# Prediksi harga saham di masa depan
tgl = pd.date_range(start=data['Tanggal'].max(), periods=10, freq='M')
predictions_df = pd.DataFrame({'Tanggal': tgl, 'Prediksi Harga': future_harga})
print("\nPrediksi Harga Saham di Masa Depan:")
print(predictions_df)
# Mendapatkan koefisien kemiringan dan intersep dari model regresi linear
slope = model.coef_[0]
intercept = model.intercept_
# Menghitung prediksi harga menggunakan fungsi linier
y_regression = slope * X + intercept
plt.figure(figsize=(15, 6))
plt.plot(data['Tanggal'], data['Terakhir'], label='Data Historis')
plt.plot(predictions_df['Tanggal'], predictions_df['Prediksi Harga'], marker='o', linestyle='-', label='Prediksi Harga')
plt.plot(data['Tanggal'], y_regression, color='red', linestyle='--', label='Garis Regresi Linear')
plt.xlabel('Tanggal')
plt.ylabel('Harga Saham')
plt.title('Prediksi Harga Saham di Masa Depan')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()
