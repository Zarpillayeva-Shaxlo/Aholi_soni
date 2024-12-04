import streamlit as st
import pandas as pd
import joblib

# Modelni yuklash
model = joblib.load("aholimodel4.pkl")

# Datasetni yuklash
data = pd.read_csv("population_data3.csv")  # Dataset fayl nomi

# Foydalanuvchi interfeysi
st.title("Aholi O'sishini Bashorat Qilish Ilovasi")
st.write("Mamlakatni tanlang:")

# Mamlakatni tanlash
countries = data['Country Name'].unique()
selected_country = st.selectbox("Mamlakatni tanlang:", countries)

# Tanlangan mamlakat ma'lumotlarini olish
country_data = data[data['Country Name'] == selected_country].iloc[:, 4:].values.flatten()  # 1960-2024

# Bashorat qilish
if st.button("Bashorat Qiling"):
    input_data = pd.DataFrame([country_data])  # Model uchun tayyorlash
    prediction = model.predict(input_data)[0]  # Bashorat

    # Bashorat natijalarini ko'rsatish
    next_years = range(2025, 2030)  # Keyingi 5 yil
    results = pd.DataFrame({
        'Yil': next_years,
        'Yil': next_years,
        'Bashorat qilingan aholi soni': prediction
    })
    st.write(f"{selected_country} mamlakati uchun bashorat qilingan aholi soni:")
    st.write(results)
