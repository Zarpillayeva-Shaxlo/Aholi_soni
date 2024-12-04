import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Modelni yuklash
model = joblib.load("aholimodel2.pkl")

# Datasetni yuklash (mamlakat va yillik ma'lumotlarni o'z ichiga olgan CSV)
data = pd.read_csv("population_data.csv")

# Foydalanuvchi interfeysi
st.title("Aholi soni bashorati ilovasi")
st.write("Mamlakatni tanlang va 5 yillik aholi sonini bashorat qiling:")

# Foydalanuvchidan mamlakatni tanlash
countries = data["Country Name"].unique()
selected_country = st.selectbox("Mamlakatni tanlang", countries)

# Tanlangan mamlakat uchun ma'lumotlarni olish
country_data = data[data["Country Name"] == selected_country].iloc[:, 4:]  # Faqat yillar ustunlarini olish

# Ma'lumotlarni tayyorlash
input_data = country_data.values.flatten()  # Tanlangan mamlakat yillik ma'lumotlari
st.write(f"{selected_country} uchun yillik ma'lumotlar yuklandi.")

# Bashorat qilish
if st.button("Bashorat qiling"):
    input_array = np.array(input_data).reshape(1, -1)
    
    # Bashorat qilinadigan yillar: 2024–2028
    predictions = model.predict(input_array)[0]
    
    # 5 yillik bashorat natijalari
    for i, year in enumerate(range(2024, 2029)):
        st.success(f"{year}-yildagi bashorat qilingan aholi soni: {int(predictions[i]):,}")
