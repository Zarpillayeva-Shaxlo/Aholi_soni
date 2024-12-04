import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Modelni yuklash
model = joblib.load("aholimodel1.pkl")

# Foydalanuvchi interfeysi
st.title("Aholi soni bashorati ilovasi")
st.write("Yillar bo'yicha aholi sonini kiriting:")

# Parametrlarni foydalanuvchidan olish
years = list(range(1960, 2020))
input_data = []

for year in years:
    value = st.number_input(f"{year}-yildagi aholi soni", value=1_000_000)
    input_data.append(value)

# Bashorat qilish
if st.button("Bashorat qiling"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"Bashorat qilingan aholi soni (2020-yil): {int(prediction):,}")
