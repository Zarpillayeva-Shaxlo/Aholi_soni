import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Modelni yuklash
with open("aholimodel3.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit interfeysi
st.title("Aholi O'sishini Bashorat Qilish")

# Foydalanuvchi kiritadigan parametrlar
st.write("Yillar bo'yicha aholi sonini kiriting:")
input_data = []
for year in range(1960, 2024):
    value = st.number_input(f"{year}-yil", min_value=0, value=500000)
    input_data.append(value)

# Bashorat qilish
if st.button("Bashorat Qilish"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    
    # Natijalarni ko'rsatish
    next_years = range(2025, 2030)
    results = pd.DataFrame({'Yil': next_years, 'Bashorat qilingan aholi soni': prediction.flatten()})
    st.write(results)
