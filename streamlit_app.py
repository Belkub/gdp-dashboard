import streamlit as st
import pandas as pd
import math
import matplotlib 
import numpy as np
from matplotlib import pyplot as plt
from string import ascii_letters, digits, punctuation
import random

st.title('Графический калькулятор')
x_min = st.number_input('Минимум', value = 0)
x_max = st.number_input('Максимум', value = 10)
steps = st.slider('Количество точек', 50, 500)
grid = st.checkbox('Сетка')
function = st.selectbox('Функция', ['log', 'sin', 'cos'])
x = np.linspace(x_min, x_max, steps)
y = getattr(np, function)(x)
fig = plt.figure()

plt.plot(x,y)
if grid:
    plt.grid()
st.pyplot(fig)

