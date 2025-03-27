import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Загрузка данных
df = pd.read_csv('titanic.csv')

# Описательная статистика
st.title('Дашбоард по датасету Титаника')
st.header('Описательная статистика')
st.write(df.describe())

# Графики
st.header('Графики')

# Гистограмма распределения выживших
survived_hist = px.histogram(df, x='Survived', title='Распределение выживших')
st.plotly_chart(survived_hist)

# Диаграмма рассеяния возраста vs стоимости билета
age_fare_scatter = px.scatter(df, x='Age', y='Fare', title='Возраст vs Стоимость билета')
st.plotly_chart(age_fare_scatter)

# Круговая диаграмма распределения по классам
class_pie = px.pie(df, names='Pclass', title='Распределение по классам')
st.plotly_chart(class_pie)

# Столбчатая диаграмма распределения по полу
gender_bar = px.bar(df, x='Sex', title='Распределение по полу')
st.plotly_chart(gender_bar)

# Интерактивная гистограмма возраста
n = st.number_input('Введите количество строк для интерактивного графика:', min_value=1, max_value=len(df), value=5, step=1)
interactive_hist = px.histogram(df.head(n), x='Age', title=f'Возраст (первые {n} строк)')
st.plotly_chart(interactive_hist)

# Вывод n строк из начальной таблицы
st.header('Вывод строк')
n_rows = st.number_input('Введите количество строк для вывода:', min_value=1, max_value=len(df), value=5, step=1)
st.write(df.head(n_rows)
