import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import scipy

# Загрузка данных
@st.cache_data
def load_data():
    url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    data = pd.read_csv(url)
    # Преобразования данных
    data['Sex'] = data['Sex'].map({'male': 'Мужчина', 'female': 'Женщина'})
    data['Survived'] = data['Survived'].map({0: 'Нет', 1: 'Да'})
    data['Pclass'] = data['Pclass'].astype('category')
    data['Age'].fillna(data['Age'].median(), inplace=True)
    return data

data = load_data()

# Настройка страницы
st.set_page_config(page_title="Титаник Дашборд", layout="wide", page_icon="🚢")
st.title("🚢 Расширенный анализ пассажиров Титаника")
st.markdown("""
Интерактивный дашборд для исследования датасета с пассажирами Титаника.
Анализируйте данные с помощью фильтров и визуализаций.
""")

# Раздел 1: Описательная статистика
st.header("📊 Описательная статистика")
with st.expander("Показать метаинформацию о датасете"):
    st.subheader("Структура данных")
    st.write(f"Форма таблицы: {data.shape[0]} строк, {data.shape[1]} столбцов")
    
    st.subheader("Типы данных")
    dtype_info = pd.DataFrame({
        'Столбец': data.columns,
        'Тип данных': data.dtypes.values,
        'Уникальных значений': [data[col].nunique() for col in data.columns]
    })
    st.dataframe(dtype_info)
    
    st.subheader("Основные статистики")
    st.dataframe(data.describe(include='all'))

# Разделитель
st.divider()

# Раздел 2: Фильтры и выборка данных
st.header("🔍 Фильтрация данных")

col1, col2 = st.columns(2)
with col1:
    # Фильтр по выживанию
    survived_filter = st.multiselect(
        "Выживание",
        options=data['Survived'].unique(),
        default=data['Survived'].unique()
    )
    
    # Фильтр по классу каюты
    pclass_filter = st.multiselect(
        "Класс каюты", 
        options=data['Pclass'].unique(),
        default=data['Pclass'].unique()
    )

with col2:
    # Фильтр по полу
    sex_filter = st.multiselect(
        "Пол",
        options=data['Sex'].unique(),
        default=data['Sex'].unique()
    )
    
    # Фильтр по возрасту
    age_range = st.slider(
        "Диапазон возраста",
        min_value=int(data['Age'].min()),
        max_value=int(data['Age'].max()),
        value=(0, 80)
    )

# Применение фильтров
filtered_data = data[
    (data['Survived'].isin(survived_filter)) &
    (data['Pclass'].isin(pclass_filter)) &
    (data['Sex'].isin(sex_filter)) &
    (data['Age'] >= age_range[0]) &
    (data['Age'] <= age_range[1])
]

# Вывод выборки данных
st.subheader("Выборка данных")
n_rows = st.number_input(
    "Количество строк для отображения", 
    min_value=1, 
    max_value=100, 
    value=10
)
st.dataframe(filtered_data.head(n_rows))

# Разделитель
st.divider()

# Раздел 3: Визуализации
st.header("📈 Визуализации данных")

# График 1: Распределение возраста (гистограмма)
st.subheader("1. Распределение возраста пассажиров")
fig1 = ff.create_distplot([filtered_data['Age'].dropna()], ['Возраст'], bin_size=2)
fig1.update_layout(xaxis_title="Возраст", yaxis_title="Плотность")
st.plotly_chart(fig1, use_container_width=True)
   

# График 2: Соотношение выживших/погибших (круговая диаграмма)
st.subheader("2. Соотношение выживших и погибших")
survived_counts = filtered_data['Survived'].value_counts()
fig2, ax2 = plt.subplots()
ax2.pie(
    survived_counts, 
    labels=survived_counts.index, 
    autopct='%1.1f%%',
    colors=['#ff9999','#66b3ff']
)
ax2.set_title("Выжившие vs Погибшие")
st.pyplot(fig2)


# График 3: Выживаемость по классу и полу (столбчатая диаграмма)
st.subheader("3. Выживаемость по классу каюты и полу")
fig3 = px.bar(
    filtered_data,
    x='Pclass',
    color='Survived',
    facet_col='Sex',
    barmode='group',
    title='Выживаемость по классу каюты и полу',
    labels={'Pclass': 'Класс каюты', 'count': 'Количество'},
    color_discrete_map={'Нет': 'red', 'Да': 'green'}
)
st.plotly_chart(fig3, use_container_width=True)

# График 4: Распределение стоимости билета (ящик с усами)
st.subheader("4. Распределение стоимости билета по классу")
fig4 = px.box(
    filtered_data,
    x='Pclass',
    y='Fare',
    color='Survived',
    title='Распределение стоимости билета по классу каюты',
    labels={'Pclass': 'Класс каюты', 'Fare': 'Стоимость билета'}
)
st.plotly_chart(fig4, use_container_width=True)

# График 5: Интерактивный график (реагирует на ввод)
st.subheader("5. Интерактивный анализ: Возраст vs Стоимость билета")

# Получаем диапазон возраста от пользователя
age_range = st.slider(
    "Выберите диапазон возраста",
    min_value=0,
    max_value=100,
    value=(0, 100),
    step=1
)

# Фильтруем данные по выбранному диапазону возраста
filtered_data = data[(data['Age'] >= age_range[0]) & (data['Age'] <= age_range[1])]

# Создаем точечный график с использованием Plotly Express
fig5 = px.scatter(
    filtered_data,
    x="Age",
    y="Fare",
    title="Точечный график: Возраст vs Стоимость билета",
    color="Survived",
    color_discrete_map={0: "red", 1: "green"},
    hover_data=["Name", "Sex"]
)

# Отображаем график
st.plotly_chart(fig5)


st.subheader("6. Корреляционная матрица")
numeric_cols = filtered_data.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = filtered_data[numeric_cols].corr()
fig4 = px.imshow(corr_matrix, text_auto=True, aspect="auto")
st.plotly_chart(fig4, use_container_width=True)
   


if plot_type == "Точечный":
    fig5 = px.scatter(
        filtered_data,
        x='Age',
        y='Fare',
        color='Survived',
        size='Siblings/Spouses Aboard',
        hover_data=['Name'],
        title='Зависимость стоимости билета от возраста',
        labels={'Fare': 'Стоимость билета', 'Age': 'Возраст'}
    )
elif plot_type == "Гексбин":
    fig5 = px.density_heatmap(
        filtered_data,
        x='Age',
        y='Fare',
        facet_col='Survived',
        title='Плотность распределения: Возраст vs Стоимость билета',
        labels={'Fare': 'Стоимость билета', 'Age': 'Возраст'}
    )
else:
    fig5 = px.violin(
        filtered_data,
        x='Survived',
        y='Fare',
        color='Sex',
        box=True,
        points="all",
        title='Распределение стоимости билета по выживаемости и полу',
        labels={'Fare': 'Стоимость билета', 'Survived': 'Выжил'}
    )

if st.button("Сбросить фильтры"):
    # Сбросьте все фильтры здесь
    st.rerun()  

