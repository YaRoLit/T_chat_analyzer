import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import re
#import pymorphy2
#import nltk
#from nltk.corpus import stopwords
#nltk.download('stopwords')
#from wordcloud import WordCloud


st.set_page_config(layout="wide")   # Полнооконное представление приложения


def preprocessing_df(link): 
    '''
    Осуществляем загрузку и предобработку датафрейма
    в аргументе передаем ссылку на файл csv с данными
    '''
    df = pd.read_csv(link, sep='\t', encoding='utf-8')
    df = df[~(df.text.isna()) | (df.text == 0)]
    df.date = pd.to_datetime(df.iloc[:, 0], dayfirst=True).dt.date
    return df


def time_histplot(df):
    '''
    Функция создает раскрывающуюся вкладку
    В которой размещается временная диаграмма по переданному датасету
    '''
    with st.expander("Динамика сообщений по времени"):
        fig, ax = plt.subplots(figsize=(25, 5))
        df.groupby('date').text.count().plot()
        title = ax.set_title(f'Динамика сообщений с {df.date[df.shape[0]]} до {df.date[0]}. Весь период составляет {(df.date[0]) - (df.date[df.shape[0]])}', fontsize=12)
        st.pyplot(fig)


def sentpie_and_chat(df):
    '''
    Функция создает раскрывающуюся вкладку
    В которой размещается два поля: слева и справа. В левом поле строится диаграмма тональности
    для сообщений из df, переданного в аргументе. В правом строится список сообщений из df
    '''
    with st.expander("Тональность сообщений в чате"):
        col1, col2 = st.columns(2)
        with col1:
            sentiment_data = df.sentiment.value_counts()
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_axes((1, 1, 1, 1))
            ax.pie(sentiment_data,
                    explode = [0.1, 0.1, 0.1],
                    autopct='%1.1f%%',
                    textprops={'fontsize': 14},
                    labels=sentiment_data.index,
                    shadow=True)
            #ax.set_title('Анализ тональности', size=20)
            st.pyplot(fig)
        with col2:
            st.dataframe(df, width=700, height=600)


def users_top(df):
    '''
    Функция создает раскрывающуюся вкладку
    В которой размещается два поля: слева и справа. В левом поле строится диаграмма распределения сообщений между юзверями
    из числа 10 наиболее активных участников переданного df. Справа - тепловая карта их сообщений по тональности.
    '''
    idx = False

    act_user = df.groupby('id').text.count().sort_values(ascending=False)[:10]  # Делаю выборку сообщений самых болтливых пользователей

    for user in act_user.index:
        idx = idx | (df.id == user)
    df_actusers = df[idx]

    user_sent = df_actusers.pivot_table(  # Делаю сводную таблицу положительных/ нейтральных/ отрицательных комментов между пользователями
        values='text',
        index='id',
        columns='sentiment',
        aggfunc='count',
    )

    with st.expander("Наиболее общительные участники чата"):
        col1, col2 = st.columns(2)
        with col1:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_axes((1, 1, 1, 1))
            pie = ax.pie(act_user, labels=act_user.index, autopct='%1.1f%%', startangle=90)   # Диаграмма распределения удельной доли сообщений между пользователями
            st.pyplot(fig)

        with col2:
            f, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(data=user_sent,           # А это тепловая карта распределения комментов по их окраске
                annot=True,
                cmap="BuPu",
                );
            st.pyplot(f)


data_url = 'https://raw.githubusercontent.com/YaRoLit/ml_test/main/Dev/Data/chat_sentiment.csv'


# Выводим заголовок страницы средствами Streamlit     
st.title('Анализ Telegram чатов')

# Вызываем функцию подгрузки csv файла с сообщениями
# link = st.text_input('Ссылка на файл чата telegram', 
#                        help=data_url, 
#                        autocomplete=data_url)

# Временная штука для создания видимости интерактива
choice = st.selectbox('Выберите чат для анализа', ("Chat1", "Chat2", "Chat3"), help='Здесь будет реальный выбор, но потом...')

# Тут жмакаем кнопку распознавания
result = st.button('Проанализировать чат', )

if result:

    df = preprocessing_df(data_url)
    
    time_histplot(df)
    sentpie_and_chat(df)
    users_top(df)