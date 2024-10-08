import pandas as pd
import numpy as np

import joblib
import json

from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn


app = FastAPI()
model = joblib.load("../models/model_logreg_v2.pkl")

# Класс валидации формы
class Form(BaseModel):
    visit_number: int
    utm_source: int
    utm_medium: int
    utm_campaign: str
    utm_adcontent: str
    device_category: str
    device_screen_resolution: int
    geo_country: str
    geo_city: str
    hit_number: int
    hit_type: str
    visit_date_m: int
    visit_date_d: int
    time: int
    dayofweek: int
    brand: str
    model: str


class Prediction(BaseModel):
    Result: float


# Отображает рабочий статус
@app.get('/status')
def status():
    return "i'm OK"


# Предсказание модели 
@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])

    # Препроцессор сырых данных
    _int = df.select_dtypes(include=['int64', 'float64']).columns
    _cat = df.select_dtypes(include=['object']).columns
    df[_int] = model['_int'].transform(df[_int])
    df[_cat] = model['_cat'].transform(df[_cat])

    # Выполнение предикта
    y = model['model'].predict(df)
    print(f'{form.dict()}: {y[0]}')

    return {
        'Result': y[0]
    }


# При необходимости, можно запуастить простой тест, бех необходимости запускать 
# весь сервис
def simple_test():
    model = joblib.load("../models/model_logreg_v2.pkl")
    _feature = model['model'].feature_names_in_.tolist()

    # Тестовые данные (можно поставить свои)
    real_test = [
        1,
        0,
        0,
        'LEoPHuyFvzoNfnzGgfcd',
        'vCIpmpaGBnIQhyYNkXqp',
        'mobile',
        259200,
        'Russia',
        'Zlatoust',
        3,
        'event',
        11,
        24,
        14,
        2,
        'other',
        'other'
    ]

    # Преобразование к нужному формату данных
    df = pd.DataFrame([real_test], columns=_feature)

    _int = df.select_dtypes(include=['int64', 'float64']).columns
    _cat = df.select_dtypes(include=['object']).columns

    df[_int] = model['_int'].transform(df[_int])
    df[_cat] = model['_cat'].transform(df[_cat])

    # Предсказание
    rezult = model['model'].predict(df)

    # Песчать
    print(f"Rezult: {rezult}")




if __name__ == '__main__':
    # Установите в 1 тот пункт, который требуется запустить
    settings = {
        'simple_test': 0,
        'server': 1
    }
    # Запуск простого теста
    if settings['simple_test']: simple_test()
    # Запуск сервера
    if settings['server']: uvicorn.run(app, host="127.0.0.1", port=8000)