# Adult Census Income Prediction

Предсказание уровня дохода (>50K или <=50K) на основе данных переписи населения США.

## О проекте

Модель предсказывает, превышает ли годовой доход человека **$50,000** или нет.
Используется датасет **Adult Census Income** (32 561 запись, 15 признаков).

## Технологии

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- CatBoost


##
- Данные

- Источник: Kaggle — Adult Census Income

## Предобработка

- Замена ? на NaN

- Заполнение пропусков (SimpleImputer(strategy='most_frequent'))

- OneHotEncoder(handle_unknown='ignore') для категорий

- StandardScaler для числовых признаков


## Модели

| Модель | Accuracy | Precision | Recall | F1 |
|--------|:--------:|:---------:|:------:|:--:|
| LightGBM | 82% | 58% | **80%** | **67%** |
| **CatBoost** | **85%** | **72%** | 58% | 64% |

## Автор
Abdulla - https://github.com/Arthos761

## Как запустить

1. Установи зависимости:
```bash
pip install -r requirements.txt