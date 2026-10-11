import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 1. ЗАГРУЗКА И ПРОСМОТР ДАННЫХ
df = pd.read_csv('adult.csv')
print(df.head())
print(df.shape)
df = df.replace('?', np.nan)



# 2. РЕДАКТИРОВАНИЕ КОЛОНОК
df = df.drop('fnlwgt', axis=1)
df = df.drop('education', axis=1)

imputer = SimpleImputer(strategy='most_frequent')
df[['native.country']] = imputer.fit_transform(df[['native.country']])
df[["workclass"]] = imputer.fit_transform(df[['workclass']])
df[['occupation']] = imputer.fit_transform(df[['occupation']])


df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

print(df.isnull().sum())


# 3. ВЫБОРКА ДАННЫХ
X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.22, random_state = 42)

# 4. ОТБОР ПРИЗНАКОВ
numeric_cols = ['age', 'education.num', 'capital.loss', 'hours.per.week']

category_cols = ['workclass', 'marital.status', 'occupation', 'relationship', 'native.country', 'sex', 'race']

# 5. МАСШТАБИРОВАНИЕ ДАННЫХ
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, category_cols)
    ]
)

# 6. СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛЕЙ
model1 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lgb', LGBMClassifier(
        class_weight='balanced',
        verbose = -1,
        random_state = 42,
        subsample=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        num_leaves=100,
        n_estimators=300,
        min_child_samples=20,
        max_depth=-1,
        learning_rate=0.1,
        colsample_bytree=0.8
    ))
])

model2 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('cat', CatBoostClassifier(
        depth=5,
        iterations=800,
        l2_leaf_reg=2,
        learning_rate=0.03,
        random_strength=0.5
    ))
])

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# 7. ОЦЕНКА МОДЕЛЕЙ

    #LGBMClassifier
accuracy_lgb = accuracy_score(y_test,model1.predict(X_test))
precision_lgb = precision_score(y_test, model1.predict(X_test))
recall_lgb = recall_score(y_test, model1.predict(X_test))
f1_lgb = f1_score(y_test, model1.predict(X_test))


print(f'=== РЕЗУЛЬТАТЫ LGB  ===\n'
      f'1) Общая точность модели: {np.round(accuracy_lgb, 2) * 100}%\n'
      f'2) Точность положительных предсказаний: {np.round(precision_lgb, 2) * 100}%\n'
      f'3) Сколько смогла правильно предсказать из ПРАВИЛЬНЫХ: {np.round(recall_lgb, 2) * 100}%\n'
      f'4) Баланс между точностью и полнотой: {np.round(f1_lgb, 2) * 100}%\n')



    # CatBoostClassifier
accuracy_cat = accuracy_score(y_test, model2.predict(X_test))
precision_cat = precision_score(y_test, model2.predict(X_test))
recall_cat = recall_score(y_test, model2.predict(X_test))
f1_cat = f1_score(y_test, model2.predict(X_test))

print('==========================================')

print(f'=== РЕЗУЛЬТАТЫ CatBOOST ===\n'
      f'1) Общая точность модели: {np.round(accuracy_cat, 2) * 100}%\n'
      f'2) Точность положительных предсказаний: {np.round(precision_cat, 2) * 100}%\n'
      f'3) Сколько смогла правильно предсказать из ПРАВИЛЬНЫХ: {np.round(recall_cat, 2) * 100}%\n'
      f'4) Баланс между точностью и полнотой: {np.round(f1_cat, 2) * 100}%\n')




# 8. ПОДБОР ПАРАМЕТРОВ ПРИ ПОМОЩИ GridSearchCV

# param_grid_cat = {
#     'cat__depth': [4,8],              # Глубина деревьев
#     'cat__learning_rate': [0.03, 0.1],   # Скорость обучения
#     'cat__iterations': [750, 1250],       # Количество деревьев
#     'cat__l2_leaf_reg': [2, 4],        # L2-регуляризация
#     'cat__random_strength': [0.5, 1.0]    # Случайность при выборе сплитов
# }
#
#

# param_lgb = {
#     'lgb__num_leaves': [31, 50, 70, 100],
#     'lgb__max_depth': [5, 7, 10, -1],
#     'lgb__learning_rate': [0.03, 0.05, 0.1],
#     'lgb__n_estimators': [100, 300, 500],
#     'lgb__min_child_samples': [20, 50, 100],
#     'lgb__subsample': [0.8, 1.0],
#     'lgb__colsample_bytree': [0.8, 1.0],
#     'lgb__reg_alpha': [0, 0.1, 0.5],
#     'lgb__reg_lambda': [0, 0.1, 0.5]
# }


# grid_cat = GridSearchCV(model2, param_grid=param_grid_cat, cv=3, n_jobs=-1)
# grid_cat.fit(X_train, y_train)
# print(f'ЛУЧШИЕ ПАРАМЕТРЫ ДЛЯ CatBoost: {grid_cat.best_params_}\n'
#       f'ЛУЧШАЯ ТОЧНОСТЬ: {np.round(grid_cat.best_score_, 2)}')



# random_lgb = RandomizedSearchCV(
#     estimator = model1,
#     param_distributions=param_lgb,
#     n_iter=20,
#     cv=3,
#     n_jobs=-1,
#     scoring='accuracy',
#     random_state=42,
# )
# random_lgb.fit(X_train, y_train)
# print(f'ЛУЧШИЕ ПАРАМЕТРЫ: {random_lgb.best_params_}')
# print(f'ЛУЧШИЙ РЕЗ-ТАТ: {random_lgb.best_score_:.4f}')








