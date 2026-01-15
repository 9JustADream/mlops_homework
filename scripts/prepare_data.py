import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
import os

def prepare_titanic_data():
    """Подготовка данных Titanic для проекта"""
    titanic = fetch_openml('titanic', version=1, as_frame=True)
    df = titanic.frame

    df = preprocess_titanic(df)

    baseline_data = df.head(600).copy()
    baseline_data.to_csv('data/baseline.csv', index=False)

    current_data = df.tail(600).copy()
    current_data = add_drift(current_data)
    current_data.to_csv('data/current.csv', index=False)

    return baseline_data, current_data

def preprocess_titanic(df):
    """Предобработка данных"""

    df = df.drop(['boat', 'body', 'home.dest', 'cabin'], axis=1, errors='ignore')
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})

    df['embarked'] = df['embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    df['age'] = df['age'].fillna(df['age'].median())
    df['fare'] = df['fare'].fillna(df['fare'].median())
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)

    df['title'] = df['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4,
        'Dr': 5, 'Rev': 6, 'Col': 7, 'Major': 7, 'Mlle': 2,
        'Countess': 8, 'Ms': 2, 'Lady': 8, 'Jonkheer': 8,
        'Don': 1, 'Dona': 8, 'Mme': 3, 'Capt': 7, 'Sir': 8
    }
    df['title'] = df['title'].map(title_mapping).fillna(0).astype(int)

    df = df.drop('name', axis=1)

    final_features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 
                     'embarked', 'title', 'family_size', 'is_alone', 'survived']
    
    int_columns = ['pclass', 'sex', 'sibsp', 'parch', 'embarked', 
               'title', 'family_size', 'is_alone', 'survived']
    
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: int(x) if pd.notnull(x) else x
            )
    
    return df[final_features]

def add_drift(df):
    """Добавляем искусственный дрифт в данные"""
    
    np.random.seed(42)
    
    age_drift = np.random.normal(7, 3, size=len(df))
    df['age'] = df['age'] + age_drift
    df.loc[df['age'] < 0, 'age'] = 0.1

    fare_drift = np.random.normal(40, 10, size=len(df))
    df['fare'] = df['fare'] + fare_drift
    df.loc[df['fare'] < 0, 'fare'] = 1.0

    df['is_alone'] = np.random.choice([0, 1], size=len(df), p=[0.3, 0.7])
 
    df['pclass'] = np.random.choice([1, 2, 3], size=len(df), p=[0.5, 0.2, 0.3])

    return df

if __name__ == "__main__":
    baseline, current = prepare_titanic_data()

    print("\nBaseline (эталонные данные):")
    print(baseline[['age', 'fare', 'pclass', 'is_alone']].describe())
    
    print("\nCurrent (данные с дрифтом):")
    print(current[['age', 'fare', 'pclass', 'is_alone']].describe())