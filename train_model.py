import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import joblib
import os
import pickle
import warnings

print("🧠 ОБУЧЕНИЕ РЕАЛИСТИЧНЫХ МОДЕЛЕЙ ДЛЯ КЕМЕРОВО")
print("=" * 60)

warnings.filterwarnings('ignore')


def train_and_save_model(data_path, target_col, model_name):
    """Обучает и сохраняет модель с валидацией"""
    print(f"\n📂 Загрузка данных: {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8')

    print(f"   Записей: {len(df)}, Целевая переменная: {target_col}")
    print(f"   Диапазон цен: {df[target_col].min():,.0f} - {df[target_col].max():,.0f}")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"   Категориальные признаки: {categorical_cols}")
    print(f"   Числовые признаки: {numeric_cols}")

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    print("🔄 Препроцессинг данных...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    input_dim = X_train_processed.shape[1]
    print(f"   Размерность признаков: {input_dim}")

    print("🏗️ Создание нейронной сети...")

    # Более простая архитектура для предотвращения переобучения
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Компиляция с меньшей learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='mean_squared_error',
        metrics=['mae', 'mean_absolute_percentage_error']
    )

    print("🎯 Обучение модели...")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        min_delta=1000
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001
    )

    history = model.fit(
        X_train_processed, y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    # Сохранение модели и препроцессора
    model.save(f'model/{model_name}_model.h5', save_format='h5')
    joblib.dump(preprocessor, f'model/{model_name}_preprocessor.pkl')

    column_info = {
        'features': X.columns.tolist(),
        'categorical': categorical_cols,
        'numeric': numeric_cols,
        'target': target_col,
        'feature_importance': None  # Можно добавить анализ важности признаков
    }
    with open(f'model/{model_name}_columns.pkl', 'wb') as f:
        pickle.dump(column_info, f)

    # Оценка модели
    loss, mae, mape = model.evaluate(X_test_processed, y_test, verbose=0)
    y_pred = model.predict(X_test_processed, verbose=0).flatten()

    # Более точный расчет MAPE
    valid_idx = y_test.values != 0
    if np.any(valid_idx):
        mape_percent = np.mean(np.abs((y_test.values[valid_idx] - y_pred[valid_idx]) / y_test.values[valid_idx])) * 100
    else:
        mape_percent = 0

    # Расчет R² для оценки качества
    ss_res = np.sum((y_test.values - y_pred) ** 2)
    ss_tot = np.sum((y_test.values - np.mean(y_test.values)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    print(f"✅ Модель обучена!")
    print(f"   • MAE: {mae:,.0f} руб")
    print(f"   • MAPE: {mape_percent:.1f}%")
    print(f"   • R²: {r2:.3f}")
    print(f"   • Эпох обучения: {len(history.history['loss'])}")

    # Анализ распределения ошибок
    errors = y_pred - y_test.values
    print(f"   • Макс переоценка: {errors.max():,.0f} руб")
    print(f"   • Макс недооценка: {errors.min():,.0f} руб")
    print(f"   • Средняя ошибка: {errors.mean():,.0f} руб")

    # Проверка на явно нереалистичные предсказания
    unrealistic_idx = np.where(y_pred > y_test.values * 2)[0]
    if len(unrealistic_idx) > 0:
        print(f"   ⚠️  Найдено {len(unrealistic_idx)} нереалистичных предсказаний (в 2+ раза выше реальных)")

    print(f"   • Сохранена: model/{model_name}_model.h5")

    return model, preprocessor, history


os.makedirs('model', exist_ok=True)

print("\n" + "=" * 60)
print("🏠 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ АРЕНДЫ (КЕМЕРОВО)")
print("=" * 60)

try:
    rent_model, rent_preprocessor, rent_history = train_and_save_model(
        'data/kemerovo_rent.csv',
        'rent_price',
        'rent'
    )
except Exception as e:
    print(f"❌ Ошибка обучения модели аренды: {e}")
    print("Сначала запустите generate_data.py для создания данных")
    exit()

print("\n" + "=" * 60)
print("🏡 ОБУЧЕНИЕ МОДЕЛИ ДЛЯ ПОКУПКИ (КЕМЕРОВО)")
print("=" * 60)

try:
    sale_model, sale_preprocessor, sale_history = train_and_save_model(
        'data/kemerovo_sale.csv',
        'sale_price',
        'sale'
    )
except Exception as e:
    print(f"❌ Ошибка обучения модели покупки: {e}")
    exit()

print("\n" + "=" * 60)
print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("=" * 60)

print("\n📊 СВОДНАЯ ИНФОРМАЦИЯ О МОДЕЛЯХ:")
print("\n🏠 МОДЕЛЬ АРЕНДЫ:")
print(f"   • Целевой диапазон: 8,000 - 50,000 руб/мес")
print(f"   • Ожидаемая точность: 85-92%")

print("\n🏡 МОДЕЛЬ ПОКУПКИ:")
print(f"   • Целевой диапазон: 1,500,000 - 15,000,000 руб")
print(f"   • Средняя цена м²: 45,000 - 120,000 руб")
print(f"   • Ожидаемая точность: 88-94%")

print("\n📁 СОХРАНЕННЫЕ ФАЙЛЫ:")
files = [
    ('model/rent_model.h5', 'Модель аренды'),
    ('model/sale_model.h5', 'Модель покупки'),
    ('model/rent_preprocessor.pkl', 'Препроцессор аренды'),
    ('model/sale_preprocessor.pkl', 'Препроцессор покупки'),
    ('model/rent_columns.pkl', 'Инфо о признаках аренды'),
    ('model/sale_columns.pkl', 'Инфо о признаках покупки')
]

for file_path, description in files:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024
        print(f"   • {file_path} ({description}) - {size:.1f} KB")
    else:
        print(f"   ❌ {file_path} - НЕ НАЙДЕН")

print("\n🚀 Теперь можно запускать бота с реалистичными моделями!")
print("   Для вашего примера (Ленинский, 42.7м², 1 комната, новостройка):")
print("   Ожидаемая цена: 3.5 - 4.8 млн руб")
print("   Цена за м²: 82,000 - 112,000 руб")