import pandas as pd
import numpy as np
import os

print("🔧 ГЕНЕРАЦИЯ РЕАЛИСТИЧНЫХ ДАННЫХ ДЛЯ РЫНКА НЕДВИЖИМОСТИ КЕМЕРОВО")
print("=" * 60)

os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)

DISTRICTS = ['Центральный', 'Ленинский', 'Рудничный', 'Заводский', 'Кировский']

print("\n🏠 Генерация данных для АРЕНДЫ...")
n_samples = 1500

data = {
    'district': np.random.choice(DISTRICTS, n_samples, p=[0.35, 0.25, 0.20, 0.15, 0.05]),
    'total_square': np.random.uniform(20, 90, n_samples).round(1),  
    'rooms': np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.35, 0.12, 0.03]),  
    'floor': np.random.randint(1, 12, n_samples),
    'total_floors': np.random.choice([5, 9, 12, 16], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    'parking': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),  
    'distance_center': np.random.uniform(0.5, 12, n_samples).round(1),
    'balcony': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
    'building_age': np.random.randint(1, 60, n_samples),
    'renovation': np.random.choice(['Требует', 'Косметический', 'Евро', 'Дизайнерский'], n_samples, p=[0.3, 0.5, 0.18, 0.02]),  
    'kitchen': np.random.uniform(5, 20, n_samples).round(1),
    'elevator': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
}

district_coef_rent = {
    'Центральный': 1.25,   
    'Ленинский': 1.15,      
    'Рудничный': 1.08,      
    'Заводский': 1.03,     
    'Кировский': 1.0
}

renovation_coef_rent = {
    'Требует': 0.8,         
    'Косметический': 1.0,
    'Евро': 1.15,          
    'Дизайнерский': 1.25   
}

base_rent = (
    data['total_square'] * 300 +       
    data['rooms'] * 1000 +             
    (12 - data['distance_center']) * 100 + 
    data['parking'] * 800 +           
    data['balcony'] * 500 +           
    data['elevator'] * 400 +          
    data['kitchen'] * 200 -            
    data['building_age'] * 10         
)

data['district_coef'] = [district_coef_rent[d] for d in data['district']]
data['renovation_coef'] = [renovation_coef_rent[r] for r in data['renovation']]

data['rent_price'] = (base_rent * data['district_coef'] * data['renovation_coef'] *
                      np.random.uniform(0.9, 1.1, n_samples))

data['rent_price'] = np.round(data['rent_price'] / 1000) * 1000
data['rent_price'] = np.clip(data['rent_price'], 8000, 50000)  

df_rent = pd.DataFrame(data)
df_rent = df_rent.drop(['district_coef', 'renovation_coef'], axis=1)
df_rent.to_csv('data/kemerovo_rent.csv', index=False, encoding='utf-8')

print("\n🏡 Генерация данных для ПОКУПКИ...")
n_samples_sale = 1200

data_sale = {
    'district': np.random.choice(DISTRICTS, n_samples_sale, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
    'total_square': np.random.uniform(25, 110, n_samples_sale).round(1),  
    'rooms': np.random.choice([1, 2, 3, 4, 5], n_samples_sale, p=[0.4, 0.35, 0.15, 0.07, 0.03]),
    'floor': np.random.randint(1, 16, n_samples_sale),
    'total_floors': np.random.choice([5, 9, 12, 16], n_samples_sale, p=[0.3, 0.4, 0.2, 0.1]),
    'parking': np.random.choice([0, 1], n_samples_sale, p=[0.7, 0.3]),
    'distance_center': np.random.uniform(0.5, 15, n_samples_sale).round(1),
    'balcony': np.random.choice([0, 1], n_samples_sale, p=[0.4, 0.6]),
    'building_age': np.random.randint(1, 80, n_samples_sale),
    'renovation': np.random.choice(['Требует', 'Косметический', 'Евро', 'Дизайнерский'], n_samples_sale, p=[0.25, 0.5, 0.22, 0.03]),  
    'kitchen': np.random.uniform(6, 22, n_samples_sale).round(1),
    'elevator': np.random.choice([0, 1], n_samples_sale, p=[0.4, 0.6]),
    'property_type': np.random.choice(['Вторичка', 'Новостройка'], n_samples_sale, p=[0.75, 0.25]), 
    'mortgage': np.random.choice([0, 1], n_samples_sale, p=[0.3, 0.7]),
}

district_coef_sale = {
    'Центральный': 1.35,    
    'Ленинский': 1.20,     
    'Рудничный': 1.10,     
    'Заводский': 1.05,      
    'Кировский': 1.0
}

property_coef = {
    'Вторичка': 1.0,
    'Новостройка': 1.15     
}

renovation_coef_sale = {
    'Требует': 0.85,        
    'Косметический': 1.0,
    'Евро': 1.20,           
    'Дизайнерский': 1.35   
}

base_price_sale = (
    data_sale['total_square'] * 45000 +      
    data_sale['rooms'] * 150000 +           
    (15 - data_sale['distance_center']) * 15000 + 
    data_sale['parking'] * 150000 +          
    data_sale['balcony'] * 80000 +           
    data_sale['elevator'] * 50000 +          
    data_sale['kitchen'] * 30000 -           
    data_sale['building_age'] * 5000         
)

data_sale['district_coef'] = [district_coef_sale[d] for d in data_sale['district']]
data_sale['renovation_coef'] = [renovation_coef_sale[r] for r in data_sale['renovation']]
data_sale['property_coef'] = [property_coef[p] for p in data_sale['property_type']]

data_sale['sale_price'] = (base_price_sale * data_sale['district_coef'] *
                          data_sale['renovation_coef'] * data_sale['property_coef'] *
                          np.random.uniform(0.85, 1.15, n_samples_sale))

data_sale['sale_price'] = np.round(data_sale['sale_price'] / 10000) * 10000
data_sale['sale_price'] = np.clip(data_sale['sale_price'], 1500000, 15000000)  

df_sale = pd.DataFrame(data_sale)
df_sale = df_sale.drop(['district_coef', 'renovation_coef', 'property_coef'], axis=1)
df_sale.to_csv('data/kemerovo_sale.csv', index=False, encoding='utf-8')

print("\n" + "="*60)
print("✅ РЕАЛИСТИЧНЫЕ ДАННЫЕ ДЛЯ КЕМЕРОВО СОЗДАНЫ!")
print("="*60)

print(f"\n🏠 АРЕНДА (реалистичные цифры для Кемерово):")
print(f"   • Средняя цена: {df_rent['rent_price'].mean():,.0f} руб/мес")
print(f"   • Диапазон: {df_rent['rent_price'].min():,.0f} - {df_rent['rent_price'].max():,.0f} руб/мес")

print(f"\n🏡 ПОКУПКА (реалистичные цифры для Кемерово):")
print(f"   • Средняя цена: {df_sale['sale_price'].mean():,.0f} руб")
print(f"   • Диапазон: {df_sale['sale_price'].min():,.0f} - {df_sale['sale_price'].max():,.0f} руб")
print(f"   • Средняя цена за м²: {(df_sale['sale_price'].mean() / df_sale['total_square'].mean()):,.0f} руб/м²")

print(f"\n🏙️ СРЕДНИЕ ЦЕНЫ ПО РАЙОНАМ КЕМЕРОВО:")
for district in DISTRICTS:
    avg_sale = df_sale[df_sale['district'] == district]['sale_price'].mean()
    avg_sqm = avg_sale / df_sale[df_sale['district'] == district]['total_square'].mean()
    print(f"   • {district}: {avg_sale:,.0f} руб ({avg_sqm:,.0f} руб/м²)")

print("\n📊 СТАТИСТИКА ПО НЕДВИЖИМОСТИ:")
print(f"   • Средняя площадь: {df_sale['total_square'].mean():.1f} м²")
print(f"   • Доля новостроек: {(df_sale['property_type'] == 'Новостройка').mean()*100:.1f}%")
print(f"   • Средний возраст домов: {df_sale['building_age'].mean():.0f} лет")

print("\n🚀 Запустите train_model.py для обучения моделей на реалистичных данных")
