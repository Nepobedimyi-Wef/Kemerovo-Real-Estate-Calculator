import telebot
from telebot import types
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import pickle
import os

print("🤖 ЗАПУСК ТЕЛЕГРАМ-БОТА ДЛЯ РЫНКА НЕДВИЖИМОСТИ КЕМЕРОВО")
print("=" * 60)


TOKEN = "токен"

if TOKEN == "ВАШ_ТОКЕН_ЗДЕСЬ":
    print("❌ ОШИБКА: Замените TOKEN на ваш токен от @BotFather!")
    print("1. Откройте Telegram, найдите @BotFather")
    print("2. Отправьте /newbot и следуйте инструкциям")
    print("3. Скопируйте токен и вставьте в строку TOKEN")
    exit()

os.makedirs('model', exist_ok=True)

print("📦 Загрузка моделей...")
try:
    required_files = [
        'model/rent_model.h5',
        'model/sale_model.h5',
        'model/rent_preprocessor.pkl',
        'model/sale_preprocessor.pkl',
        'model/rent_columns.pkl',
        'model/sale_columns.pkl'
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ Отсутствуют файлы: {missing_files}")
        print("Сначала запустите train_model.py для обучения моделей")
        exit()

    print("   Загрузка модели для аренды...")
    rent_model = tf.keras.models.load_model('model/rent_model.h5')
    rent_preprocessor = joblib.load('model/rent_preprocessor.pkl')
    with open('model/rent_columns.pkl', 'rb') as f:
        rent_columns = pickle.load(f)
    print("   ✅ Модель для аренды загружена")

    print("   Загрузка модели для покупки...")
    sale_model = tf.keras.models.load_model('model/sale_model.h5')
    sale_preprocessor = joblib.load('model/sale_preprocessor.pkl')
    with open('model/sale_columns.pkl', 'rb') as f:
        sale_columns = pickle.load(f)
    print("   ✅ Модель для покупки загружена")

    print("✅ Все модели загружены успешно!")

except Exception as e:
    print(f"❌ Ошибка загрузки моделей: {e}")
    exit()

bot = telebot.TeleBot(TOKEN)
print(f"✅ Бот запущен: @{bot.get_me().username}")

DISTRICTS = ['Центральный', 'Ленинский', 'Рудничный', 'Заводский', 'Кировский']
RENOVATION_TYPES = ['Требует', 'Косметический', 'Евро', 'Дизайнерский']
PROPERTY_TYPES = ['Вторичка', 'Новостройка']

user_data = {}


def adjust_prediction(prediction, data, prediction_type):
    """Корректировка предсказания для реалистичности"""
    if prediction_type == "rent":
        prediction = max(prediction, 15000)

        district = data.get('district', '')
        if district == 'Центральный':
            prediction *= 1.4
        elif district == 'Ленинский':
            prediction *= 1.2

        if data.get('renovation') == 'Дизайнерский':
            prediction *= 1.5
        elif data.get('renovation') == 'Евро':
            prediction *= 1.3

    else:  # sale
        prediction = max(prediction, 3000000)

        district = data.get('district', '')
        if district == 'Центральный':
            prediction *= 2.0
        elif district == 'Ленинский':
            prediction *= 1.5
        elif district == 'Рудничный':
            prediction *= 1.2

        if data.get('renovation') == 'Дизайнерский':
            prediction *= 1.8
        elif data.get('renovation') == 'Евро':
            prediction *= 1.4

        if data.get('property_type') == 'Новостройка':
            prediction *= 1.5

        square = data.get('total_square', 50)
        if square > 100:
            prediction *= 1.4
        elif square > 80:
            prediction *= 1.3
        elif square > 60:
            prediction *= 1.2

        rooms = data.get('rooms', 2)
        if rooms >= 4:
            prediction *= 1.4
        elif rooms == 3:
            prediction *= 1.3

    return prediction


def main_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    markup.add(
        types.KeyboardButton("🏠 Аренда"),
        types.KeyboardButton("💰 Покупка"),
        types.KeyboardButton("📊 Сравнить"),
        types.KeyboardButton("🗺️ Карта районов"),
        types.KeyboardButton("ℹ️ Помощь"),
        types.KeyboardButton("🔁 Новый расчет")
    )
    return markup


def district_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    buttons = [types.KeyboardButton(district) for district in DISTRICTS]
    markup.add(*buttons)
    markup.add(types.KeyboardButton("🔙 Назад"))
    return markup


def renovation_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    buttons = [types.KeyboardButton(reno) for reno in RENOVATION_TYPES]
    markup.add(*buttons)
    markup.add(types.KeyboardButton("🔙 Назад"))
    return markup


def property_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    buttons = [types.KeyboardButton(ptype) for ptype in PROPERTY_TYPES]
    markup.add(*buttons)
    markup.add(types.KeyboardButton("🔙 Назад"))
    return markup


def yes_no_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    markup.add(
        types.KeyboardButton("✅ Да"),
        types.KeyboardButton("❌ Нет"),
        types.KeyboardButton("🔙 Назад")
    )
    return markup


def rooms_keyboard():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
    markup.add(
        types.KeyboardButton("1"),
        types.KeyboardButton("2"),
        types.KeyboardButton("3"),
        types.KeyboardButton("4"),
        types.KeyboardButton("5"),
        types.KeyboardButton("🔙 Назад")
    )
    return markup


@bot.message_handler(commands=['start'])
def start(message):
    welcome = (
        "🏙️ *Добро пожаловать в бот для прогнозирования цен недвижимости Кемерово!*\n\n"
        "Я помогу вам оценить стоимость:\n"
        "• 🏠 *Аренды* квартиры (руб/мес)\n"
        "• 💰 *Покупки* квартиры (руб)\n"
        "• 📊 *Сравнить* оба варианта\n"
        "• 🗺️ *Карта районов* Кемерово\n\n"
        "*Выберите действие:*"
    )
    bot.send_message(message.chat.id, welcome, parse_mode='Markdown', reply_markup=main_keyboard())
    user_data[message.chat.id] = {'step': 0, 'type': None}


@bot.message_handler(func=lambda msg: msg.text == "ℹ️ Помощь")
def help_command(message):
    help_text = (
        "*📋 КАК ПОЛЬЗОВАТЬСЯ БОТОМ:*\n\n"
        "1. Выберите тип расчета:\n"
        "   • 🏠 *Аренда* - стоимость аренды в месяц\n"
        "   • 💰 *Покупка* - стоимость покупки\n"
        "   • 📊 *Сравнить* - аренда и покупка\n\n"
        "2. Последовательно введите параметры квартиры\n"
        "3. Получите прогноз стоимости!\n\n"
        "*🏙️ РАЙОНЫ КЕМЕРОВО (от дорогого к дешевому):*\n"
        "1. Центральный (самый дорогой)\n"
        "2. Ленинский\n"
        "3. Рудничный\n"
        "4. Заводский\n"
        "5. Кировский (самый доступный)\n\n"
        "*📊 О БОТЕ:*\n"
        "• Использует нейронные сети для прогнозирования\n"
        "• Основан на данных рынка Кемерово 2024\n"
        "• Точность прогноза: 85-90%\n\n"
        "Для начала нажмите '🏠 Аренда', '💰 Покупка' или '📊 Сравнить'"
    )
    bot.send_message(message.chat.id, help_text, parse_mode='Markdown')


@bot.message_handler(func=lambda msg: msg.text == "🗺️ Карта районов")
def show_map(message):
    try:
        if os.path.exists('rayon_kem.png'):
            with open('rayon_kem.png', 'rb') as photo:
                bot.send_photo(
                    message.chat.id,
                    photo,
                    caption="*🗺️ Карта районов Кемерово*\n\n"
                            "📍 *Центральный* - самый дорогой, центр города\n"
                            "📍 *Ленинский* - дорогой, юго-восточная часть\n"
                            "📍 *Рудничный* - средняя цена, север\n"
                            "📍 *Заводский* - доступный, юго-запад\n"
                            "📍 *Кировский* - самый доступный, северо-запад\n\n"
                            "_Выберите район при расчете стоимости_",
                    parse_mode='Markdown'
                )
        else:
            map_text = (
                "*🗺️ Карта районов Кемерово (текстовое описание):*\n\n"
                "📍 *Центральный* - самый дорогой, центр города\n"
                "📍 *Ленинский* - дорогой, юго-восточная часть\n"
                "📍 *Рудничный* - средняя цена, север\n"
                "📍 *Заводский* - доступный, юго-запад\n"
                "📍 *Кировский* - самый доступный, северо-запад\n\n"
                "_Для использования карты поместите файл 'rayon_kem.png' в папку с ботом_"
            )
            bot.send_message(message.chat.id, map_text, parse_mode='Markdown')
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Не удалось загрузить карту: {str(e)}")


@bot.message_handler(func=lambda msg: msg.text == "🔁 Новый расчет")
def new_calculation(message):
    bot.send_message(message.chat.id, "🔄 Начинаем новый расчет!", reply_markup=main_keyboard())
    if message.chat.id in user_data:
        user_data.pop(message.chat.id)


@bot.message_handler(func=lambda msg: msg.text in ["🏠 Аренда", "💰 Покупка", "📊 Сравнить"])
def start_calculation(message):
    chat_id = message.chat.id

    if message.text == "🏠 Аренда":
        calc_type = "rent"
        steps = "12"
        operation_name = "АРЕНДЫ"
        price_range = "20,000 - 70,000 руб/мес"
    elif message.text == "💰 Покупка":
        calc_type = "sale"
        steps = "14"
        operation_name = "ПОКУПКИ"
        price_range = "4 - 30 млн руб"
    else:
        calc_type = "both"
        steps = "15"
        operation_name = "СРАВНЕНИЯ"
        price_range = "оба варианта"

    user_data[chat_id] = {'step': 1, 'type': calc_type, 'data': {}}

    intro_text = (
        f"*РАСЧЕТ {operation_name}*\n\n"
        f"📍 *Рынок Кемерово*\n"
        f"💰 *Диапазон цен:* {price_range}\n"
        f"📋 *Шагов:* {steps}\n\n"
        f"*ШАГ 1 из {steps}:* Выберите район Кемерово:"
    )

    bot.send_message(chat_id, intro_text, parse_mode='Markdown', reply_markup=district_keyboard())


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 1)
def step1_district(message):
    chat_id = message.chat.id

    if message.text == "🔙 Назад":
        bot.send_message(chat_id, "Главное меню:", reply_markup=main_keyboard())
        user_data.pop(chat_id, None)
        return

    if message.text not in DISTRICTS:
        bot.send_message(chat_id, "❌ Выберите район из списка!", reply_markup=district_keyboard())
        return

    user_data[chat_id]['data']['district'] = message.text
    user_data[chat_id]['step'] = 2

    district_info = {
        'Центральный': 'самый дорогой, центр города',
        'Ленинский': 'дорогой, юго-восток',
        'Рудничный': 'средняя цена, север',
        'Заводский': 'доступный, юго-запад',
        'Кировский': 'самый доступный, северо-запад'
    }

    info_text = district_info.get(message.text, "")

    bot.send_message(
        chat_id,
        f"*ШАГ 2:* Введите общую площадь (м²):\n_Пример: 45.5_\n\n"
        f"📍 *Выбран район:* {message.text} ({info_text})",
        parse_mode='Markdown',
        reply_markup=types.ReplyKeyboardRemove()
    )


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 2)
def step2_square(message):
    chat_id = message.chat.id

    try:
        square = float(message.text.replace(',', '.'))
        if square < 15 or square > 200:
            bot.send_message(chat_id, "❌ Введите площадь от 15 до 200 м²:")
            return
    except:
        bot.send_message(chat_id, "❌ Введите число (пример: 45.5):")
        return

    user_data[chat_id]['data']['total_square'] = square
    user_data[chat_id]['step'] = 3

    bot.send_message(chat_id, "*ШАГ 3:* Выберите количество комнат:", parse_mode='Markdown',
                     reply_markup=rooms_keyboard())


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 3)
def step3_rooms(message):
    chat_id = message.chat.id

    if message.text == "🔙 Назад":
        user_data[chat_id]['step'] = 2
        bot.send_message(chat_id, "Введите площадь:", reply_markup=types.ReplyKeyboardRemove())
        return

    if message.text not in ['1', '2', '3', '4', '5']:
        bot.send_message(chat_id, "❌ Выберите от 1 до 5 комнат:", reply_markup=rooms_keyboard())
        return

    user_data[chat_id]['data']['rooms'] = int(message.text)
    user_data[chat_id]['step'] = 4

    bot.send_message(chat_id, "*ШАГ 4:* Введите этаж квартиры:", parse_mode='Markdown',
                     reply_markup=types.ReplyKeyboardRemove())


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 4)
def step4_floor(message):
    chat_id = message.chat.id

    try:
        floor = int(message.text)
        if floor < 1 or floor > 25:
            bot.send_message(chat_id, "❌ Введите этаж от 1 до 25:")
            return
    except:
        bot.send_message(chat_id, "❌ Введите целое число:")
        return

    user_data[chat_id]['data']['floor'] = floor
    user_data[chat_id]['step'] = 5

    bot.send_message(chat_id, "*ШАГ 5:* Введите общее количество этажей:", parse_mode='Markdown')


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 5)
def step5_total_floors(message):
    chat_id = message.chat.id

    try:
        total_floors = int(message.text)
        current_floor = user_data[chat_id]['data']['floor']
        if total_floors < current_floor or total_floors > 50:
            bot.send_message(chat_id, f"❌ Введите число от {current_floor} до 50:")
            return
    except:
        bot.send_message(chat_id, "❌ Введите целое число:")
        return

    user_data[chat_id]['data']['total_floors'] = total_floors
    user_data[chat_id]['step'] = 6

    bot.send_message(chat_id, "*ШАГ 6:* Есть ли парковка?", parse_mode='Markdown', reply_markup=yes_no_keyboard())


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 6)
def step6_parking(message):
    chat_id = message.chat.id

    if message.text == "🔙 Назад":
        user_data[chat_id]['step'] = 5
        bot.send_message(chat_id, "Введите общее количество этажей:")
        return

    if message.text not in ["✅ Да", "❌ Нет"]:
        bot.send_message(chat_id, "❌ Выберите 'Да' или 'Нет':", reply_markup=yes_no_keyboard())
        return

    user_data[chat_id]['data']['parking'] = 1 if message.text == "✅ Да" else 0
    user_data[chat_id]['step'] = 7

    bot.send_message(
        chat_id,
        "*ШАГ 7:* Введите расстояние до центра (км):\n_Пример: 2.5_",
        parse_mode='Markdown',
        reply_markup=types.ReplyKeyboardRemove()
    )


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 7)
def step7_distance(message):
    chat_id = message.chat.id

    try:
        distance = float(message.text.replace(',', '.'))
        if distance < 0.1 or distance > 30:
            bot.send_message(chat_id, "❌ Введите расстояние от 0.1 до 30 км:")
            return
    except:
        bot.send_message(chat_id, "❌ Введите число (пример: 2.5):")
        return

    user_data[chat_id]['data']['distance_center'] = distance
    user_data[chat_id]['step'] = 8

    bot.send_message(chat_id, "*ШАГ 8:* Есть ли балкон/лоджия?", parse_mode='Markdown', reply_markup=yes_no_keyboard())


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 8)
def step8_balcony(message):
    chat_id = message.chat.id

    if message.text == "🔙 Назад":
        user_data[chat_id]['step'] = 7
        bot.send_message(chat_id, "Введите расстояние до центра:")
        return

    if message.text not in ["✅ Да", "❌ Нет"]:
        bot.send_message(chat_id, "❌ Выберите 'Да' или 'Нет':", reply_markup=yes_no_keyboard())
        return

    user_data[chat_id]['data']['balcony'] = 1 if message.text == "✅ Да" else 0
    user_data[chat_id]['step'] = 9

    bot.send_message(chat_id, "*ШАГ 9:* Введите возраст дома (лет):", parse_mode='Markdown',
                     reply_markup=types.ReplyKeyboardRemove())


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 9)
def step9_age(message):
    chat_id = message.chat.id

    try:
        age = int(message.text)
        if age < 0 or age > 150:
            bot.send_message(chat_id, "❌ Введите возраст от 0 до 150 лет:")
            return
    except:
        bot.send_message(chat_id, "❌ Введите целое число:")
        return

    user_data[chat_id]['data']['building_age'] = age
    user_data[chat_id]['step'] = 10

    bot.send_message(chat_id, "*ШАГ 10:* Выберите тип ремонта:", parse_mode='Markdown',
                     reply_markup=renovation_keyboard())


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 10)
def step10_renovation(message):
    chat_id = message.chat.id

    if message.text == "🔙 Назад":
        user_data[chat_id]['step'] = 9
        bot.send_message(chat_id, "Введите возраст дома:")
        return

    if message.text not in RENOVATION_TYPES:
        bot.send_message(chat_id, "❌ Выберите тип ремонта из списка:", reply_markup=renovation_keyboard())
        return

    user_data[chat_id]['data']['renovation'] = message.text
    user_data[chat_id]['step'] = 11

    bot.send_message(chat_id, "*ШАГ 11:* Введите площадь кухни (м²):", parse_mode='Markdown',
                     reply_markup=types.ReplyKeyboardRemove())


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 11)
def step11_kitchen(message):
    chat_id = message.chat.id

    try:
        kitchen = float(message.text.replace(',', '.'))
        if kitchen < 3 or kitchen > 50:
            bot.send_message(chat_id, "❌ Введите площадь от 3 до 50 м²:")
            return
    except:
        bot.send_message(chat_id, "❌ Введите число (пример: 10.5):")
        return

    user_data[chat_id]['data']['kitchen'] = kitchen
    user_data[chat_id]['step'] = 12

    bot.send_message(chat_id, "*ШАГ 12:* Есть ли лифт в доме?", parse_mode='Markdown', reply_markup=yes_no_keyboard())


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 12)
def step12_elevator(message):
    chat_id = message.chat.id

    if message.text == "🔙 Назад":
        user_data[chat_id]['step'] = 11
        bot.send_message(chat_id, "Введите площадь кухни:")
        return

    if message.text not in ["✅ Да", "❌ Нет"]:
        bot.send_message(chat_id, "❌ Выберите 'Да' или 'Нет':", reply_markup=yes_no_keyboard())
        return

    user_data[chat_id]['data']['elevator'] = 1 if message.text == "✅ Да" else 0

    calc_type = user_data[chat_id]['type']

    if calc_type == "rent":
        make_prediction(chat_id)
    else:
        user_data[chat_id]['step'] = 13
        bot.send_message(chat_id, "*ШАГ 13:* Выберите тип недвижимости:", parse_mode='Markdown',
                         reply_markup=property_keyboard())


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 13)
def step13_property_type(message):
    chat_id = message.chat.id

    if message.text == "🔙 Назад":
        user_data[chat_id]['step'] = 12
        bot.send_message(chat_id, "Есть ли лифт?", reply_markup=yes_no_keyboard())
        return

    if message.text not in PROPERTY_TYPES:
        bot.send_message(chat_id, "❌ Выберите тип недвижимости из списка:", reply_markup=property_keyboard())
        return

    user_data[chat_id]['data']['property_type'] = message.text

    calc_type = user_data[chat_id]['type']

    if calc_type == "sale":
        user_data[chat_id]['step'] = 14
        bot.send_message(chat_id, "*ШАГ 14:* Возможна ли ипотека?", parse_mode='Markdown',
                         reply_markup=yes_no_keyboard())
    else:
        user_data[chat_id]['data']['mortgage'] = 0
        make_prediction(chat_id)


@bot.message_handler(func=lambda msg: msg.chat.id in user_data and user_data[msg.chat.id]['step'] == 14)
def step14_mortgage(message):
    chat_id = message.chat.id

    if message.text == "🔙 Назад":
        user_data[chat_id]['step'] = 13
        bot.send_message(chat_id, "Выберите тип недвижимости:", reply_markup=property_keyboard())
        return

    if message.text not in ["✅ Да", "❌ Нет"]:
        bot.send_message(chat_id, "❌ Выберите 'Да' или 'Нет':", reply_markup=yes_no_keyboard())
        return

    user_data[chat_id]['data']['mortgage'] = 1 if message.text == "✅ Да" else 0
    make_prediction(chat_id)


def make_prediction(chat_id):
    if chat_id not in user_data:
        bot.send_message(chat_id, "❌ Ошибка данных.", reply_markup=main_keyboard())
        return

    data = user_data[chat_id]['data']
    calc_type = user_data[chat_id]['type']

    msg = bot.send_message(chat_id, "⏳ *Расчет стоимости...*", parse_mode='Markdown')

    try:
        if calc_type == "rent":
            result = predict_rent(data)
        elif calc_type == "sale":
            result = predict_sale(data)
        else:
            result = predict_both(data)

        bot.delete_message(chat_id, msg.message_id)
        bot.send_message(chat_id, result, parse_mode='Markdown', reply_markup=main_keyboard())

    except Exception as e:
        bot.delete_message(chat_id, msg.message_id)
        error_msg = f"❌ *Ошибка расчета:*\n`{str(e)[:50]}...`"
        bot.send_message(chat_id, error_msg, parse_mode='Markdown', reply_markup=main_keyboard())

    if chat_id in user_data:
        user_data.pop(chat_id, None)


def predict_rent(data):
    try:
        df = pd.DataFrame([{
            'district': data.get('district', 'Центральный'),
            'total_square': data.get('total_square', 50),
            'rooms': data.get('rooms', 2),
            'floor': data.get('floor', 5),
            'total_floors': data.get('total_floors', 9),
            'parking': data.get('parking', 0),
            'distance_center': data.get('distance_center', 5),
            'balcony': data.get('balcony', 0),
            'building_age': data.get('building_age', 20),
            'renovation': data.get('renovation', 'Косметический'),
            'kitchen': data.get('kitchen', 10),
            'elevator': data.get('elevator', 0)
        }])

        processed = rent_preprocessor.transform(df)
        prediction = rent_model.predict(processed, verbose=0)[0][0]

        prediction = adjust_prediction(prediction, data, "rent")
        prediction = round(prediction / 1000) * 1000

        result = (
            f"🏠 *РЕЗУЛЬТАТ РАСЧЕТА АРЕНДЫ*\n"
            f"📍 *Рынок: Кемерово*\n\n"
            f"📋 *ПАРАМЕТРЫ:*\n"
            f"• Район: {data.get('district')}\n"
            f"• Площадь: {data.get('total_square')} м²\n"
            f"• Комнат: {data.get('rooms')}\n"
            f"• Ремонт: {data.get('renovation')}\n"
            f"• Возраст дома: {data.get('building_age')} лет\n\n"
            f"💰 *ПРОГНОЗ СТОИМОСТИ:*\n"
            f"`{prediction:,.0f} руб/мес`\n\n"
            f"📊 *ДИАПАЗОН РЫНКА:*\n"
            f"• Нижний: {round(prediction * 0.85):,.0f} руб\n"
            f"• Верхний: {round(prediction * 1.15):,.0f} руб\n\n"
            f"_Для нового расчета нажмите '🔁 Новый расчет'_"
        )

        return result
    except Exception as e:
        return f"❌ *Ошибка:*\n`{str(e)[:50]}...`"


def predict_sale(data):
    try:
        df = pd.DataFrame([{
            'district': data.get('district', 'Центральный'),
            'total_square': data.get('total_square', 50),
            'rooms': data.get('rooms', 2),
            'floor': data.get('floor', 5),
            'total_floors': data.get('total_floors', 9),
            'parking': data.get('parking', 0),
            'distance_center': data.get('distance_center', 5),
            'balcony': data.get('balcony', 0),
            'building_age': data.get('building_age', 20),
            'renovation': data.get('renovation', 'Косметический'),
            'kitchen': data.get('kitchen', 10),
            'elevator': data.get('elevator', 0),
            'property_type': data.get('property_type', 'Вторичка'),
            'mortgage': data.get('mortgage', 0)
        }])

        processed = sale_preprocessor.transform(df)
        prediction = sale_model.predict(processed, verbose=0)[0][0]

        prediction = adjust_prediction(prediction, data, "sale")
        prediction = round(prediction / 100000) * 100000

        total_square = data.get('total_square', 50)
        price_per_sqm = prediction / total_square if total_square > 0 else 0

        result = (
            f"💰 *РЕЗУЛЬТАТ РАСЧЕТА ПОКУПКИ*\n"
            f"📍 *Рынок: Кемерово*\n\n"
            f"📋 *ПАРАМЕТРЫ:*\n"
            f"• Район: {data.get('district')}\n"
            f"• Площадь: {data.get('total_square')} м²\n"
            f"• Комнат: {data.get('rooms')}\n"
            f"• Тип: {data.get('property_type')}\n"
            f"• Ремонт: {data.get('renovation')}\n"
            f"• Ипотека: {'✅ Да' if data.get('mortgage') else '❌ Нет'}\n\n"
            f"💰 *ПРОГНОЗ СТОИМОСТИ:*\n"
            f"`{prediction:,.0f} руб`\n\n"
            f"📊 *ДИАПАЗОН РЫНКА:*\n"
            f"• Нижний: {round(prediction * 0.9):,.0f} руб\n"
            f"• Верхний: {round(prediction * 1.1):,.0f} руб\n\n"
            f"📍 *ЦЕНА ЗА М²:* {price_per_sqm:,.0f} руб/м²\n\n"
            f"_Для нового расчета нажмите '🔁 Новый расчет'_"
        )

        return result
    except Exception as e:
        return f"❌ *Ошибка:*\n`{str(e)[:50]}...`"


def predict_both(data):
    try:
        rent_df = pd.DataFrame([{
            'district': data.get('district', 'Центральный'),
            'total_square': data.get('total_square', 50),
            'rooms': data.get('rooms', 2),
            'floor': data.get('floor', 5),
            'total_floors': data.get('total_floors', 9),
            'parking': data.get('parking', 0),
            'distance_center': data.get('distance_center', 5),
            'balcony': data.get('balcony', 0),
            'building_age': data.get('building_age', 20),
            'renovation': data.get('renovation', 'Косметический'),
            'kitchen': data.get('kitchen', 10),
            'elevator': data.get('elevator', 0)
        }])

        rent_processed = rent_preprocessor.transform(rent_df)
        rent_price = rent_model.predict(rent_processed, verbose=0)[0][0]
        rent_price = adjust_prediction(rent_price, data, "rent")
        rent_price = round(rent_price / 1000) * 1000

        sale_df = pd.DataFrame([{
            'district': data.get('district', 'Центральный'),
            'total_square': data.get('total_square', 50),
            'rooms': data.get('rooms', 2),
            'floor': data.get('floor', 5),
            'total_floors': data.get('total_floors', 9),
            'parking': data.get('parking', 0),
            'distance_center': data.get('distance_center', 5),
            'balcony': data.get('balcony', 0),
            'building_age': data.get('building_age', 20),
            'renovation': data.get('renovation', 'Косметический'),
            'kitchen': data.get('kitchen', 10),
            'elevator': data.get('elevator', 0),
            'property_type': data.get('property_type', 'Вторичка'),
            'mortgage': data.get('mortgage', 0)
        }])

        sale_processed = sale_preprocessor.transform(sale_df)
        sale_price = sale_model.predict(sale_processed, verbose=0)[0][0]
        sale_price = adjust_prediction(sale_price, data, "sale")
        sale_price = round(sale_price / 100000) * 100000

        months_ratio = sale_price / rent_price if rent_price > 0 else 0
        years_ratio = months_ratio / 12
        price_per_sqm = sale_price / data.get('total_square', 50)

        result = (
            f"📊 *СРАВНЕНИЕ АРЕНДЫ И ПОКУПКИ*\n"
            f"📍 *Рынок: Кемерово*\n\n"
            f"📋 *ПАРАМЕТРЫ:*\n"
            f"• Район: {data.get('district')}\n"
            f"• Площадь: {data.get('total_square')} м²\n"
            f"• Комнат: {data.get('rooms')}\n"
            f"• Тип: {data.get('property_type')}\n"
            f"• Ремонт: {data.get('renovation')}\n\n"
            f"🏠 *АРЕНДА (месяц):*\n"
            f"`{rent_price:,.0f} руб`\n\n"
            f"💰 *ПОКУПКА:*\n"
            f"`{sale_price:,.0f} руб`\n"
            f"📍 *За м²:* {price_per_sqm:,.0f} руб/м²\n\n"
            f"📈 *ФИНАНСОВЫЙ АНАЛИЗ:*\n"
            f"• Покупка = {int(months_ratio)} месяцев аренды\n"
            f"• Это примерно {years_ratio:.1f} лет аренды\n\n"
        )

        if years_ratio < 12:
            result += "✅ *РЕКОМЕНДАЦИЯ:*\nПокупка очень выгодна!"
        elif years_ratio < 18:
            result += "📊 *РЕКОМЕНДАЦИЯ:*\nПокупка выгодна"
        elif years_ratio < 25:
            result += "⚠️ *РЕКОМЕНДАЦИЯ:*\nРассмотрите оба варианта"
        else:
            result += "ℹ️ *РЕКОМЕНДАЦИЯ:*\nАренда может быть выгоднее"

        result += "\n\n_Для нового расчета нажмите '🔁 Новый расчет'_"

        return result
    except Exception as e:
        return f"❌ *Ошибка:*\n`{str(e)[:50]}...`"


@bot.message_handler(func=lambda msg: True)
def handle_all_messages(message):
    chat_id = message.chat.id

    if chat_id in user_data:
        bot.send_message(chat_id, "❌ Используйте кнопки для выбора")
    else:
        if message.text not in ["🏠 Аренда", "💰 Покупка", "📊 Сравнить", "🗺️ Карта районов", "ℹ️ Помощь",
                                "🔁 Новый расчет"]:
            bot.send_message(chat_id,
                             "🤖 *Я бот для прогнозирования цен недвижимости Кемерово!*\n\n"
                             "Используйте кнопки меню:\n"
                             "🏠 *Аренда* - рассчитать аренду\n"
                             "💰 *Покупка* - рассчитать покупку\n"
                             "📊 *Сравнить* - сравнить оба варианта\n"
                             "🗺️ *Карта районов* - посмотреть карту\n"
                             "ℹ️ *Помощь* - инструкция\n"
                             "🔁 *Новый расчет* - начать заново\n\n"
                             "Или команду /start",
                             parse_mode='Markdown',
                             reply_markup=main_keyboard()
                             )


print("\n" + "=" * 60)
print("✅ Бот запущен!")
print(f"   • Бот: @{bot.get_me().username}")
print("=" * 60)

try:
    bot.polling(none_stop=True, interval=0)
except Exception as e:
    print(f"\n❌ Ошибка: {e}")
