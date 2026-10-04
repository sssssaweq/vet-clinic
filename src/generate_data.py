"""
Скрипт генерации синтетических данных для системы учёта
ветеринарных осмотров (молочное направление КРС, голштинская порода).

Создаёт:
- 100 животных (голштинская порода)
- 500 осмотров
- 10 ветеринаров
- 21 диагноз (20 болезней + "Здоров")

Результат: CSV-файлы в папке data/
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Фиксируем seed для воспроизводимости
random.seed(42)
np.random.seed(42)

# ============ НАСТРОЙКИ ============
N_ANIMALS = 100
N_EXAMINATIONS = 500
N_VETS = 10
DATA_DIR = "data"

# Доля здоровых осмотров (30%)
HEALTHY_RATE = 0.30

BREED = "Голштинская"

NICKNAMES = [
    "Бурёнка", "Зорька", "Милка", "Ночка", "Ласточка",
    "Ромашка", "Пеструшка", "Чернушка", "Белка", "Дочка",
    "Красавка", "Умница", "Солнышко", "Звёздочка", "Рыжуха",
    "Пятнушка", "Марта", "Галка", "Снежинка", "Веснушка"
]

# Справочник диагнозов: (название, хронический)
# Первый — "Здоров" (не болезнь)
DIAGNOSES = [
    ("Здоров", 0),          # diagnosis_id = 1
    ("Мастит", 0),          # 2
    ("Ламинит", 1),         # 3
    ("Эндометрит", 0),      # 4
    ("Кетоз", 0),           # 5
    ("Ацидоз рубца", 0),    # 6
    ("Тимпания рубца", 0),  # 7
    ("Бронхопневмония", 0), # 8
    ("Паратуберкулез", 1),  # 9
    ("Лейкоз", 1),          # 10
    ("Туберкулез", 1),      # 11
    ("Бруцеллез", 1),       # 12
    ("Ящур", 0),            # 13
    ("Некробактериоз", 0),  # 14
    ("Копытная гниль", 0),  # 15
    ("Отит", 0),            # 16
    ("Конъюнктивит", 0),    # 17
    ("Авитаминоз", 0),      # 18
    ("Гипокальциемия", 0),  # 19
    ("Травматический ретикулит", 0),  # 20
    ("Пироплазмоз", 0),     # 21
]

TREATMENTS = [
    "Антибиотик", "Витамины", "Противовоспалительное",
    "Покой", "Смена корма", "Обработка копыт",
    "Гормональная терапия", "Инфузия", "Пробиотики"
]

VET_NAMES = [
    "Иванов И.И.", "Петрова А.С.", "Сидоров П.Н.",
    "Кузнецова М.В.", "Смирнов Д.А.", "Волкова Е.П.",
    "Морозов С.С.", "Новикова О.А.", "Фёдоров К.Л.",
    "Соколова Т.В."
]


# ============ ГЕНЕРАЦИЯ ============

def generate_vets():
    """Создаёт таблицу ветеринаров."""
    vets = []
    for i, name in enumerate(VET_NAMES[:N_VETS], start=1):
        vets.append({
            "vet_id": i,
            "full_name": name,
            "phone": f"+7-900-{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(10,99)}"
        })
    return pd.DataFrame(vets)


def generate_diagnoses():
    """Создаёт справочник диагнозов."""
    diagnoses = []
    for i, (name, is_chronic) in enumerate(DIAGNOSES, start=1):
        diagnoses.append({
            "diagnosis_id": i,
            "name": name,
            "is_chronic": is_chronic
        })
    return pd.DataFrame(diagnoses)


def generate_animals():
    """Создаёт таблицу животных"""
    animals = []
    for i in range(1, N_ANIMALS + 1):
        age = random.randint(2, 10)
        lactation_number = min(age - 1, random.randint(1, 5))
        milk_yield = random.randint(6000, 10000)
        lameness_score = random.choices(
            [0, 1, 2, 3, 4, 5],
            weights=[50, 25, 15, 5, 3, 2]
        )[0]

        animals.append({
            "animal_id": i,
            "nickname": random.choice(NICKNAMES),
            "number": f"{i:03d}",
            "age": age,
            "breed": BREED,
            "gender": "female",
            "lactation_number": lactation_number,
            "milk_yield": milk_yield,
            "lameness_score": lameness_score
        })
    return pd.DataFrame(animals)


def random_date(start: str, end: str) -> str:
    """Случайная дата между start и end."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    delta = end_dt - start_dt
    random_days = random.randint(0, delta.days)
    return (start_dt + timedelta(days=random_days)).strftime("%Y-%m-%d")


def generate_examinations(animals_df):
    """
    Создаёт таблицу осмотров с целевой переменной is_sick.
    
    Логика is_sick:
    - 30% осмотров — "Здоров" (is_sick = 0)
    - 70% осмотров — болезнь (is_sick = 1)
    
    При этом вероятность болезни зависит от признаков животного:
    чем выше хромота, возраст, число лактаций — тем чаще болеет.
    """
    examinations = []

    # Создаём словарь животных для быстрого доступа
    animals_dict = animals_df.set_index("animal_id").to_dict("index")

    for i in range(1, N_EXAMINATIONS + 1):
        animal_id = random.randint(1, N_ANIMALS)
        animal = animals_dict[animal_id]

        # Вычисляем "склонность к болезни" на основе признаков
        # Чем выше — тем вероятнее болезнь
        risk_score = (
            animal["lameness_score"] * 1.5 +
            (animal["age"] - 2) * 0.3 +
            animal["lactation_number"] * 0.4
        )

        # Нормализуем в вероятность от 0.2 до 0.9
        sick_probability = min(0.9, max(0.2, 0.3 + risk_score * 0.05))

        # Определяем is_sick
        is_sick = 1 if random.random() < sick_probability else 0

        # Если болен — случайный диагноз (кроме "Здоров")
        # Если здоров — диагноз "Здоров" (diagnosis_id = 1)
        if is_sick:
            diagnosis_id = random.randint(2, len(DIAGNOSES))
            treatment = random.choice(TREATMENTS)
        else:
            diagnosis_id = 1  # "Здоров"
            treatment = "—"

        examinations.append({
            "examination_id": i,
            "animal_id": animal_id,
            "vet_id": random.randint(1, N_VETS),
            "diagnosis_id": diagnosis_id,
            "exam_date": random_date("2023-01-01", "2025-10-01"),
            "treatment": treatment,
            "is_sick": is_sick
        })

    return pd.DataFrame(examinations)


# ============ MAIN ============

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    vets = generate_vets()
    diagnoses = generate_diagnoses()
    animals = generate_animals()
    examinations = generate_examinations(animals)

    # utf-8-sig — чтобы Excel правильно открывал русские буквы
    vets.to_csv(f"{DATA_DIR}/vets.csv", index=False, encoding="utf-8-sig")
    diagnoses.to_csv(f"{DATA_DIR}/diagnoses.csv", index=False, encoding="utf-8-sig")
    animals.to_csv(f"{DATA_DIR}/animals.csv", index=False, encoding="utf-8-sig")
    examinations.to_csv(f"{DATA_DIR}/examinations.csv", index=False, encoding="utf-8-sig")

    # Статистика
    sick_count = examinations["is_sick"].sum()
    healthy_count = len(examinations) - sick_count

    print("=" * 50)
    print("Данные сгенерированы:")
    print(f"  Животные:    {len(animals)} (все — голштинская)")
    print(f"  Осмотры:     {len(examinations)}")
    print(f"  Ветеринары:  {len(vets)}")
    print(f"  Диагнозы:    {len(diagnoses)} (включая 'Здоров')")
    print()
    print(f"  Больных:     {sick_count} ({sick_count/len(examinations)*100:.1f}%)")
    print(f"  Здоровых:    {healthy_count} ({healthy_count/len(examinations)*100:.1f}%)")
    print("=" * 50)
    print(f"Файлы сохранены в папку: {DATA_DIR}/")
    print()
    print("Пример осмотров:")
    print(examinations.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
