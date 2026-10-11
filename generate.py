
"""
Генератор синтетических данных для информационной системы
учёта ветеринарных осмотров молочного КРС.

Данные специально генерируются как временная история:
- 100 животных;
- 500 последовательных осмотров;
- 10 ветеринаров;
- 21 диагноз (20 болезней + "Здоров");
- период наблюдения: 2023-01-01 — 2025-10-01.

Основная идея:
каждый осмотр является фактическим результатом в определённый момент времени.
В дальнейшем модель сможет использовать предыдущие осмотры животного
для прогноза риска заболевания на следующем осмотре.

Файлы:
    data/animals.csv
    data/vets.csv
    data/diagnoses.csv
    data/examinations.csv
"""

import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd


# ============================================================
# НАСТРОЙКИ
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

N_ANIMALS = 100
N_EXAMINATIONS = 500
N_VETS = 10

DATA_DIR = "data"

START_DATE = date(2023, 1, 1)
END_DATE = date(2025, 10, 1)

BREED = "Голштинская"
GENDER = "female"

# Базовая доля здоровых осмотров.
# Фактическая доля может немного отличаться, поскольку
# вероятность болезни зависит от характеристик животного
# и его истории.
BASE_SICK_PROBABILITY = 0.30


# ============================================================
# СПРАВОЧНИКИ
# ============================================================

NICKNAMES = [
    "Бурёнка",
    "Зорька",
    "Милка",
    "Ночка",
    "Ласточка",
    "Ромашка",
    "Пеструшка",
    "Чернушка",
    "Белка",
    "Дочка",
    "Красавка",
    "Умница",
    "Солнышко",
    "Звёздочка",
    "Рыжуха",
    "Пятнушка",
    "Марта",
    "Галка",
    "Снежинка",
    "Веснушка",
]


# (название диагноза, хронический)
DIAGNOSES = [
    ("Здоров", 0),
    ("Мастит", 0),
    ("Ламинит", 1),
    ("Эндометрит", 0),
    ("Кетоз", 0),
    ("Ацидоз рубца", 0),
    ("Тимпания рубца", 0),
    ("Бронхопневмония", 0),
    ("Паратуберкулез", 1),
    ("Лейкоз", 1),
    ("Туберкулез", 1),
    ("Бруцеллез", 1),
    ("Ящур", 0),
    ("Некробактериоз", 0),
    ("Копытная гниль", 0),
    ("Отит", 0),
    ("Конъюнктивит", 0),
    ("Авитаминоз", 0),
    ("Гипокальциемия", 0),
    ("Травматический ретикулит", 0),
    ("Пироплазмоз", 0),
]


TREATMENTS = [
    "Антибиотик",
    "Витамины",
    "Противовоспалительное",
    "Покой",
    "Смена корма",
    "Обработка копыт",
    "Гормональная терапия",
    "Инфузия",
    "Пробиотики",
]


VET_NAMES = [
    "Иванов И.И.",
    "Петрова А.С.",
    "Сидоров П.Н.",
    "Кузнецова М.В.",
    "Смирнов Д.А.",
    "Волкова Е.П.",
    "Морозов С.С.",
    "Новикова О.А.",
    "Фёдоров К.Л.",
    "Соколова Т.В.",
]


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def random_date_between(start_date: date, end_date: date) -> date:
    """Возвращает случайную дату между двумя датами."""
    days = (end_date - start_date).days
    return start_date + timedelta(days=random.randint(0, days))


def calculate_age(birth_date: date, exam_date: date) -> int:
    """Возраст животного в полных годах на дату осмотра."""
    age = exam_date.year - birth_date.year

    if (exam_date.month, exam_date.day) < (
        birth_date.month,
        birth_date.day,
    ):
        age -= 1

    return max(age, 1)


def clamp(value, minimum, maximum):
    """Ограничивает значение диапазоном."""
    return max(minimum, min(value, maximum))


def sigmoid(x):
    """Сигмоидальная функция."""
    return 1 / (1 + np.exp(-x))


# ============================================================
# ВЕТЕРИНАРЫ
# ============================================================

def generate_vets():
    """Создаёт справочник ветеринаров."""

    vets = []

    for vet_id, name in enumerate(VET_NAMES[:N_VETS], start=1):
        vets.append(
            {
                "vet_id": vet_id,
                "full_name": name,
                "phone": (
                    f"+7-900-"
                    f"{random.randint(100, 999)}-"
                    f"{random.randint(10, 99)}-"
                    f"{random.randint(10, 99)}"
                ),
            }
        )

    return pd.DataFrame(vets)


# ============================================================
# ДИАГНОЗЫ
# ============================================================

def generate_diagnoses():
    """Создаёт справочник диагнозов."""

    diagnoses = []

    for diagnosis_id, (name, is_chronic) in enumerate(
        DIAGNOSES,
        start=1,
    ):
        diagnoses.append(
            {
                "diagnosis_id": diagnosis_id,
                "name": name,
                "is_chronic": is_chronic,
            }
        )

    return pd.DataFrame(diagnoses)


# ============================================================
# ЖИВОТНЫЕ
# ============================================================

def generate_animals():
    """
    Создаёт животных.

    В animals.csv находятся только относительно постоянные
    характеристики животного.

    Динамические показатели:
    - возраст;
    - номер лактации;
    - удой;
    - хромота

    будут храниться в examinations.csv, поскольку они
    меняются со временем.
    """

    animals = []

    for animal_id in range(1, N_ANIMALS + 1):

        # Возраст животного на начало периода наблюдения.
        initial_age = random.randint(2, 8)

        # День рождения рассчитываем относительно начала наблюдений.
        birth_date = START_DATE - timedelta(
            days=initial_age * 365 + random.randint(0, 364)
        )

        animals.append(
            {
                "animal_id": animal_id,
                "nickname": random.choice(NICKNAMES),
                "number": f"{animal_id:03d}",
                "breed": BREED,
                "gender": GENDER,
                "birth_date": birth_date.strftime("%Y-%m-%d"),
            }
        )

    return pd.DataFrame(animals)


# ============================================================
# ДИНАМИЧЕСКИЕ ХАРАКТЕРИСТИКИ
# ============================================================

def generate_initial_lactation(age):
    """Начальный номер лактации."""

    maximum_lactation = min(max(age - 1, 1), 5)

    return random.randint(1, maximum_lactation)


def generate_initial_milk_yield():
    """
    Начальный удой в кг за условный период.

    Значение является синтетическим и используется
    исключительно для формирования ML-признаков.
    """

    return random.randint(6000, 10000)


def generate_initial_lameness():
    """Начальный бал хромоты от 0 до 5."""

    return random.choices(
        population=[0, 1, 2, 3, 4, 5],
        weights=[50, 25, 15, 5, 3, 2],
        k=1,
    )[0]


def update_dynamic_features(
    age,
    lactation_number,
    milk_yield,
    lameness_score,
    previous_sick,
):
    """
    Немного изменяет динамические характеристики
    между последовательными осмотрами.

    Таким образом, данные не являются полностью независимыми:
    состояние животного зависит от его предыдущего состояния.
    """

    # Удой постепенно меняется.
    milk_change = random.randint(-500, 500)

    # После болезни удой с некоторой вероятностью снижается.
    if previous_sick:
        milk_change -= random.randint(100, 600)

    milk_yield = clamp(
        milk_yield + milk_change,
        4500,
        11000,
    )

    # Хромота имеет некоторую инерцию.
    lameness_change = random.choices(
        [-2, -1, 0, 1, 2],
        weights=[5, 15, 60, 15, 5],
        k=1,
    )[0]

    # После болезни состояние может некоторое время
    # оставаться хуже.
    if previous_sick and random.random() < 0.35:
        lameness_change += 1

    lameness_score = clamp(
        lameness_score + lameness_change,
        0,
        5,
    )

    # Лактация увеличивается постепенно.
    # Не на каждом осмотре, а с небольшой вероятностью.
    if random.random() < 0.20:
        lactation_number = min(
            lactation_number + 1,
            6,
        )

    return (
        lactation_number,
        milk_yield,
        lameness_score,
    )


# ============================================================
# ВЕРОЯТНОСТЬ БОЛЕЗНИ
# ============================================================

def calculate_sick_probability(
    age,
    lactation_number,
    milk_yield,
    lameness_score,
    previous_sick,
    diseases_last_6_months,
    chronic_disease,
    month,
):
    """
    Рассчитывает вероятность заболевания.

    ВАЖНО:
    эта функция используется только для генерации
    синтетического фактического результата.

    В дальнейшем ML-модель должна самостоятельно
    учиться находить эти зависимости.

    Используются признаки из ТЗ:
    - возраст;
    - количество предыдущих осмотров;
    - частота заболеваний;
    - хронические диагнозы;
    - сезонность;
    - хромота;
    - лактация;
    - удой;
    - предыдущий результат.
    """

    # Базовый уровень риска.
    score = -1.1

    # Возраст.
    if age >= 7:
        score += 0.45
    elif age >= 5:
        score += 0.20

    # Лактация.
    score += (lactation_number - 1) * 0.12

    # Хромота.
    score += lameness_score * 0.22

    # Удой.
    # Слишком низкий относительно обычного диапазона удой
    # немного увеличивает риск.
    if milk_yield < 6000:
        score += 0.30
    elif milk_yield < 7000:
        score += 0.15

    # Болезни за последние 6 месяцев.
    score += diseases_last_6_months * 0.25

    # Хроническое заболевание.
    if chronic_disease:
        score += 0.55

    # Предыдущий осмотр был болезненным.
    if previous_sick:
        score += 0.35

    # Сезонность.
    # Зимой и в переходные периоды риск немного выше.
    if month in [1, 2, 3, 11, 12]:
        score += 0.15

    probability = sigmoid(score)

    # Не позволяем получить абсолютно 0 или 1.
    probability = clamp(
        probability,
        0.05,
        0.90,
    )

    return probability


# ============================================================
# ДИАГНОЗ
# ============================================================

def choose_diagnosis(is_sick, previous_diagnosis_id):
    """
    Выбирает фактический диагноз.

    Если животное здорово:
        diagnosis_id = 1

    Если больно:
        выбирается один из 20 диагнозов.

    При наличии хронического заболевания вероятность
    повторного хронического диагноза увеличивается.
    """

    if not is_sick:
        return 1

    # Список хронических диагнозов.
    chronic_ids = [
        diagnosis_id
        for diagnosis_id, (_, is_chronic) in enumerate(
            DIAGNOSES,
            start=1,
        )
        if is_chronic
    ]

    # Если предыдущий диагноз был хроническим,
    # повышаем вероятность повторения.
    if previous_diagnosis_id in chronic_ids:
        if random.random() < 0.45:
            return previous_diagnosis_id

    return random.randint(2, len(DIAGNOSES))


# ============================================================
# ГЕНЕРАЦИЯ ПОСЛЕДОВАТЕЛЬНЫХ ОСМОТРОВ
# ============================================================

def generate_examinations(animals_df):
    """
    Создаёт 500 осмотров.

    Основное отличие от предыдущей версии:

    1. Для каждого животного создаётся последовательность
       осмотров.
    2. Осмотры идут строго по времени.
    3. Состояние животного постепенно меняется.
    4. Для каждого осмотра существует фактический результат
       is_sick.
    5. История предыдущих заболеваний используется только
       для генерации текущего результата.

    В дальнейшем модель сможет использовать предыдущие
    строки для прогнозирования следующей.
    """

    animals = animals_df.copy()

    # --------------------------------------------------------
    # Создаём количество осмотров для каждого животного.
    # --------------------------------------------------------

    # Каждое животное получает минимум 3 осмотра.
    # Остальные осмотры распределяем случайно.
    examinations_per_animal = {
        animal_id: 3
        for animal_id in animals["animal_id"]
    }

    remaining = N_EXAMINATIONS - sum(
        examinations_per_animal.values()
    )

    animal_ids = list(animals["animal_id"])

    for _ in range(remaining):
        animal_id = random.choice(animal_ids)
        examinations_per_animal[animal_id] += 1

    # --------------------------------------------------------
    # Генерация.
    # --------------------------------------------------------

    examinations = []
    examination_id = 1

    for _, animal in animals.iterrows():

        animal_id = int(animal["animal_id"])

        birth_date = date.fromisoformat(
            animal["birth_date"]
        )

        number_of_examinations = examinations_per_animal[
            animal_id
        ]

        # ----------------------------------------------------
        # Создаём даты именно для этого животного.
        # ----------------------------------------------------

        total_days = (END_DATE - START_DATE).days

        # Получаем случайные точки времени и сортируем.
        offsets = sorted(
            random.sample(
                range(total_days + 1),
                number_of_examinations,
            )
        )

        exam_dates = [
            START_DATE + timedelta(days=offset)
            for offset in offsets
        ]

        # ----------------------------------------------------
        # Начальное состояние животного.
        # ----------------------------------------------------

        first_age = calculate_age(
            birth_date,
            exam_dates[0],
        )

        lactation_number = generate_initial_lactation(
            first_age
        )

        milk_yield = generate_initial_milk_yield()

        lameness_score = generate_initial_lameness()

        previous_sick = False
        previous_diagnosis_id = 1

        # История болезни для формирования признаков.
        previous_examinations = []

        # ----------------------------------------------------
        # Последовательная генерация осмотров.
        # ----------------------------------------------------

        for exam_index, exam_date in enumerate(exam_dates):

            age = calculate_age(
                birth_date,
                exam_date,
            )

            # Начиная со второго осмотра обновляем состояние.
            if exam_index > 0:
                (
                    lactation_number,
                    milk_yield,
                    lameness_score,
                ) = update_dynamic_features(
                    age=age,
                    lactation_number=lactation_number,
                    milk_yield=milk_yield,
                    lameness_score=lameness_score,
                    previous_sick=previous_sick,
                )

            # ------------------------------------------------
            # История за последние 6 месяцев.
            # ------------------------------------------------

            six_months_ago = exam_date - timedelta(days=180)

            recent_examinations = [
                item
                for item in previous_examinations
                if item["exam_date"] >= six_months_ago
            ]

            diseases_last_6_months = sum(
                item["is_sick"]
                for item in recent_examinations
            )

            # ------------------------------------------------
            # Есть ли хроническое заболевание в истории.
            # ------------------------------------------------

            chronic_disease = any(
                item["is_chronic"]
                for item in recent_examinations
            )

            # ------------------------------------------------
            # Вероятность болезни.
            # ------------------------------------------------

            sick_probability = calculate_sick_probability(
                age=age,
                lactation_number=lactation_number,
                milk_yield=milk_yield,
                lameness_score=lameness_score,
                previous_sick=previous_sick,
                diseases_last_6_months=diseases_last_6_months,
                chronic_disease=chronic_disease,
                month=exam_date.month,
            )

            # ------------------------------------------------
            # Фактический результат.
            # ------------------------------------------------

            is_sick = int(
                random.random() < sick_probability
            )

            # ------------------------------------------------
            # Фактический диагноз.
            # ------------------------------------------------

            diagnosis_id = choose_diagnosis(
                is_sick=is_sick,
                previous_diagnosis_id=previous_diagnosis_id,
            )

            if is_sick:
                treatment = random.choice(TREATMENTS)
            else:
                treatment = "—"

            # ------------------------------------------------
            # Ветеринар.
            # ------------------------------------------------

            vet_id = random.randint(1, N_VETS)

            # ------------------------------------------------
            # Сохраняем осмотр.
            # ------------------------------------------------

            examination = {
                "examination_id": examination_id,
                "animal_id": animal_id,
                "vet_id": vet_id,
                "exam_date": exam_date.strftime("%Y-%m-%d"),

                # Динамические признаки.
                "age": age,
                "lactation_number": lactation_number,
                "milk_yield": milk_yield,
                "lameness_score": lameness_score,

                # Исторические признаки.
                "previous_examinations_count": len(
                    previous_examinations
                ),
                "diseases_last_6_months": diseases_last_6_months,
                "has_chronic_disease": int(chronic_disease),

                # Фактический результат.
                "diagnosis_id": diagnosis_id,
                "treatment": treatment,
                "is_sick": is_sick,
            }

            examinations.append(examination)

            # ------------------------------------------------
            # Добавляем результат в историю.
            # ------------------------------------------------

            is_chronic = DIAGNOSES[
                diagnosis_id - 1
            ][1]

            previous_examinations.append(
                {
                    "exam_date": exam_date,
                    "is_sick": is_sick,
                    "diagnosis_id": diagnosis_id,
                    "is_chronic": is_chronic,
                }
            )

            previous_sick = bool(is_sick)
            previous_diagnosis_id = diagnosis_id

            examination_id += 1

    examinations_df = pd.DataFrame(examinations)

    # --------------------------------------------------------
    # Финальная сортировка.
    # --------------------------------------------------------

    examinations_df["exam_date"] = pd.to_datetime(
        examinations_df["exam_date"]
    )

    examinations_df = examinations_df.sort_values(
        by=["exam_date", "animal_id"]
    ).reset_index(drop=True)

    examinations_df["exam_date"] = (
        examinations_df["exam_date"]
        .dt.strftime("%Y-%m-%d")
    )

    return examinations_df


# ============================================================
# ПРОВЕРКА ВРЕМЕННОЙ СТРУКТУРЫ
# ============================================================

def validate_data(animals, examinations):
    """
    Проверяет основные требования к датасету.

    Функция нужна для того, чтобы случайно не получить
    некорректные данные перед обучением модели.
    """

    errors = []

    # Количество животных.
    if len(animals) != N_ANIMALS:
        errors.append(
            f"Ожидалось {N_ANIMALS} животных, "
            f"получено {len(animals)}."
        )

    # Количество осмотров.
    if len(examinations) != N_EXAMINATIONS:
        errors.append(
            f"Ожидалось {N_EXAMINATIONS} осмотров, "
            f"получено {len(examinations)}."
        )

    # Проверяем даты.
    dates = pd.to_datetime(
        examinations["exam_date"]
    )

    if dates.min().date() < START_DATE:
        errors.append(
            "Найдена дата раньше начала периода."
        )

    if dates.max().date() > END_DATE:
        errors.append(
            "Найдена дата позже конца периода."
        )

    # Проверяем временную последовательность
    # внутри каждого животного.
    for animal_id, group in examinations.groupby(
        "animal_id"
    ):
        group_dates = pd.to_datetime(
            group["exam_date"]
        )

        if not group_dates.is_monotonic_increasing:
            errors.append(
                f"Нарушена последовательность дат "
                f"животного {animal_id}."
            )

    # Проверяем диапазоны.
    if not examinations["is_sick"].isin([0, 1]).all():
        errors.append(
            "is_sick должен содержать только 0 и 1."
        )

    if not examinations["lameness_score"].between(
        0, 5
    ).all():
        errors.append(
            "lameness_score должен быть от 0 до 5."
        )

    if not examinations["age"].between(
        1, 20
    ).all():
        errors.append(
            "Обнаружен некорректный возраст."
        )

    if errors:
        print("\nОШИБКИ ПРОВЕРКИ:")

        for error in errors:
            print(f"  - {error}")

        raise ValueError(
            "Сгенерированные данные не прошли проверку."
        )

    print("\nПроверка данных пройдена успешно.")


# ============================================================
# СОХРАНЕНИЕ
# ============================================================

def save_data(
    animals,
    vets,
    diagnoses,
    examinations,
):
    """Сохраняет все таблицы в CSV."""

    os.makedirs(
        DATA_DIR,
        exist_ok=True,
    )

    encoding = "utf-8-sig"

    animals.to_csv(
        f"{DATA_DIR}/animals.csv",
        index=False,
        encoding=encoding,
    )

    vets.to_csv(
        f"{DATA_DIR}/vets.csv",
        index=False,
        encoding=encoding,
    )

    diagnoses.to_csv(
        f"{DATA_DIR}/diagnoses.csv",
        index=False,
        encoding=encoding,
    )

    examinations.to_csv(
        f"{DATA_DIR}/examinations.csv",
        index=False,
        encoding=encoding,
    )


# ============================================================
# СТАТИСТИКА
# ============================================================

def print_statistics(
    animals,
    vets,
    diagnoses,
    examinations,
):
    """Выводит статистику сгенерированных данных."""

    sick_count = int(
        examinations["is_sick"].sum()
    )

    healthy_count = (
        len(examinations) - sick_count
    )

    print("\n" + "=" * 60)
    print("СИНТЕТИЧЕСКИЕ ДАННЫЕ СГЕНЕРИРОВАНЫ")
    print("=" * 60)

    print(
        f"Животные:       {len(animals)}"
    )

    print(
        f"Осмотры:        {len(examinations)}"
    )

    print(
        f"Ветеринары:     {len(vets)}"
    )

    print(
        f"Диагнозы:       {len(diagnoses)}"
    )

    print(
        f"Период:         "
        f"{examinations['exam_date'].min()} — "
        f"{examinations['exam_date'].max()}"
    )

    print()

    print(
        f"Больные:        "
        f"{sick_count} "
        f"({sick_count / len(examinations) * 100:.1f}%)"
    )

    print(
        f"Здоровые:       "
        f"{healthy_count} "
        f"({healthy_count / len(examinations) * 100:.1f}%)"
    )

    print()

    print(
        "Количество осмотров на животное:"
    )

    examinations_per_animal = (
        examinations
        .groupby("animal_id")
        .size()
    )

    print(
        f"  минимум:      "
        f"{examinations_per_animal.min()}"
    )

    print(
        f"  максимум:     "
        f"{examinations_per_animal.max()}"
    )

    print(
        f"  среднее:      "
        f"{examinations_per_animal.mean():.2f}"
    )

    print()

    print(
        "Распределение по годам:"
    )

    years = pd.to_datetime(
        examinations["exam_date"]
    ).dt.year.value_counts().sort_index()

    for year, count in years.items():
        print(
            f"  {year}:         {count} осмотров"
        )

    print()

    print(
        "Пример временной истории:"
    )

    example_animal = (
        examinations["animal_id"]
        .value_counts()
        .idxmax()
    )

    example = examinations[
        examinations["animal_id"] == example_animal
    ].sort_values("exam_date")

    print(
        example[
            [
                "animal_id",
                "exam_date",
                "age",
                "lactation_number",
                "milk_yield",
                "lameness_score",
                "diseases_last_6_months",
                "is_sick",
                "diagnosis_id",
            ]
        ].to_string(index=False)
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ВЕТЕРИНАРНЫХ ДАННЫХ")
    print("=" * 60)

    print(
        f"Период: {START_DATE} — {END_DATE}"
    )

    print(
        f"Животных: {N_ANIMALS}"
    )

    print(
        f"Осмотров: {N_EXAMINATIONS}"
    )

    print()

    # --------------------------------------------------------
    # Справочники и животные.
    # --------------------------------------------------------

    vets = generate_vets()

    diagnoses = generate_diagnoses()

    animals = generate_animals()

    # --------------------------------------------------------
    # Последовательные осмотры.
    # --------------------------------------------------------

    examinations = generate_examinations(
        animals
    )

    # --------------------------------------------------------
    # Проверка.
    # --------------------------------------------------------

    validate_data(
        animals=animals,
        examinations=examinations,
    )

    # --------------------------------------------------------
    # Сохранение.
    # --------------------------------------------------------

    save_data(
        animals=animals,
        vets=vets,
        diagnoses=diagnoses,
        examinations=examinations,
    )

    # --------------------------------------------------------
    # Статистика.
    # --------------------------------------------------------

    print_statistics(
        animals=animals,
        vets=vets,
        diagnoses=diagnoses,
        examinations=examinations,
    )

    print()
    print(
        f"CSV-файлы сохранены в папку: {DATA_DIR}/"
    )
    print()


if __name__ == "__main__":
    main()

