"""
Модель прогнозирования риска заболевания (логистическая регрессия).

Что делает скрипт:
1. Загружает осмотры и справочник диагнозов (data/examinations.csv, data/diagnoses.csv).
2. Формирует признаки (только те, что известны ДО результата осмотра).
3. Делит данные ПО ВРЕМЕНИ в пропорции 70/20/10:
   train - обучение, validation - подбор параметров, test - финальная проверка.
4. ЧАСТЬ 1. Общая модель: «будет ли выявлена какая-либо болезнь» (да/нет).
5. ЧАСТЬ 2. Отдельная модель для каждой достаточно частой болезни
   (мастит, ламинит, кетоз и т.д.): риск именно этой болезни на осмотре.
   Болезни, которых в данных меньше MIN_CASES случаев, не моделируются.
6. Считает метрики: accuracy, precision, recall, F1-score, ROC-AUC.
7. Показывает, как каждый признак влияет на риск каждой болезни,
   и проверяет, что модель нашла связи, заложенные в генераторе данных.
8. Сохраняет модели, метрики и графики в папку models/.

Запуск из корня репозитория:
    python src/train_baseline.py
"""

import json
import textwrap
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# НАСТРОЙКИ
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

SEED = 42

# Пропорции разделения по ТЗ: 70 / 20 / 10.
TRAIN_SHARE = 0.70
VAL_SHARE = 0.20

# Порог «высокого риска» из ТЗ для общей модели и стандартный порог 0.5.
RISK_THRESHOLD = 0.6
DEFAULT_THRESHOLD = 0.5

# Кандидаты для подбора силы регуляризации.
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

TARGET = "is_sick"

FEATURES = [
    "age",
    "lactation_number",
    "milk_yield",
    "lameness_score",
    "previous_examinations_count",
    "diseases_last_6_months",
    "has_chronic_disease",
    "previous_sick",         # результат предыдущего осмотра
    "days_since_last_exam",  # интервал между осмотрами
    "is_winter_season",      # сезонность
]

# Понятные названия признаков для графиков.
FEATURE_LABELS = {
    "age": "Возраст",
    "lactation_number": "Номер лактации",
    "milk_yield": "Удой",
    "lameness_score": "Балл хромоты",
    "previous_examinations_count": "Число прошлых осмотров",
    "diseases_last_6_months": "Болезни за 6 мес.",
    "has_chronic_disease": "Хроническая болезнь",
    "previous_sick": "Болело на прошлом осмотре",
    "days_since_last_exam": "Дней с прошлого осмотра",
    "is_winter_season": "Зимний сезон",
}

# Болезни (названия берутся из diagnoses.csv) и короткие коды для файлов.
DISEASE_CODES = {
    "Ламинит": "laminitis",
    "Некробактериоз": "necrobacillosis",
    "Копытная гниль": "hoof_rot",
    "Мастит": "mastitis",
    "Эндометрит": "endometritis",
    "Кетоз": "ketosis",
    "Ацидоз рубца": "acidosis",
    "Тимпания рубца": "bloat",
    "Гипокальциемия": "hypocalcemia",
    "Авитаминоз": "avitaminosis",
    "Травматический ретикулит": "reticulitis",
    "Бронхопневмония": "bronchopneumonia",
    "Отит": "otitis",
    "Конъюнктивит": "conjunctivitis",
    "Паратуберкулез": "paratuberculosis",
    "Лейкоз": "leukosis",
    "Туберкулез": "tuberculosis",
    "Бруцеллез": "brucellosis",
    "Ящур": "foot_and_mouth",
    "Пироплазмоз": "piroplasmosis",
}

# Модель строится только для болезней, у которых во всех данных не меньше стольких случаев.
# У более редких болезней слишком мало примеров, чтобы честно проверить качество.
MIN_CASES = 150

# Связи «признак -> болезнь», заложенные в генераторе данных (src/generate.py).
# Нужны, чтобы проверить: нашла ли модель то, что мы заложили.
# +1: признак повышает риск болезни, -1: понижает.
EXPECTED_EFFECTS = {
    ("laminitis", "lameness_score"): +1,
    ("necrobacillosis", "lameness_score"): +1,
    ("hoof_rot", "lameness_score"): +1,
    ("hoof_rot", "is_winter_season"): +1,
    ("mastitis", "milk_yield"): +1,
    ("mastitis", "lactation_number"): +1,
    ("endometritis", "lactation_number"): +1,
    ("ketosis", "milk_yield"): -1,
    ("acidosis", "milk_yield"): -1,
    ("hypocalcemia", "lactation_number"): +1,
    ("avitaminosis", "is_winter_season"): +1,
    ("bronchopneumonia", "is_winter_season"): +1,
    ("otitis", "is_winter_season"): +1,
    ("paratuberculosis", "age"): +1,
    ("leukosis", "age"): +1,
}

DISEASE_TARGETS = [f"dis_{code}" for code in DISEASE_CODES.values()]

# Эти столбцы НЕЛЬЗЯ использовать как признаки:
# диагноз, лечение и название болезни известны только после осмотра (утечка цели).
LEAKAGE_COLUMNS = ["diagnosis_id", "treatment", "disease", TARGET] + DISEASE_TARGETS


# ============================================================
# ПОДГОТОВКА ДАННЫХ
# ============================================================

def load_data() -> pd.DataFrame:
    """Загружает осмотры, добавляет название болезни и сортирует по времени."""
    df = pd.read_csv(
        DATA_DIR / "examinations.csv", encoding="utf-8-sig", parse_dates=["exam_date"]
    )
    diagnoses = pd.read_csv(DATA_DIR / "diagnoses.csv", encoding="utf-8-sig")
    df = df.merge(
        diagnoses[["diagnosis_id", "name"]].rename(columns={"name": "disease"}),
        on="diagnosis_id",
        how="left",
    )
    return df.sort_values(["exam_date", "animal_id"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет производные признаки и целевые столбцы по болезням.

    Признаки используют только прошлые осмотры того же животного,
    поэтому утечки будущего нет.
    """
    df = df.copy()
    by_animal = df.groupby("animal_id")

    # Результат предыдущего осмотра (для первого осмотра - 0).
    df["previous_sick"] = by_animal[TARGET].shift(1).fillna(0).astype(int)

    # Интервал в днях с прошлого осмотра. Для первого осмотра
    # пропуск заполняем медианой - признака «прошлого» у него нет.
    gap = by_animal["exam_date"].diff().dt.days
    df["days_since_last_exam"] = gap.fillna(gap.median())

    # Сезонность: зимние и переходные месяцы (как в генераторе данных).
    df["is_winter_season"] = df["exam_date"].dt.month.isin([1, 2, 3, 11, 12]).astype(int)

    # Целевой столбец каждой болезни: 1, если на осмотре выявлена именно ЭТА болезнь.
    for disease, code in DISEASE_CODES.items():
        df[f"dis_{code}"] = ((df[TARGET] == 1) & (df["disease"] == disease)).astype(int)

    return df


def split_by_time(df: pd.DataFrame):
    """
    Делит данные по времени: более ранние осмотры - в train,
    более поздние - в validation и test. Так модель не «подглядывает»
    в будущее, как это было бы при случайном разделении.
    """
    n = len(df)
    # round() вместо int(): из-за float 0.7 + 0.2 = 0.8999..., и int() давал неточное разделение.
    train_end = round(n * TRAIN_SHARE)
    val_end = round(n * (TRAIN_SHARE + VAL_SHARE))

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# ============================================================
# МОДЕЛЬ И МЕТРИКИ
# ============================================================

def build_model(c: float):
    """Масштабирование признаков + логистическая регрессия."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=1000, random_state=SEED),
    )


# Если несколько значений C дают почти одинаковое качество (разница меньше этого числа),
# берём самое маленькое C: чем меньше C, тем «осторожнее» и проще модель.
C_TOLERANCE = 0.002


def find_best_c(train, val, target, verbose=True):
    """
    Перебирает C, учит модель на train и сравнивает по ROC-AUC на validation.
    Из практически равных вариантов выбирает самое простое (наименьшее C).
    """
    scores = {}

    for c in C_GRID:
        model = build_model(c)
        model.fit(train[FEATURES], train[target])
        scores[c] = roc_auc_score(val[target], model.predict_proba(val[FEATURES])[:, 1])
        if verbose:
            print(f"  C={c:<5} val ROC-AUC={scores[c]:.4f}")

    top = max(scores.values())
    best_c = min(c for c, auc in scores.items() if auc >= top - C_TOLERANCE)

    return best_c, scores[best_c]


def choose_threshold(y_true, proba) -> float:
    """Порог, при котором F1-score на валидации максимален."""
    best_threshold, best_f1 = 0.5, -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        score = f1_score(y_true, (proba >= threshold).astype(int), zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), score

    return round(best_threshold, 2)


def compute_metrics(y_true, proba, threshold: float) -> dict:
    """Считает метрики качества при заданном пороге."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics(title: str, metrics: dict) -> None:
    print(
        f"{title:<28} "
        f"acc={metrics['accuracy']:.3f}  "
        f"prec={metrics['precision']:.3f}  "
        f"rec={metrics['recall']:.3f}  "
        f"f1={metrics['f1']:.3f}  "
        f"auc={metrics['roc_auc']:.3f}"
    )


# ============================================================
# ЧАСТЬ 1. ОБЩАЯ МОДЕЛЬ: БОЛЕН ИЛИ НЕТ
# ============================================================

def run_overall(train, val, test):
    print("\n" + "=" * 74)
    print("ЧАСТЬ 1. ОБЩАЯ МОДЕЛЬ: БУДЕТ ЛИ ВЫЯВЛЕНА КАКАЯ-ЛИБО БОЛЕЗНЬ")
    print("=" * 74)

    print("Подбор C на валидационной выборке:")
    best_c, best_auc = find_best_c(train, val, TARGET)
    print(f"\nЛучшее C = {best_c} (val ROC-AUC = {best_auc:.4f})")

    # Итоговая модель учится на train. Тест используется один раз - для финальной оценки.
    model = build_model(best_c)
    model.fit(train[FEATURES], train[TARGET])

    results = {
        "model": "LogisticRegression",
        "target": TARGET,
        "best_C": best_c,
        "features": FEATURES,
        "split": {"train": len(train), "validation": len(val), "test": len(test)},
        "metrics": {},
    }

    print("\nМетрики:")
    for split_name, part in [("train", train), ("validation", val), ("test", test)]:
        proba = model.predict_proba(part[FEATURES])[:, 1]
        for threshold in (DEFAULT_THRESHOLD, RISK_THRESHOLD):
            metrics = compute_metrics(part[TARGET], proba, threshold)
            results["metrics"][f"{split_name}@{threshold}"] = metrics
            print_metrics(f"{split_name} (порог {threshold})", metrics)

    joblib.dump(model, MODELS_DIR / "baseline_logreg.joblib")
    with open(MODELS_DIR / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return model


# ============================================================
# ЧАСТЬ 2. МОДЕЛИ ПО КОНКРЕТНЫМ БОЛЕЗНЯМ
# ============================================================

def select_diseases(df) -> dict:
    """Болезни, по которым достаточно случаев для обучения модели: название -> код."""
    return {
        name: code
        for name, code in DISEASE_CODES.items()
        if df[f"dis_{code}"].sum() >= MIN_CASES
    }


def run_diseases(df, train, val, test):
    print("\n" + "=" * 74)
    print("ЧАСТЬ 2. МОДЕЛИ ПО КОНКРЕТНЫМ БОЛЕЗНЯМ")
    print("=" * 74)
    print("Для каждой болезни своя модель: «будет ли на осмотре выявлена именно ЭТА болезнь».")
    print("Порог подбирается на валидации по максимуму F1 (у редких болезней 0.6 слишком высок).")

    modeled = select_diseases(df)
    skipped = [f"{n} ({int(df[f'dis_{c}'].sum())})" for n, c in DISEASE_CODES.items() if n not in modeled]
    print(f"Моделируем {len(modeled)} болезней (не менее {MIN_CASES} случаев).")
    print("Пропущены из-за малого числа случаев: " + ", ".join(skipped) + "\n")

    diseases = {}

    for disease, code in modeled.items():
        target = f"dis_{code}"

        best_c, val_auc = find_best_c(train, val, target, verbose=False)
        model = build_model(best_c)
        model.fit(train[FEATURES], train[target])

        proba = {
            "validation": model.predict_proba(val[FEATURES])[:, 1],
            "test": model.predict_proba(test[FEATURES])[:, 1],
        }
        threshold = choose_threshold(val[target], proba["validation"])

        metrics = {
            "validation": compute_metrics(val[target], proba["validation"], threshold),
            "test": compute_metrics(test[target], proba["test"], threshold),
        }

        diseases[code] = {
            "name": disease,
            "model": model,
            "best_C": best_c,
            "threshold": threshold,
            "cases": {
                "train": int(train[target].sum()),
                "validation": int(val[target].sum()),
                "test": int(test[target].sum()),
            },
            "metrics": metrics,
            "coefficients": dict(zip(FEATURES, (float(c) for c in model[-1].coef_[0]))),
        }

    # Итоговая таблица.
    print(f"{'Болезнь':<22}{'случаев test':>13}{'C':>7}{'порог':>7}{'AUC val':>9}{'AUC test':>10}"
          f"{'prec':>7}{'recall':>8}{'F1':>7}   (precision/recall/F1 - на тесте)")
    for code, d in diseases.items():
        t = d["metrics"]["test"]
        print(
            f"{d['name']:<22}{d['cases']['test']:>13}{d['best_C']:>7}{d['threshold']:>7.2f}"
            f"{d['metrics']['validation']['roc_auc']:>9.3f}{t['roc_auc']:>10.3f}"
            f"{t['precision']:>7.3f}{t['recall']:>8.3f}{t['f1']:>7.3f}"
        )

    return diseases


# ============================================================
# ВЛИЯНИЕ ПРИЗНАКОВ НА РИСК
# ============================================================

def build_effects_table(overall_model, diseases) -> pd.DataFrame:
    """
    Таблица коэффициентов: строки - признаки, столбцы - модели.
    Коэффициенты считаются на стандартизованных признаках,
    поэтому их можно сравнивать между собой.
    """
    table = pd.DataFrame(index=FEATURES)
    table["Любая болезнь"] = overall_model[-1].coef_[0]

    for d in diseases.values():
        table[d["name"]] = pd.Series(d["coefficients"])

    return table


def check_expected_effects(table: pd.DataFrame, diseases) -> int:
    """Проверяет, нашла ли модель связи, заложенные в генераторе."""
    print("\n" + "=" * 74)
    print("ПРОВЕРКА: НАШЛА ЛИ МОДЕЛЬ СВЯЗИ, ЗАЛОЖЕННЫЕ В ГЕНЕРАТОРЕ")
    print("=" * 74)

    found = 0
    checked = 0
    for (code, feature), sign in EXPECTED_EFFECTS.items():
        if code not in diseases:   # болезнь не моделируется (мало случаев)
            continue
        checked += 1
        name = diseases[code]["name"]
        coef = table.loc[feature, name]
        rank = int((table[name].abs() > abs(coef)).sum()) + 1
        ok = coef * sign > 0
        found += int(ok)
        expected = "повышает" if sign > 0 else "понижает"
        print(
            f"[{'OK' if ok else 'НЕТ'}] {name}: «{FEATURE_LABELS[feature]}» {expected} риск; "
            f"коэффициент {coef:+.2f}, по силе {rank}-й из {len(FEATURES)}"
        )

    print(f"\nВосстановлено связей: {found} из {checked}")
    return found


# ============================================================
# ГРАФИКИ
# ============================================================

def save_overall_plots(model, validation, test, table) -> None:
    """ROC-кривая и вклад признаков общей модели."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, part in [("validation", validation), ("test", test)]:
        proba = model.predict_proba(part[FEATURES])[:, 1]
        fpr, tpr, _ = roc_curve(part[TARGET], proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(part[TARGET], proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="случайный выбор")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая: общая модель")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "roc_baseline.png", dpi=150)
    plt.close(fig)

    coefs = table["Любая болезнь"].sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh([FEATURE_LABELS[f] for f in coefs.index], coefs.values,
            color=["#d1495b" if c > 0 else "#2e86ab" for c in coefs.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Коэффициент (стандартизованные признаки)")
    ax.set_title("Вклад признаков в риск любой болезни")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "feature_importance_baseline.png", dpi=150)
    plt.close(fig)


def grid_axes(n_panels: int, cols: int, cell_width: float, cell_height: float):
    """Создаёт сетку графиков под нужное число панелей; лишние клетки скрывает."""
    rows = -(-n_panels // cols)   # деление с округлением вверх
    fig, axes = plt.subplots(rows, cols, figsize=(cell_width * cols, cell_height * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_panels:]:
        ax.axis("off")
    return fig, axes[:n_panels]


def save_roc_grid(overall_model, diseases, validation, test) -> None:
    """ROC-графики: общая модель и по одному графику на каждую болезнь."""
    panels = [("Любая болезнь", overall_model, TARGET)]
    panels += [(d["name"], d["model"], f"dis_{code}") for code, d in diseases.items()]

    fig, axes = grid_axes(len(panels), cols=4, cell_width=4.2, cell_height=4.0)
    for ax, (title, model, target) in zip(axes, panels):
        for label, part in [("validation", validation), ("test", test)]:
            proba = model.predict_proba(part[FEATURES])[:, 1]
            fpr, tpr, _ = roc_curve(part[target], proba)
            ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(part[target], proba):.3f})")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="случайный выбор")
        ax.set_title(title)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=7)

    fig.suptitle("ROC-кривые: общая модель и модели по болезням")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "roc_diseases.png", dpi=130)
    plt.close(fig)


def save_effects_heatmap(table: pd.DataFrame, diseases) -> None:
    """Тепловая карта: влияние каждого признака на риск каждой болезни."""
    name_to_code = {d["name"]: code for code, d in diseases.items()}
    labels = [FEATURE_LABELS[f] for f in table.index]
    columns = [textwrap.fill(c, 12, break_long_words=False) for c in table.columns]
    values = table.values
    limit = np.abs(values).max()

    fig, ax = plt.subplots(figsize=(17, 7))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i, feature in enumerate(table.index):
        for j, column in enumerate(table.columns):
            value = values[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value) > 0.6 * limit else "black")

            # Обводим связи, заложенные в генераторе.
            code = name_to_code.get(column)
            if (code, feature) in EXPECTED_EFFECTS:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", linewidth=2.5))

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Коэффициент: красный повышает риск, синий понижает")
    ax.set_title("Влияние признаков на риск болезней (чёрной рамкой обведены связи, заложенные в данных)")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "feature_effects_heatmap.png", dpi=150)
    plt.close(fig)


def save_effects_grid(table: pd.DataFrame) -> None:
    """Столбчатые графики: вклад признаков в риск (общая модель и каждая болезнь)."""
    limit = np.abs(table.values).max() * 1.1

    fig, axes = grid_axes(len(table.columns), cols=4, cell_width=5.0, cell_height=3.6)
    for ax, column in zip(axes, table.columns):
        coefs = table[column].sort_values()
        ax.barh([FEATURE_LABELS[f] for f in coefs.index], coefs.values,
                color=["#d1495b" if c > 0 else "#2e86ab" for c in coefs.values])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlim(-limit, limit)
        ax.set_title(column)
        ax.set_xlabel("Коэффициент (стандартизованные признаки)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Вклад признаков в риск: красным - повышают, синим - понижают")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "feature_effects_diseases.png", dpi=130)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    df = add_features(load_data())

    # Защита от случайной утечки цели в признаки.
    assert not set(FEATURES) & set(LEAKAGE_COLUMNS), "В признаках есть утечка цели"
    assert df[FEATURES].isna().sum().sum() == 0, "В признаках есть пропуски"
    assert df["disease"].notna().all(), "У диагноза не указано название"
    assert set(df.loc[df[TARGET] == 1, "disease"]) <= set(DISEASE_CODES), (
        "В данных есть болезнь, которой нет в DISEASE_CODES"
    )

    train, val, test = split_by_time(df)

    print("=" * 74)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ РИСКА ЗАБОЛЕВАНИЯ")
    print("=" * 74)
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        print(
            f"{name:<11} {len(part):>4} осмотров  "
            f"{part['exam_date'].min().date()} - {part['exam_date'].max().date()}  "
            f"доля больных: {part[TARGET].mean():.3f}"
        )

    overall_model = run_overall(train, val, test)
    diseases = run_diseases(df, train, val, test)

    table = build_effects_table(overall_model, diseases)
    check_expected_effects(table, diseases)

    # Сохранение результатов по болезням.
    disease_results = {
        "features": FEATURES,
        "min_cases": MIN_CASES,
        "split": {"train": len(train), "validation": len(val), "test": len(test)},
        "diseases": {},
    }
    for code, d in diseases.items():
        joblib.dump(d["model"], MODELS_DIR / f"disease_{code}.joblib")
        disease_results["diseases"][code] = {k: v for k, v in d.items() if k != "model"}

    with open(MODELS_DIR / "metrics_diseases.json", "w", encoding="utf-8") as f:
        json.dump(disease_results, f, ensure_ascii=False, indent=2)

    table.rename(index=FEATURE_LABELS).round(4).to_csv(
        MODELS_DIR / "feature_effects.csv", encoding="utf-8-sig"
    )

    # Графики.
    save_overall_plots(overall_model, val, test, table)
    save_roc_grid(overall_model, diseases, val, test)
    save_effects_heatmap(table, diseases)
    save_effects_grid(table)

    print(f"\nМодели, метрики и графики сохранены в: {MODELS_DIR}/")


if __name__ == "__main__":
    main()
