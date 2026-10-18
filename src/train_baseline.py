"""
Базовая модель прогнозирования риска заболевания (логистическая регрессия).

Что делает скрипт:
1. Загружает осмотры из data/examinations.csv.
2. Формирует признаки (только те, что известны ДО результата осмотра).
3. Делит данные ПО ВРЕМЕНИ в пропорции 70/20/10:
   train — обучение, validation — подбор параметров, test — финальная проверка.
4. Подбирает коэффициент регуляризации C на валидационной выборке.
5. Считает метрики: accuracy, precision, recall, F1-score, ROC-AUC.
6. Сохраняет модель, метрики и графики в папку models/.

Запуск из корня репозитория:
    python src/train_baseline.py
"""

import json
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


# НАСТРОЙКИ


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "examinations.csv"
MODELS_DIR = ROOT / "models"

SEED = 42

# Пропорции разделения по ТЗ: 70 / 20 / 10.
TRAIN_SHARE = 0.70
VAL_SHARE = 0.20

# Порог «высокого риска» из ТЗ (пример: 0.6) и стандартный порог 0.5.
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

# Эти столбцы НЕЛЬЗЯ использовать как признаки:
# diagnosis_id и treatment известны только после осмотра (утечка цели),
# идентификаторы не несут смысла для прогноза.
LEAKAGE_COLUMNS = ["diagnosis_id", "treatment", "is_sick"]



# ПОДГОТОВКА ДАННЫХ


def load_data() -> pd.DataFrame:
    """Загружает осмотры и сортирует их по времени."""
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["exam_date"])
    return df.sort_values(["exam_date", "animal_id"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет производные признаки.

    Все они используют только прошлые осмотры того же животного,
    поэтому утечки будущего нет.
    """
    df = df.copy()
    by_animal = df.groupby("animal_id")

    # Результат предыдущего осмотра (для первого осмотра — 0).
    df["previous_sick"] = by_animal[TARGET].shift(1).fillna(0).astype(int)

    # Интервал в днях с прошлого осмотра. Для первого осмотра
    # пропуск заполняем медианой — признака «прошлого» у него нет.
    gap = by_animal["exam_date"].diff().dt.days
    df["days_since_last_exam"] = gap.fillna(gap.median())

    # Сезонность: зимние и переходные месяцы (как в генераторе данных).
    df["is_winter_season"] = df["exam_date"].dt.month.isin([1, 2, 3, 11, 12]).astype(int)

    return df


def split_by_time(df: pd.DataFrame):
    """
    Делит данные по времени: более ранние осмотры — в train,
    более поздние — в validation и test. Так модель не «подглядывает»
    в будущее, как это было бы при случайном разделении.
    """
    n = len(df)
    # round() вместо int(): из-за float 0.7 + 0.2 = 0.8999..., и int() давал 99/51.
    train_end = round(n * TRAIN_SHARE)
    val_end = round(n * (TRAIN_SHARE + VAL_SHARE))

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# МЕТРИКИ


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



# ГРАФИКИ


def save_plots(model, X_val, y_val, X_test, y_test) -> None:
    """Сохраняет ROC-кривую и важность признаков."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, X, y in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="случайный выбор")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая: логистическая регрессия")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "roc_baseline.png", dpi=150)
    plt.close(fig)

    # Коэффициенты считаются на стандартизованных признаках,
    # поэтому их можно сравнивать между собой.
    coefs = model[-1].coef_[0]
    order = np.argsort(coefs)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(np.array(FEATURES)[order], coefs[order],
            color=["#d1495b" if c > 0 else "#2e86ab" for c in coefs[order]])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Коэффициент (стандартизованные признаки)")
    ax.set_title("Вклад признаков в риск заболевания")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "feature_importance_baseline.png", dpi=150)
    plt.close(fig)



# MAIN


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    df = add_features(load_data())

    # Защита от случайной утечки цели в признаки.
    assert not set(FEATURES) & set(LEAKAGE_COLUMNS), "В признаках есть утечка цели"
    assert df[FEATURES].isna().sum().sum() == 0, "В признаках есть пропуски"

    train, val, test = split_by_time(df)

    print("=" * 70)
    print("БАЗОВАЯ МОДЕЛЬ: ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
    print("=" * 70)
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        print(
            f"{name:<11} {len(part):>4} осмотров  "
            f"{part['exam_date'].min().date()} — {part['exam_date'].max().date()}  "
            f"доля больных: {part[TARGET].mean():.3f}"
        )

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    # --------------------------------------------------------
    # Подбор C по ROC-AUC на валидационной выборке.
    # --------------------------------------------------------
    print("\nПодбор C на валидационной выборке:")
    best_c, best_auc = None, -1.0

    for c in C_GRID:
        candidate = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=1000, random_state=SEED),
        )
        candidate.fit(X_train, y_train)
        auc = roc_auc_score(y_val, candidate.predict_proba(X_val)[:, 1])
        print(f"  C={c:<5} val ROC-AUC={auc:.4f}")

        if auc > best_auc:
            best_c, best_auc = c, auc

    print(f"\nЛучшее C = {best_c} (val ROC-AUC = {best_auc:.4f})")

    # --------------------------------------------------------
    # Итоговая модель обучается на train с лучшим C.
    # Тест используется ровно один раз — для финальной оценки.
    # --------------------------------------------------------
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best_c, max_iter=1000, random_state=SEED),
    )
    model.fit(X_train, y_train)

    results = {
        "model": "LogisticRegression",
        "best_C": best_c,
        "features": FEATURES,
        "split": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
        },
        "metrics": {},
    }

    print("\nМетрики:")
    for split_name, X, y in [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        proba = model.predict_proba(X)[:, 1]
        for threshold in (DEFAULT_THRESHOLD, RISK_THRESHOLD):
            metrics = compute_metrics(y, proba, threshold)
            results["metrics"][f"{split_name}@{threshold}"] = metrics
            print_metrics(f"{split_name} (порог {threshold})", metrics)

    # --------------------------------------------------------
    # Сохранение.
    # --------------------------------------------------------
    joblib.dump(model, MODELS_DIR / "baseline_logreg.joblib")

    with open(MODELS_DIR / "metrics_baseline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    save_plots(model, X_val, y_val, X_test, y_test)

    print(f"\nМодель, метрики и графики сохранены в: {MODELS_DIR}/")


if __name__ == "__main__":
    main()