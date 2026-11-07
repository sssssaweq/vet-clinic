"""
Прогноз риска заболевания на 2026 год: общий и по конкретным болезням.

Что делает скрипт:
1. Берёт лучшие C и пороги, найденные в train_baseline.py
   (models/metrics_baseline.json и models/metrics_diseases.json).
2. Переобучает все модели на ВСЕХ осмотрах (2023-2025): качество уже проверено
   на тесте, поэтому для прогноза можно использовать все накопленные данные.
3. Для каждого животного считает признаки на дату 01.01.2026 по его
   последнему известному осмотру.
4. Считает вероятность заболевания: общую и по каждой болезни.
   Если вероятность не ниже порога - риск «Высокий».
5. Сохраняет data/predictions_2026.csv (его будет читать Streamlit),
   модели и графики распределения рисков.

Перед запуском нужно один раз выполнить:
    python src/train_baseline.py

Запуск из корня репозитория:
    python src/predict_2026.py
"""

import json
from datetime import date, timedelta

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Общие настройки и функции берём из train_baseline.py.
from train_baseline import (
    DATA_DIR,
    FEATURES,
    MODELS_DIR,
    RISK_THRESHOLD,
    TARGET,
    add_features,
    build_model,
    grid_axes,
    load_data,
)

# ============================================================
# НАСТРОЙКИ
# ============================================================

FORECAST_YEAR = 2026
FORECAST_DATE = date(FORECAST_YEAR, 1, 1)

# Зимние и переходные месяцы (как в генераторе данных).
WINTER_MONTHS = [1, 2, 3, 11, 12]

HIGH = "Высокий"
NOT_HIGH = "Не высокий"


# ============================================================
# ФУНКЦИИ
# ============================================================

def load_settings() -> dict:
    """Читает лучшие C и пороги, найденные при обучении."""
    overall_path = MODELS_DIR / "metrics_baseline.json"
    diseases_path = MODELS_DIR / "metrics_diseases.json"

    if not overall_path.exists() or not diseases_path.exists():
        raise SystemExit(
            "Не найдены результаты обучения в папке models/. "
            "Сначала запусти: python src/train_baseline.py"
        )

    with open(overall_path, encoding="utf-8") as f:
        overall = json.load(f)
    with open(diseases_path, encoding="utf-8") as f:
        diseases = json.load(f)["diseases"]

    return {
        "overall_c": overall["best_C"],
        "diseases": {
            code: {"c": d["best_C"], "threshold": d["threshold"], "name": d["name"]}
            for code, d in diseases.items()
        },
    }


def calculate_age(birth_date: date, on_date: date) -> int:
    """Возраст в полных годах на заданную дату."""
    age = on_date.year - birth_date.year
    if (on_date.month, on_date.day) < (birth_date.month, birth_date.day):
        age -= 1
    return max(age, 1)


def build_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует признаки каждого животного на дату FORECAST_DATE.

    Будущих осмотров ещё нет, поэтому:
    - динамические показатели (лактация, удой, хромота) берутся
      из последнего известного осмотра;
    - возраст пересчитывается на дату прогноза;
    - историческая часть (болезни за 6 месяцев, хроника, интервал
      с прошлого осмотра, прошлый результат) считается относительно
      даты прогноза.
    """
    animals = pd.read_csv(DATA_DIR / "animals.csv", encoding="utf-8-sig")
    diagnoses = pd.read_csv(DATA_DIR / "diagnoses.csv", encoding="utf-8-sig")
    chronic_ids = set(diagnoses.loc[diagnoses["is_chronic"] == 1, "diagnosis_id"])

    forecast_ts = pd.Timestamp(FORECAST_DATE)
    window_start = forecast_ts - timedelta(days=180)

    rows = []
    for animal in animals.itertuples():
        history = df[df["animal_id"] == animal.animal_id].sort_values("exam_date")
        last = history.iloc[-1]
        recent = history[history["exam_date"] >= window_start]

        rows.append(
            {
                "animal_id": animal.animal_id,
                "nickname": animal.nickname,
                "number": f"{animal.number:03d}" if str(animal.number).isdigit() else animal.number,
                "forecast_date": FORECAST_DATE.isoformat(),
                "last_exam_date": last["exam_date"].date().isoformat(),
                "age": calculate_age(date.fromisoformat(animal.birth_date), FORECAST_DATE),
                "lactation_number": int(last["lactation_number"]),
                "milk_yield": int(last["milk_yield"]),
                "lameness_score": int(last["lameness_score"]),
                "previous_examinations_count": len(history),
                "diseases_last_6_months": int(recent[TARGET].sum()),
                "has_chronic_disease": int(recent["diagnosis_id"].isin(chronic_ids).any()),
                "previous_sick": int(last[TARGET]),
                "days_since_last_exam": (forecast_ts - last["exam_date"]).days,
                "is_winter_season": int(FORECAST_DATE.month in WINTER_MONTHS),
            }
        )

    return pd.DataFrame(rows)


def save_risk_distribution_plot(forecast: pd.DataFrame, settings: dict) -> None:
    """Гистограммы: сколько животных в каком диапазоне вероятности."""
    panels = [("Любая болезнь", "risk_probability", RISK_THRESHOLD)]
    panels += [
        (d["name"], f"risk_{code}", d["threshold"])
        for code, d in settings["diseases"].items()
    ]

    fig, axes = grid_axes(len(panels), cols=4, cell_width=4.2, cell_height=3.4)
    for ax, (title, column, threshold) in zip(axes, panels):
        ax.hist(forecast[column], bins=15, color="#2e86ab", edgecolor="white")
        ax.axvline(threshold, color="#d1495b", linestyle="--", label=f"порог ({threshold})")
        ax.set_title(title)
        ax.set_xlabel("Вероятность", fontsize=8)
        ax.set_ylabel("Число животных", fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle(f"Прогноз риска на {FORECAST_YEAR} год")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "risk_distribution_2026.png", dpi=130)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 74)
    print(f"ПРОГНОЗ РИСКА ЗАБОЛЕВАНИЯ НА {FORECAST_YEAR} ГОД")
    print("=" * 74)

    df = add_features(load_data())
    settings = load_settings()

    print(
        f"Модели обучаются на {len(df)} осмотрах "
        f"({df['exam_date'].min().date()} - {df['exam_date'].max().date()})"
    )

    # Признаки на 01.01.2026.
    forecast = build_forecast_features(df)

    # --------------------------------------------------------
    # Общая модель: будет ли выявлена какая-либо болезнь.
    # --------------------------------------------------------
    overall = build_model(settings["overall_c"])
    overall.fit(df[FEATURES], df[TARGET])

    forecast["risk_probability"] = overall.predict_proba(forecast[FEATURES])[:, 1].round(4)
    forecast["risk_level"] = forecast["risk_probability"].apply(
        lambda p: HIGH if p >= RISK_THRESHOLD else NOT_HIGH
    )

    joblib.dump(overall, MODELS_DIR / "final_logreg.joblib")

    # --------------------------------------------------------
    # Модели по конкретным болезням.
    # --------------------------------------------------------
    names = {}
    for code, d in settings["diseases"].items():
        model = build_model(d["c"])
        model.fit(df[FEATURES], df[f"dis_{code}"])

        forecast[f"risk_{code}"] = model.predict_proba(forecast[FEATURES])[:, 1].round(4)
        forecast[f"level_{code}"] = forecast[f"risk_{code}"].apply(
            lambda p, t=d["threshold"]: HIGH if p >= t else NOT_HIGH
        )
        names[code] = d["name"]

        joblib.dump(model, MODELS_DIR / f"final_disease_{code}.joblib")

    # Самая «подозрительная» болезнь: та, чей риск сильнее всего выше обычной
    # частоты этой болезни в данных (так частые болезни не вытесняют редкие).
    lift = pd.DataFrame(
        {
            code: forecast[f"risk_{code}"] / df[f"dis_{code}"].mean()
            for code in settings["diseases"]
        }
    )
    forecast["top_disease"] = lift.idxmax(axis=1).map(names)
    # Список болезней с высоким риском (риск не ниже порога болезни).
    forecast["high_risk_diseases"] = forecast.apply(
        lambda row: "; ".join(
            names[code] for code in names if row[f"level_{code}"] == HIGH
        ) or "нет",
        axis=1,
    )

    forecast = forecast.sort_values("risk_probability", ascending=False).reset_index(drop=True)

    # --------------------------------------------------------
    # Сводка.
    # --------------------------------------------------------
    high_total = int((forecast["risk_level"] == HIGH).sum())
    print(
        f"\nЖивотных: {len(forecast)}\n"
        f"Любая болезнь: высокий риск (>= {RISK_THRESHOLD}) у {high_total}; "
        f"вероятность мин {forecast['risk_probability'].min():.3f}, "
        f"медиана {forecast['risk_probability'].median():.3f}, "
        f"макс {forecast['risk_probability'].max():.3f}"
    )
    print("\nВысокий риск по болезням:")
    for code, d in settings["diseases"].items():
        count = int((forecast[f"level_{code}"] == HIGH).sum())
        print(f"  {d['name']:<26} порог {d['threshold']:.2f}: {count} животных")

    print("\nТоп-5 по общему риску:")
    print(
        forecast[["animal_id", "nickname", "number", "risk_probability", "risk_level", "top_disease"]]
        .head(5)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Сохранение.
    # --------------------------------------------------------
    forecast.to_csv(DATA_DIR / "predictions_2026.csv", index=False, encoding="utf-8-sig")
    save_risk_distribution_plot(forecast, settings)

    print(
        "\nСохранено: data/predictions_2026.csv, модели final_*.joblib, "
        "models/risk_distribution_2026.png"
    )


if __name__ == "__main__":
    main()
