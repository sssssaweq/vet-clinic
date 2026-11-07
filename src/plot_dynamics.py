"""
Графики динамики: как меняются удой и хромота после конкретной болезни.

Что делает скрипт:
1. Для каждой болезни находит «начало болезни» - осмотр, на котором болезнь выявлена
   впервые (на прошлом осмотре этой болезни не было).
2. Смотрит, как менялись удой и хромота у этих коров на осмотрах ДО и ПОСЛЕ болезни,
   и показывает средние значения.
3. Рисует удой нескольких конкретных коров по времени с отметками болезней.

Запуск из корня репозитория (после generate.py):
    python src/plot_dynamics.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from train_baseline import DISEASE_CODES, MIN_CASES, MODELS_DIR, grid_axes, load_data

# ============================================================
# НАСТРОЙКИ
# ============================================================

# Сколько осмотров смотрим до болезни и после неё.
BEFORE = 3
AFTER = 4

# Болезни, для которых рисуем удой отдельных коров.
EXAMPLE_DISEASES = ["Мастит", "Кетоз", "Ламинит", "Бронхопневмония"]


# ============================================================
# ФУНКЦИИ
# ============================================================

def find_episodes(df: pd.DataFrame, disease: str) -> list[pd.DataFrame]:
    """
    Возвращает «окна» вокруг начала каждой болезни.

    Окно - это таблица из осмотров одной коровы: BEFORE осмотров до болезни,
    сам осмотр с болезнью (offset = 0) и AFTER осмотров после.
    Колонка offset показывает положение осмотра относительно болезни.
    """
    windows = []
    for _, animal_df in df.groupby("animal_id"):
        animal_df = animal_df.sort_values("exam_date").reset_index(drop=True)
        has_disease = animal_df["disease"] == disease

        for i in animal_df.index:
            # Начало болезни: сейчас она есть, на прошлом осмотре её не было.
            started = has_disease[i] and not (i > 0 and has_disease[i - 1])
            # Берём только полные окна, чтобы все точки графика считались по одним коровам.
            if started and i - BEFORE >= 0 and i + AFTER < len(animal_df):
                window = animal_df.loc[i - BEFORE : i + AFTER].copy()
                window["offset"] = range(-BEFORE, AFTER + 1)
                windows.append(window)
    return windows


def mean_change(windows: list[pd.DataFrame], column: str) -> pd.Series:
    """
    Среднее изменение показателя относительно осмотра перед болезнью (offset = -1).
    Так мы сравниваем не абсолютные значения, а «насколько стало больше или меньше».
    """
    changes = []
    for window in windows:
        base = window.loc[window["offset"] == -1, column].iloc[0]
        changes.append(window.set_index("offset")[column] - base)
    return pd.concat(changes, axis=1).mean(axis=1)


def modeled_diseases(df: pd.DataFrame) -> list[str]:
    """Болезни, у которых достаточно случаев (как в train_baseline.py)."""
    counts = df.loc[df["is_sick"] == 1, "disease"].value_counts()
    return [d for d in DISEASE_CODES if counts.get(d, 0) >= MIN_CASES]


def save_after_disease_plots(df: pd.DataFrame, diseases: list[str]) -> None:
    """Два файла: изменение удоя и изменение хромоты вокруг начала каждой болезни."""
    for column, label, file_name in [
        ("milk_yield", "Изменение удоя, кг", "dynamics_milk_by_disease.png"),
        ("lameness_score", "Изменение хромоты, баллы", "dynamics_lameness_by_disease.png"),
    ]:
        fig, axes = grid_axes(len(diseases), cols=4, cell_width=4.2, cell_height=3.2)
        for ax, disease in zip(axes, diseases):
            windows = find_episodes(df, disease)
            change = mean_change(windows, column)
            ax.plot(change.index, change.values, marker="o", color="#2e86ab", linewidth=2)
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.axvline(0, color="#444444", linestyle="--", linewidth=0.8)
            ax.set_title(f"{disease} ({len(windows)} случаев)", fontsize=10)
            ax.set_xlabel("Осмотров от начала болезни", fontsize=8)
            ax.set_ylabel(label, fontsize=8)

        fig.suptitle(
            f"{label} до и после болезни "
            "(0 - осмотр с болезнью; за ноль взят осмотр перед болезнью)"
        )
        fig.tight_layout()
        fig.savefig(MODELS_DIR / file_name, dpi=130)
        plt.close(fig)


def save_cow_examples_plot(df: pd.DataFrame) -> None:
    """Удой отдельных коров по времени. Цветные точки - осмотры с нужной болезнью."""
    fig, axes = plt.subplots(len(EXAMPLE_DISEASES), 1, figsize=(12, 3.2 * len(EXAMPLE_DISEASES)), sharex=True)

    for ax, disease in zip(axes, EXAMPLE_DISEASES):
        # Корова, у которой больше всего осмотров с этой болезнью.
        counts = df[df["disease"] == disease].groupby("animal_id").size()
        animal_id = counts.idxmax()
        cow = df[df["animal_id"] == animal_id].sort_values("exam_date")

        ax.plot(cow["exam_date"], cow["milk_yield"], color="#999999", linewidth=1.2, label="удой")
        other = cow[(cow["is_sick"] == 1) & (cow["disease"] != disease)]
        this = cow[cow["disease"] == disease]
        ax.scatter(other["exam_date"], other["milk_yield"], color="#bbbbbb", s=25, label="другая болезнь")
        ax.scatter(this["exam_date"], this["milk_yield"], color="#d1495b", s=55, zorder=3, label=disease)
        ax.set_ylabel("Удой, кг")
        ax.set_title(f"Корова №{animal_id}: чаще всего болела болезнью «{disease}»", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, ncol=3)

    axes[-1].set_xlabel("Дата осмотра")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "dynamics_cow_examples.png", dpi=130)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    df = load_data()
    diseases = modeled_diseases(df)
    save_after_disease_plots(df, diseases)
    save_cow_examples_plot(df)

    print("Сохранено: models/dynamics_milk_by_disease.png, "
          "models/dynamics_lameness_by_disease.png, models/dynamics_cow_examples.png")
    print("\nСреднее изменение удоя через 1 осмотр после начала болезни (кг):")
    for disease in diseases:
        change = mean_change(find_episodes(df, disease), "milk_yield")
        print(f"  {disease:<20} {change[1]:+.0f}")


if __name__ == "__main__":
    main()
