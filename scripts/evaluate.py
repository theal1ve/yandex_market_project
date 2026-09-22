"""Оценка готовой модели на test.tsv (с ground truth).

Ожидает test.tsv с колонками category_id и department_id.
Запуск:
    python scripts/evaluate.py

Результаты:
    reports/metrics_report.txt
    reports/predictions.csv
    reports/coverage.png
    reports/rule_sources.png
    reports/confusion_department.png
    reports/top_categories.png
    reports/top_errors.png
"""


import argparse
import gzip
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.features import (
    build_category_text,
    build_description_char_text,
    build_description_word_text,
)
from src.inference import predict_bayes_knn
from src.rules import apply_rules, rule_sources

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


start_time = time.time()


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Оценка модели на test.tsv")
    p.add_argument("--model", default="models/model.pkl.gz")
    p.add_argument("--test", default="data/test.tsv")
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--top-n", type=int, default=20,
                   help="сколько классов показывать на графиках")
    p.add_argument("--min-support", type=int, default=5,
                   help="минимум примеров класса для попадания в top-ошибок")
    return p.parse_args()


# ---------- Model ----------

def load_model(path: str) -> dict:
    with gzip.open(path, "rb") as f:
        m = {
            "tf_idf_category":            pickle.load(f),
            "X_norm_train":               pickle.load(f),
            "category_labels_train":      pickle.load(f),
            "department_labels_train":    pickle.load(f),
            "department_classes":         pickle.load(f),
            "department_to_idx":          pickle.load(f),
            "train_department_idx_train": pickle.load(f),
            "fallback":                   pickle.load(f),
            "t_word_department":          pickle.load(f),
            "t_char_department":          pickle.load(f),
            "SVC_department":             pickle.load(f),
            "vender_code_map":            pickle.load(f),
            "SCN_category_map":           pickle.load(f),
            "SCN_deparment_map":          pickle.load(f),
            "title_vender_name_map":      pickle.load(f),
            "title_SCN_map":              pickle.load(f),
            "category_to_department":     pickle.load(f),
            "temperature":                pickle.load(f),
        }
    return m


# ---------- Inference ----------

def run_inference(df: pd.DataFrame, m: dict):
    n = len(df)
    category_preds = np.full(n, -1, dtype=np.int64)

    rule_pred = apply_rules(
        df, m["vender_code_map"], m["SCN_category_map"],
        m["title_vender_name_map"], m["title_SCN_map"],
    )
    rule_mask = rule_pred.notna().values
    category_preds[rule_mask] = rule_pred[rule_mask].values.astype(np.int64)

    rule_src = rule_sources(
        df, m["vender_code_map"], m["SCN_category_map"],
        m["title_vender_name_map"], m["title_SCN_map"],
    )

    fb_mask = ~rule_mask
    n_fb = int(fb_mask.sum())
    if n_fb > 0:
        df_fb = df[fb_mask].reset_index(drop=True)
        X_cat = m["tf_idf_category"].transform(build_category_text(df_fb))
        X_w = m["t_word_department"].transform(
            df_fb.apply(build_description_word_text, axis=1))
        X_c = m["t_char_department"].transform(
            df_fb.apply(build_description_char_text, axis=1))
        X_d = hstack([X_w, X_c])
        cat_fb = predict_bayes_knn(
            X_cat, X_d, m["SVC_department"],
            m["X_norm_train"], m["category_labels_train"],
            m["train_department_idx_train"],
            temperature=m["temperature"],
        )
        category_preds[fb_mask] = cat_fb

    department_preds = np.array(
        [m["category_to_department"].get(int(c), -1) for c in category_preds],
        dtype=np.int64,
    )

    n_dept_fallback = 0
    unk = department_preds == -1
    if unk.sum() > 0:
        n_dept_fallback = int(unk.sum())
        Xw = m["t_word_department"].transform(
            df[unk].apply(build_description_word_text, axis=1))
        Xc = m["t_char_department"].transform(
            df[unk].apply(build_description_char_text, axis=1))
        department_preds[unk] = m["SVC_department"].predict(
            hstack([Xw, Xc])).astype(int)
        category_preds[unk & (category_preds == -1)] = m["fallback"]

    return category_preds, department_preds, rule_src, n_fb, n_dept_fallback


# ---------- Plots ----------

def plot_coverage(rule_src, n_fb, n_dept_fallback, n_total, out_path):
    labels = ["rule-based", "KNN", "dept fallback"]
    counts = [int(rule_src.notna().sum()), n_fb, n_dept_fallback]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, counts, color=["#4C72B0", "#DD8452", "#55A868"])
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{c}  ({c / n_total:.1%})", va="center")
    ax.set_xlim(0, max(counts) * 1.25 if max(counts) > 0 else 1)
    ax.set_xlabel("Объектов")
    ax.set_title("Покрытие по слоям пайплайна")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_rule_sources(rule_src, out_path):
    vc = rule_src.value_counts()
    if vc.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(vc.index.tolist(), vc.values.tolist(), color="#4C72B0")
    total = int(vc.sum())
    for bar, c in zip(bars, vc.values):
        ax.text(bar.get_width() + total * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{c}  ({c / total:.1%})", va="center")
    ax.set_xlim(0, int(vc.max()) * 1.25)
    ax.set_xlabel("Объектов")
    ax.set_title("Источники rule-based предсказаний")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_confusion(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(invalid="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.4),
                                    max(7, len(labels) * 0.4)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Доля")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_top_categories(y_true, y_pred, top_n, out_path):
    true_counts = Counter(y_true.tolist())
    pred_counts = Counter(y_pred.tolist())
    top = [c for c, _ in true_counts.most_common(top_n)]
    x = np.arange(len(top))
    w = 0.4

    fig, ax = plt.subplots(figsize=(max(10, len(top) * 0.5), 5))
    ax.bar(x - w / 2, [true_counts[c] for c in top], w, label="Истина", color="#4C72B0")
    ax.bar(x + w / 2, [pred_counts.get(c, 0) for c in top], w, label="Предсказано", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in top], rotation=60, fontsize=8)
    ax.set_ylabel("Объектов")
    ax.set_title(f"Top-{top_n} категорий: истина vs предсказание")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_top_errors(y_true, y_pred, top_n, min_support, out_path):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    rows = []
    for c in classes:
        mask = y_true == c
        sup = int(mask.sum())
        if sup < min_support:
            continue
        acc = float((y_pred[mask] == c).mean())
        rows.append((c, sup, acc, 1 - acc))
    rows.sort(key=lambda r: r[3], reverse=True)
    rows = rows[:top_n]
    if not rows:
        return

    cats = [str(r[0]) for r in rows]
    errs = [r[3] for r in rows]
    sups = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(max(10, len(cats) * 0.5), 5))
    bars = ax.bar(cats, errs, color="#C44E52")
    for bar, s, e in zip(bars, sups, errs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={s}\n{e:.0%}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, min(1.0, max(errs) * 1.3 + 0.05))
    ax.set_ylabel("Доля ошибок (1 - recall)")
    ax.set_title(f"Top-{top_n} категорий по ошибкам (support ≥ {min_support})")
    plt.xticks(rotation=60, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ---------- Main ----------

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"|{time.time() - start_time:.1f}s| загрузка модели {args.model}")
    m = load_model(args.model)

    print(f"|{time.time() - start_time:.1f}s| загрузка {args.test}")
    df = pd.read_csv(args.test, sep="\t")

    for col in ("category_id", "department_id"):
        if col not in df.columns:
            print(f"ОШИБКА: в {args.test} нет колонки '{col}' — нечем считать метрики")
            sys.exit(1)

    n = len(df)
    print(f"|{time.time() - start_time:.1f}s| строк: {n}")

    print(f"|{time.time() - start_time:.1f}s| предсказание")
    cat_pred, dep_pred, rule_src, n_fb, n_dept_fb = run_inference(df, m)

    y_cat = df["category_id"].values
    y_dep = df["department_id"].values

    # --- метрики ---
    dep_f1 = f1_score(y_dep, dep_pred, average="weighted", zero_division=0)
    dep_acc = accuracy_score(y_dep, dep_pred)
    dep_f1_macro = f1_score(y_dep, dep_pred, average="macro", zero_division=0)

    cat_acc = accuracy_score(y_cat, cat_pred)
    cat_f1 = f1_score(y_cat, cat_pred, average="weighted", zero_division=0)
    cat_f1_macro = f1_score(y_cat, cat_pred, average="macro", zero_division=0)

    score = 30 * dep_f1 + 70 * cat_acc

    # --- текстовый отчёт ---
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("-" * 64)
    emit("ОТЧЁТ ОБ ОЦЕНКЕ МОДЕЛИ")
    emit("-" * 64)
    emit(f"Модель:                {args.model}")
    emit(f"Тест:                  {args.test}")
    emit(f"Строк:                 {n}")
    emit("")
    emit("Покрытие:")
    emit(f"  rule-based:          {int(rule_src.notna().sum()):>7}  "
         f"({rule_src.notna().mean():.1%})")
    emit(f"  KNN:                 {n_fb:>7}  ({n_fb / n:.1%})")
    emit(f"  department fallback: {n_dept_fb:>7}  ({n_dept_fb / n:.1%})")
    emit(f"  category_id=-1:      {int((cat_pred == -1).sum()):>7}")
    emit(f"  department_id=-1:    {int((dep_pred == -1).sum()):>7}")
    emit("")
    emit("Метрики department_id:")
    emit(f"  accuracy:            {dep_acc:.4f}")
    emit(f"  F1 weighted:         {dep_f1:.4f}")
    emit(f"  F1 macro:            {dep_f1_macro:.4f}")
    emit("")
    emit("Метрики category_id:")
    emit(f"  accuracy:            {cat_acc:.4f}")
    emit(f"  F1 weighted:         {cat_f1:.4f}")
    emit(f"  F1 macro:            {cat_f1_macro:.4f}")
    emit("")
    emit(f"ИТОГОВЫЙ SCORE:        {score:.4f}  "
         f"(30*{dep_f1:.4f} + 70*{cat_acc:.4f})")
    emit("-" * 64)

    emit("")
    emit("Classification report — department_id:")
    emit(classification_report(y_dep, dep_pred, zero_division=0))

    emit("Classification report — category_id (топ-30 по support):")
    vc = pd.Series(y_cat).value_counts()
    top30 = vc.head(30).index.tolist()
    mask = np.isin(y_cat, top30)
    if mask.any():
        emit(classification_report(y_cat[mask], cat_pred[mask], zero_division=0))
    else:
        emit("  (нет данных)")

    report_path = out_dir / "metrics_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n|{time.time() - start_time:.1f}s| сохранён {report_path}")

    # --- predictions.csv ---
    preds_df = pd.DataFrame({
        "category_true":   y_cat,
        "category_pred":   cat_pred,
        "department_true": y_dep,
        "department_pred": dep_pred,
        "rule_source":     rule_src.values,
    })
    preds_path = out_dir / "predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"|{time.time() - start_time:.1f}s| сохранён {preds_path}")

    # --- графики ---
    print(f"|{time.time() - start_time:.1f}s| графики")

    plot_coverage(rule_src, n_fb, n_dept_fb, n, out_dir / "coverage.png")
    plot_rule_sources(rule_src, out_dir / "rule_sources.png")

    dep_labels = sorted(np.unique(y_dep).tolist())
    plot_confusion(y_dep, dep_pred, dep_labels,
                   "Confusion matrix — department_id (normalized)",
                   out_dir / "confusion_department.png")

    plot_top_categories(y_cat, cat_pred, args.top_n,
                        out_dir / "top_categories.png")
    plot_top_errors(y_cat, cat_pred, args.top_n, args.min_support,
                    out_dir / "top_errors.png")

    print(f"|{time.time() - start_time:.1f}s| готово. Все артефакты в {out_dir}/")


if __name__ == "__main__":
    main()