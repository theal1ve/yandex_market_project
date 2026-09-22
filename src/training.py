"""Обучение пайплайна"""
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

from .features import (
    build_category_text,
    build_description_char_text,
    build_description_word_text,
)
from .rules import apply_rules, build_rule_dicts


start_time = time.time()


def train_bayes_knn(df: pd.DataFrame, tf_idf_category) -> tuple:
    df = df.reset_index(drop=True)
    X: np.ndarray = tf_idf_category.transform(build_category_text(df))
    X_norm: np.ndarray = normalize(X, norm="l2").toarray().astype(np.float16)
    category_labels: np.ndarray = df["category_id"].values
    department_labels: np.ndarray = df["department_id"].values
    fallback: int = int(df["category_id"].value_counts().index[0])
    department_classes: np.ndarray = np.array(sorted(df["department_id"].unique()))
    department_to_idx: dict = {d: i for i, d in enumerate(department_classes)}
    train_department_idx: np.ndarray = np.array([department_to_idx[d] for d in department_labels])
    print(f"  KNN: {len(df)} train vectors  |{time.time()-start_time:.1f}s|")
    return X_norm, category_labels, department_labels, department_classes, department_to_idx, train_department_idx, fallback


def full_predict(df: pd.DataFrame, tf_idf_category,
                 X_norm_train: np.ndarray, category_labels_train: list, department_labels_train: list,
                 department_classes: list, department_to_idx: dict, train_department_idx_train: list, fallback: int,
                 t_word_department, t_char_department, SVC_department: dict,
                 vender_code_map: dict, SCN_category_map: dict, SCN_deparment_map: dict,
                 title_vender_name_map: dict, title_SCN_map: dict, category_to_department: dict,
                 temperature: int = 1.0, batch_size=64):

    df = df.reset_index(drop=True)
    n = len(df)

    category_preds = np.full(n, -1, dtype=np.int64)
    rule_pred = apply_rules(df, vender_code_map, SCN_category_map, title_vender_name_map, title_SCN_map)
    rule_mask = rule_pred.notna().values
    category_preds[rule_mask] = rule_pred[rule_mask].values.astype(np.int64)
    print(f"  rule-based: {rule_mask.sum()}/{n} = {rule_mask.mean():.1%}")

    fb_mask = ~rule_mask
    if fb_mask.sum() > 0:
        df_fb = df[fb_mask].reset_index(drop=True)
        n_fb = len(df_fb)
        X_cat_fb = tf_idf_category.transform(build_category_text(df_fb))
        X_word_fb = t_word_department.transform(df_fb.apply(build_description_word_text, axis=1))
        X_char_fb = t_char_department.transform(df_fb.apply(build_description_char_text, axis=1))
        X_dep_fb = hstack([X_word_fb, X_char_fb])

        X_val_norm = normalize(X_cat_fb, norm="l2").toarray().astype(np.float32)
        dep_scores = SVC_department.decision_function(X_dep_fb)
        dep_proba = softmax(dep_scores / temperature, axis=1)

        fb_cat_preds = np.empty(n_fb, dtype=np.int64)
        for start in range(0, n_fb, batch_size):
            end = min(start + batch_size, n_fb)
            sims = X_val_norm[start:end] @ X_norm_train.T
            weighted = sims * dep_proba[start:end][:, train_department_idx_train]
            fb_cat_preds[start:end] = category_labels_train[np.argmax(weighted, axis=1)]
            if start % 256 == 0:
                print(f"    KNN {end}/{n_fb}  |{time.time()-start_time:.0f}s|", end="\r")
        print()
        category_preds[fb_mask] = fb_cat_preds

    dep_preds = np.array([category_to_department.get(int(c), -1)
                         for c in category_preds], dtype=np.int64)

    unk = dep_preds == -1
    if unk.sum() > 0:
        Xw = t_word_department.transform(df[unk].apply(build_description_word_text, axis=1))
        Xc = t_char_department.transform(df[unk].apply(build_description_char_text, axis=1))
        dep_preds[unk] = SVC_department.predict(hstack([Xw, Xc])).astype(int)

    return category_preds, dep_preds


def train_full(df: pd.DataFrame, tag: str = ""):
    df = df.reset_index(drop=True)

    print(f"|{time.time()-start_time:.1f}s| {tag} rule-based")
    vc_map, sc_cat_map, sc_dep_map, tvn_map, tsc_map, cat_to_dep = build_rule_dicts(df)
    print(f"  rule покрытие: {apply_rules(df, vc_map, sc_cat_map, tvn_map, tsc_map).notna().mean():.1%}")

    print(f"|{time.time()-start_time:.1f}s| {tag} TF-IDF category")
    tfidf_cat = TfidfVectorizer(max_features=50000, sublinear_tf=True,
                                ngram_range=(1, 2), min_df=1, dtype=np.float32)
    tfidf_cat.fit(build_category_text(df))

    print(f"|{time.time()-start_time:.1f}s| {tag} KNN matrix ")
    (X_norm_tr, cat_labels_tr, dep_labels_tr,
     dep_classes, dep_to_idx, train_dep_idx_tr, fallback) = train_bayes_knn(df, tfidf_cat)

    print(f"|{time.time()-start_time:.1f}s| {tag} department TF-IDF + SVC")
    tw_dep = TfidfVectorizer(max_features=100000, ngram_range=(1, 3),
                             sublinear_tf=True, min_df=1, dtype=np.float32)
    tc_dep = TfidfVectorizer(max_features=50000, analyzer="char_wb",
                             ngram_range=(3, 5), sublinear_tf=True, min_df=2, dtype=np.float32)
    X_word = tw_dep.fit_transform(df.apply(build_description_word_text, axis=1))
    X_char = tc_dep.fit_transform(df.apply(build_description_char_text, axis=1))
    svc_dep = LinearSVC(C=1.0, max_iter=5000, dual=True, class_weight="balanced")
    svc_dep.fit(hstack([X_word, X_char]), df["department_id"].values)

    return (tfidf_cat,
            X_norm_tr, cat_labels_tr, dep_labels_tr,
            dep_classes, dep_to_idx, train_dep_idx_tr, fallback,
            tw_dep, tc_dep, svc_dep,
            vc_map, sc_cat_map, sc_dep_map, tvn_map, tsc_map, cat_to_dep)