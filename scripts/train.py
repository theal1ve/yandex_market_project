import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import pickle
import gzip
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.training import train_full, full_predict, start_time


print(f"|{time.time()-start_time:.1f}s| загрузка")
df = pd.read_csv("data/train.tsv", sep="\t")
df = df.drop_duplicates(subset=[
                        "title", "description", "shop_category_name", "category_id", "department_id"])
print(f"|{time.time()-start_time:.1f}s| строк: {len(df)}")

train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df["department_id"])
known_cats = set(train_df["category_id"].unique())
val_df_f = val_df[val_df["category_id"].isin(
    known_cats)].copy().reset_index(drop=True)
print(f"|{time.time()-start_time:.1f}s| train: {len(train_df)}, val: {len(val_df_f)}")

print(f"\n|{time.time()-start_time:.1f}s| обучение на train")
res_v = train_full(train_df, "[val]")

TEMPERATURE = 1.0

print(f"\n|{time.time()-start_time:.1f}s| val метрики")
cat_p, dep_p = full_predict(val_df_f, *res_v)
dep_f1 = f1_score(val_df_f["department_id"], dep_p,
                  average="weighted", zero_division=0)
cat_acc = accuracy_score(val_df_f["category_id"], cat_p)
cat_f1 = f1_score(val_df_f["category_id"], cat_p,
                  average="weighted", zero_division=0)
val_score = 30*dep_f1 + 70*cat_acc
print(classification_report(val_df_f["department_id"], dep_p, zero_division=0))
print(f"department_id f1:  {dep_f1:.4f}")
print(f"category_id   f1:  {cat_f1:.4f}  acc: {cat_acc:.4f}")
print(f"score:    {val_score:.4f}")

print(f"\n|{time.time()-start_time:.1f}s| обучение на ВСЕХ данных")
res_f = train_full(df, "[final]")

print(f"\n|{time.time()-start_time:.1f}s| сохранение")
with gzip.open("model.pkl.gz", "wb", compresslevel=6) as f:
    for obj in res_f:
        pickle.dump(obj, f)
    pickle.dump(TEMPERATURE, f)

size = os.path.getsize("model.pkl.gz") / 1024 / 1024
print(f"|{time.time()-start_time:.1f}s| готово {size:.1f} мб")
print(f"score: {val_score:.4f}")