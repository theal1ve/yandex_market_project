import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import pickle
import gzip
import time
from scipy.sparse import hstack

from src.features import (
    build_category_text,
    build_description_char_text,
    build_description_word_text,
)
from src.rules import apply_rules
from src.inference import predict_bayes_knn


start_time = time.time()


def main() -> None:
    print(f"|{time.time()-start_time:.1f}s| загрузка модели")
    with gzip.open("model.pkl.gz", "rb") as f:
        tf_idf_category = pickle.load(f)
        X_norm_train: np.ndarray = pickle.load(f)
        category_labels_train: list = pickle.load(f)
        department_labels_train: list = pickle.load(f)
        department_classes: list = pickle.load(f)
        department_to_idx: dict = pickle.load(f)
        train_department_idx_train: list = pickle.load(f)
        fallback: int = pickle.load(f)
        t_word_department = pickle.load(f)
        t_char_department = pickle.load(f)
        SVC_department = pickle.load(f)
        vender_code_map: dict = pickle.load(f)
        SCN_category_map: dict = pickle.load(f)
        SCN_deparment_map: dict = pickle.load(f)
        title_vender_name_map: dict = pickle.load(f)
        title_SCN_map: dict = pickle.load(f)
        category_to_department: dict = pickle.load(f)
        temperature: float = pickle.load(f)

    print(f"|{time.time()-start_time:.1f}s| модель загружена")

    print(f"|{time.time()-start_time:.1f}s| загрузка тестовых данных")
    test: pd.DataFrame = pd.read_csv("data/test.tsv", sep="\t")
    test = test.reset_index(drop=True)
    n = len(test)

    print(f"|{time.time()-start_time:.1f}s| тест")

    category_preds: np.ndarray = np.full(n, -1, dtype=np.int64)

    rule_pred = apply_rules(test, vender_code_map, SCN_category_map,
                            title_vender_name_map, title_SCN_map)
    rule_mask = rule_pred.notna().values
    category_preds[rule_mask] = rule_pred[rule_mask].values.astype(np.int64)

    print(f"|{time.time()-start_time:.1f}s| rule-based {rule_mask.sum()}/{n} = {rule_mask.mean():.1%}")

    fb_mask = ~rule_mask
    if fb_mask.sum() > 0:
        df_fb: pd.DataFrame = test[fb_mask].reset_index(drop=True)
        print(f"|{time.time()-start_time:.1f}s| KNN для {fb_mask.sum()} строк")

        X_category_fb = tf_idf_category.transform(build_category_text(df_fb))
        X_word_fb = t_word_department.transform(df_fb.apply(build_description_word_text, axis=1))
        X_char_fb = t_char_department.transform(df_fb.apply(build_description_char_text, axis=1))
        X_department_fb = hstack([X_word_fb, X_char_fb])

        category_fb = predict_bayes_knn(
            X_category_fb, X_department_fb, SVC_department,
            X_norm_train, category_labels_train, train_department_idx_train,
            temperature=temperature,
        )
        category_preds[fb_mask] = category_fb

    department_preds = np.array([category_to_department.get(int(c), -1)
                                 for c in category_preds], dtype=np.int64)

    unk = department_preds == -1
    if unk.sum() > 0:
        print(f"|{time.time()-start_time:.1f}s| department fallback для {unk.sum()} строк")
        Xw = t_word_department.transform(test[unk].apply(build_description_word_text, axis=1))
        Xc = t_char_department.transform(test[unk].apply(build_description_char_text, axis=1))
        department_preds[unk] = SVC_department.predict(hstack([Xw, Xc])).astype(int)
        category_preds[unk & (category_preds == -1)] = fallback

    print(f"|{time.time()-start_time:.1f}s| сохранение prediction.csv")
    out = pd.DataFrame({
        "category_id":   category_preds,
        "department_id": department_preds,
    })
    out.to_csv("prediction.csv", index=False)
    print(f"|{time.time()-start_time:.1f}s| готово")


if __name__ == "__main__":
    main()