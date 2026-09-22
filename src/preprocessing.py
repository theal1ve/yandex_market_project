"""Очистка текста, лемматизация, работа с брендами и shop_category_name"""
import re

import pandas as pd
import pymorphy3


NO_BRAND = {
    "нет бренда", "без бренда", "no brand", "нет брендаs", "без брендаs", "",
    "没有品牌", "无品牌", "другие бренды", "другие", "другойбренд",
    "универсальный", "универсальная", "н/а", "н.а", "n/a", "na", "unknown",
    "jiemiwl", "romiky", "jiemi", "джи чонг", "juxiangying", "linglingmaoyi",
    "muzimaoyi", "qingyemaoyi", "nobrand", "no_brand", "oem", "generic", "прочие",
}

morph = pymorphy3.MorphAnalyzer()


def lemmatize(text: str) -> str:
    return " ".join(morph.parse(w)[0].normal_form for w in text.split())


def clean_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zа-я0-9\s\-]', ' ', text)
    return lemmatize(re.sub(r'\s+', ' ', text).strip())


def clean_desc(text: str, max_len: int = 1500) -> str:
    if not text or pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\\[rn]+', '\n', text)
    text = text[:max_len].lower()
    text = re.sub(r'[^a-zа-я0-9\s\-\n]', ' ', text)
    return lemmatize(re.sub(r'\s+', ' ', text).strip())


def make_vendor(row) -> str:
    name: str = str(row["vendor_name"]).strip(
    ).lower() if pd.notna(row["vendor_name"]) else ""
    if re.search(r'[\u4e00-\u9fff]', name):
        return ""
    if name and name not in NO_BRAND:
        return clean_text(row["vendor_name"])
    return ""


def get_SCN_last(SCN: str) -> str:
    if pd.isna(SCN) or str(SCN).strip() in ["", "-"]:
        return ""
    parts: list = str(SCN).split(" / ")
    if len(parts) > 1:
        last = parts[-1].strip()
        if re.search(r'[\u4e00-\u9fff]', last):
            return ""
        return last
    return ""