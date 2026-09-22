# Product Category Classifier

Классификация товаров интернет-магазина: предсказание `category_id`
и `department_id` по текстовым признакам.

## Пайплайн

1. **Rule-based** Словари однозначных соответствий:
   `vendor_code -> category_id`, `shop_category_name -> category_id`,
   `(title, shop_category_name) -> category_id`, `(title, vendor_name) -> category_id`.
2. **KNN по TF-IDF с байесовским взвешиванием**
   Косинусная близость умножается на вероятность отдела (`LinearSVC + softmax`).
3. **Department fallback** Для категорий без отображения отдел
   предсказывается `LinearSVC` на `hstack(word TF-IDF, char_wb TF-IDF)`.

## Признаки

- **Title:** лемматизация (pymorphy3), первые 1–2 слова как маркеры,
  regex-детекторы параметров (мощность, объём, дюймы, ...) и 15+ товарных типов.
- **Shop category name:** очистка, последний сегмент иерархии, отсев китайских иероглифов.
- **Description:** очистка HTML, обрезка до 1500 символов, regex-детекторы.
- **Vendor name:** отсев «без бренда» и китайских перекупов.

## Стек

Python 3.10+, pandas, numpy, scikit-learn, scipy, pymorphy3.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pymorphy3 download
```

## Обучение

Требуется `train.tsv` в корне проекта:

```bash
python scripts/train.py
```

Сохраняет `model.pkl.gz` (18 объектов: 17 из пайплайна + `TEMPERATURE`).

## Тест 

Требуется `test.tsv` в корне проекта
Создан `test.tsv` из `train.tsv` для проверки
(Модель не обучалась на данных, которые в `test.tsv`)

```bash
python scripts/evaluate.py
```

Сохраняет графики и текстовый отчет в reports/

## Инференс

Требуется `test.tsv` в корне проекта:

```bash
python scripts/predict.py
```

Сохраняет `prediction.csv` с колонками `category_id`, `department_id`.

## Структура

```
src/
  preprocessing.py  — очистка и лемматизация
  features.py       — признаки для TF-IDF
  rules.py          — rule-based слой
  training.py       — обучение
  inference.py      — KNN-предсказание
scripts/
  train.py
  predict.py
```