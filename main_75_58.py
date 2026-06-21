import pandas as pd
import numpy as np
import pickle
import gzip
import re
import time
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
from scipy.special import softmax
import pymorphy3

# засекаем время
start_time = time.time()

NO_BRAND = set(
    ("нет бренда", "без бренда", "no brand", "нет брендаs", "без брендаs", "",
    "没有品牌", "无品牌", "другие бренды", "другие", "другойбренд",
    "универсальный", "универсальная", "н/а", "н.а", "n/a", "na", "unknown",
    "jiemiwl", "romiky", "jiemi", "джи чонг", "juxiangying", "linglingmaoyi",
    "muzimaoyi", "qingyemaoyi", "nobrand", "no_brand", "oem", "generic", "прочие",)
)

morph = pymorphy3.MorphAnalyzer()


def lemmatize(text: str) -> str:
    # разбиваем текст на слова, берем их начальную форму
    return " ".join(morph.parse(w)[0].normal_form for w in text.split())


def clean_text(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # убираем теги
    text = re.sub(r'[^a-zа-я0-9\s\-]', ' ', text)  # цифры, буквы, пробелы
    return lemmatize(re.sub(r'\s+', ' ', text).strip())

# отдельно чистим описание


def clean_desc(text: str, max_len: int = 1500) -> str:
    if not text or pd.isna(text):
        return ""
    text = str(text)
    # заменяем <br> на перенос строки
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\\[rn]+', '\n', text)  # \\n заменяем \n
    text = text[:max_len].lower()  # обрезаем до max_len
    text = re.sub(r'[^a-zа-я0-9\s\-\n]', ' ', text)
    return lemmatize(re.sub(r'\s+', ' ', text).strip())


# обрабатываем vendor_name
def make_vendor(row: str) -> str:
    # если вендер нейм не пустой, оставляем. иначе пустая строка
    name: str = str(row["vendor_name"]).strip(
    ).lower() if pd.notna(row["vendor_name"]) else ""

    # проверка на китайские символы
    if re.search(r'[\u4e00-\u9fff]', name):
        return ""
    # проверяем, что бренд не пустой
    if name and name not in NO_BRAND:
        return clean_text(row["vendor_name"])
    return ""

# обработка shop_category_name


def get_SCN_last(SCN: str) -> str:
    if pd.isna(SCN) or str(SCN).strip() in ["", "-"]:
        return ""
    # разбиваем на разделы и берем последний
    parts: list = str(SCN).split(" / ")
    if len(parts) > 1:
        last = parts[-1].strip()
        if re.search(r'[\u4e00-\u9fff]', last):
            return ""
        return last
    return ""


# EDA на title
def title_features(title: str) -> str:
    if not title or pd.isna(title):
        return ""
    title_lower: str = str(title).lower()
    tokens: list = []
    # разбиваем заголовок на слова
    words: list = re.sub(r'[^a-zа-я0-9\s]', ' ', title_lower).split()
    if words:
        tokens.append(f"__fw_{words[0]}__")  # первое слово
    if len(words) >= 2:
        tokens.append(f"__bg_{words[0]}_{words[1]}__")  # первые два слова

    # ищем различные параметры товара
    numeric_features: dict = {
        "__feat_memory__": r'\d+\s*(гб|gb|мб|mb)',
        "__feat_watt__": r'\d+\s*(вт|w\b|ватт)',
        "__feat_inch__": r'\d+\s*(дюйм|inch|")',
        "__feat_volume__": r'\d+\s*(л\b|мл|литр)',
        "__feat_size__": r'\d+\s*(см|мм|mm|cm)\b',
        "__feat_weight__": r'\d+\s*(кг|г\b|грамм)',
        "__feat_volt__": r'\d+\s*(v\b|вольт)',
        "__feat_dims__": r'\d+x\d+'
    }
    for feature, pattern in numeric_features.items():
        if re.search(pattern, title_lower):
            tokens.append(feature)
    category_features: dict = {
        "__type_clothing__":  r'\b(футболк|джинс|платье|куртк|пальто|свитер|худи|леггинс|шорты|юбк|носки|трусы|бюстгальтер|купальник|колготки|кардиган|блузк|рубашк|толстовк)',
        "__type_shoes__":     r'\b(кроссовк|туфли|ботинки|сапоги|кеды|сандали|тапочк|мокасин|слипон)',
        "__type_food__":      r'\b(консерв|крупа|мука|сахар|чай|кофе|шоколад|конфет|печенье|масло|соус|лапш|рис\b|специ|пряност|приправ)',
        "__type_cosmetic__":  r'\b(крем|шампунь|гель|помад|тушь|тональн|духи|парфюм|дезодорант|лосьон|сыворотк|маска для лица|мицеллярн)',
        "__type_phone__":     r'\b(чехол|наушники|зарядк|powerbank|power bank|кабель usb|беспроводн зарядк|защитн стекл)',
        "__type_tool__":      r'\b(дрель|перфоратор|шуруповерт|болгарка|лобзик|отвертк|гайковерт|шлифовальн|сварочн|пневматическ|бесщеточн|гравер|сверл|фреза|плоскогубц|ключ разводн|зубило|стамеска|рубанок)',
        "__type_plumbing__":  r'\b(термостат|душевой|биде|смеситель|труб|клапан|фитинг|муфта|сифон|слив|унитаз|ванн|радиатор|котел|котёл|насос\b|помп|кран\b|водонагреватель)',
        "__type_electric__":  r'\b(розетк|выключател|автомат\b|щиток|провод\b|кабель\b|клемм|трансформатор|инвертор|диммер|реле\b|датчик|удлинитель)',
        "__type_garden__":    r'\b(семена|удобрение|горшок|лопат|грабли|шланг|газонокосилк|теплиц|садов|карбюратор|триммер|культиватор|поливочн|клумб|рассад|кашпо|скарификатор|аэратор|опрыскивател|мотоблок|бензопил)',
        "__type_pet__":       r'\b(корм для|лоток|поводок|ошейник|клетка|аквариум|переноск|когтеточк)',
        "__type_book__":      r'\b(книга|учебник|роман|повесть|сборник|энциклопедия|журнал|комикс)',
        "__type_pc__":        r'\b(ноутбук|монитор|принтер|клавиатур|мышь\b|ssd|nvme|процессор|видеокарт|материнск|оперативн|жестк.диск|usb.хаб|картридж|проектор|сканер|xbox|playstation|nintendo|ps[0-9]|wi-fi|роутер|системный блок|веб.камер)',
        "__type_appliance__": r'\b(пылесос|стиральн|посудомоечн|холодильник|морозильник|кондиционер|обогреватель|вентилятор\b|фен\b|мультиварк|блендер|миксер|тостер|утюг|кофемашин|соковыжималк|хлебопечк|аэрогриль)',
        "__type_climate__":   r'\b(нагреватель|тепловентилятор|увлажнитель|осушитель|вытяжк|котёл|котел|инфракрасн обогрев)',
        "__type_sport__":     r'\b(гантел|коврик для йоги|велосипед|самокат|ракетк|мяч|боксерск|груша\b|велотренажер|дайвинг|shimano|байдарк|каяк|туристическ|лыж|сноуборд|альпинизм|страховочн|спальный мешок)',
        "__type_home__":      r'\b(тумба|ваза|прихожей|прикроватн|мебель\b|стеллаж|светильник|люстра|бра\b|ночник|торшер|карниз|жалюзи|шторы|покрывало|подушк|одеял|скатерть|диван|кресл)',
        "__type_auto__":      r'\b(автомагнол|магнитол|мотогарнитур|видеорегистратор|автосигнализ|парктроник|автомобильн)',
    }
    for feature, pattern in category_features.items():
        if re.search(pattern, title_lower):
            tokens.append(feature)

    return " ".join(tokens)

# EDA на description


def desc_features(desc: str) -> str:
    if not desc or pd.isna(desc):
        return "__no_desc__"

    desc_lower: str = re.sub(r'<[^>]+>', ' ', str(desc)).lower()
    tokens: list = []

    features_dict = {
        "__desc_material__":    r'материал\s*:',
        "__desc_composition__": r'состав\s*:',
        "__desc_power__":       r'мощност',
        "__desc_size__":        r'размер\s*:',
        "__desc_usage__":       r'применени',
        "__desc_os__":         r'windows|macos|android|процессор|оперативн',
        "__desc_energy__":     r'электропотреблени|потребляемая мощност|энергопотреблени',
        "__desc_garden__":     r'посадк|почв|поли[вт]|садов|огород',
        "__desc_install__":    r'установк|монтаж|крепл|сборк',
        "__desc_plumbing__":   r'диаметр|резьба|dn\d|pp-r|pex|фитинг',
        "__desc_kids__":       r'возраст|для детей|детск',
        "__desc_material_kw__": r'нержавеющ|алюминий|пластик|металл|дерев|силикон|керамик'
    }

    for feature, pattern in features_dict.items():
        if re.search(pattern, desc_lower):
            tokens.append(feature)

    if len(desc_lower) < 100:
        tokens.append("__desc_short__")

    elif len(desc_lower) > 2000:
        tokens.append("__desc_long__")

    return " ".join(tokens)


def build_category_text(df: pd.DataFrame) -> str:
    df = df.reset_index(drop=True)

    SCN = df["shop_category_name"].fillna("").apply(lambda x: clean_text(x) if str(
        x).strip() not in ["-", ""] and not re.search(r'[\u4e00-\u9fff]', str(x)) else "")

    SCN_last = df["shop_category_name"].apply(get_SCN_last).apply(clean_text)
    title = df["title"].fillna("").apply(clean_text)
    desc = df["description"].apply(lambda x: clean_desc(x))
    vendor = df.apply(make_vendor, axis=1)
    title_fe = df["title"].apply(title_features)
    desc_fe = df["description"].apply(desc_features)

    return (title+" ")*4 + (SCN+" ")*2 + (SCN_last+" ")*3 + (title_fe+" ")*3 + vendor+" " + desc+" " + desc_fe


# для глубокого анализа
def build_description_word_text(row: str) -> str:

    SCN = str(row.get("shop_category_name", "") or "").strip()
    SCN = "" if SCN in ["-", "nan", ""] else SCN
    if re.search(r'[\u4e00-\u9fff]', SCN):
        SCN = ""

    SCN_last = get_SCN_last(row.get("shop_category_name", ""))

    title = clean_text(row.get("title", ""))
    desc = clean_desc(str(row.get("description", "") or ""), 800)
    vendor = clean_text(str(row.get("vendor_name", "") or ""))
    vendor_code = str(row.get("vendor_code", "") or "")

    # фильтруем китайских перекупов
    if re.search(r'jiemiwl|jiemi|romiky|juxiang|linglin|muzimao|qingye', vendor_code.lower()):
        vendor_code = ""

    title_fe = title_features(row.get("title", ""))

    return (f"{clean_text(SCN)} " * 6 + f"{clean_text(SCN_last)} " * 3 +
            f"{title} " * 2 + f"{title_fe} " + f"{vendor} " + f"{desc} " + vendor_code)

# для быстрого анализа


def build_description_char_text(row: str) -> str:
    SCN = str(row.get("shop_category_name", "") or "").strip()
    SCN = "" if SCN in ["-", "nan", ""] else SCN
    if re.search(r'[\u4e00-\u9fff]', SCN):
        SCN = ""
    title = str(row.get("title", "") or "").lower()[:300]
    vendor = str(row.get("vendor_name", "") or "").lower()[:50]
    vender_code = str(row.get("vendor_code", "") or "").lower()[:30]
    return f"{SCN.lower()} {title} {vendor} {vender_code}"


def apply_rules(df: pd.DataFrame, vendor_code_map: dict, SCN_category_map: dict, title_vendor_name_map: dict, title_SCN_map: dict) -> pd.DataFrame:
    result = pd.Series(np.nan, index=range(len(df)))

    df = df.reset_index(drop=True)

    # создаем ключ для сопоставления title + SCN
    title_SCN_k: str = df["title"].fillna("").str.lower().str.strip(
    ) + "|||" + df["shop_category_name"].fillna("").str.lower().str.strip()

    # применяем
    result = result.where(title_SCN_k.map(title_SCN_map).isna(),
                          title_SCN_k.map(title_SCN_map))

    # создаем ключ для сопоставления title + vendor_name
    title_vendor_name_k: str = df["title"].fillna("").str.lower().str.strip(
    ) + "|||" + df["vendor_name"].fillna("").str.lower().str.strip()

    # применяем
    result = result.where(title_vendor_name_k.map(title_vendor_name_map).isna(),
                          title_vendor_name_k.map(title_vendor_name_map))

    # сопоставление по SCN
    result = result.where(df["shop_category_name"].map(
        SCN_category_map).isna(), df["shop_category_name"].map(SCN_category_map))

    # сопоставление по vender_code
    result = result.where(df["vendor_code"].map(
        vendor_code_map).isna(),
        df["vendor_code"].map(vendor_code_map))

    return result


def predict_bayes_knn(X_val_category: np.ndarray, X_val_department: np.ndarray, SVC_department,
                      X_norm_train: np.ndarray, category_labels_train: list, train_department_idx_train: list,
                      SCN_val: list, SCN_prior: list, uniform_prior: np.ndarray, all_category_ids: list, category_to_idx: list,
                      fallback: int, temperature: float = 1.0, beta: float = 0.3) -> np.ndarray:

    # l2 норма к данным
    X_val_norm: np.ndarray = normalize(
        X_val_category, norm="l2").toarray().astype(np.float32)

    # косинусные сходства
    sims: np.ndarray = X_val_norm @ X_norm_train.T

    # расчет вероятностей department
    department_scores = SVC_department.decision_function(X_val_department)
    department_proba = softmax(department_scores / temperature, axis=1)

    weighted = sims * department_proba[:, train_department_idx_train]

    # апорные вероятности
    if beta > 0.0:
        # получение индексов
        n_val = X_val_category.shape[0]
        train_category_idx = np.array([category_labels_train.get(c, 0)
                                       for c in category_labels_train], dtype=np.int32)
        # создание матрицы
        SCN_prior_mat = np.empty(
            (n_val, len(category_labels_train)), dtype=np.float32)
        
        for i, SCN in enumerate(SCN_val):
            SCN_str = str(SCN).strip() if pd.notna(SCN) else ""
            # если пусто, то использует равномерное распред
            SCN_prior_mat[i] = uniform_prior if SCN_str in ("", "-", "nan") else SCN_prior.get(SCN_str, uniform_prior)

        neighbor_SCN_prior = SCN_prior_mat[:, train_category_idx]
        if beta != 1.0:
            neighbor_SCN_prior = np.power(neighbor_SCN_prior, beta)
        weighted = weighted * neighbor_SCN_prior

    # категория с макс взвешенным сходством
    return category_labels_train[np.argmax(weighted, axis=1)]


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
        SVC_department: dict = pickle.load(f)   
        vender_code_map: dict = pickle.load(f)   
        SCN_category_map: dict = pickle.load(f)   
        SCN_deparment_map: dict = pickle.load(f)   
        title_vender_name_map: dict = pickle.load(f)   
        title_SCN_map: dict = pickle.load(f)   
        category_to_department: dict = pickle.load(f)   
        SCN_prior: dict = pickle.load(f)   
        uniform_prior: np.ndarray = pickle.load(f)   
        all_category_ids: list = pickle.load(f)   
        category_to_idx: dict = pickle.load(f)
        temperature: float = pickle.load(f)   
        beta: float = pickle.load(f)   

    print(f"|{time.time()-start_time:.1f}s| модель загружена")

    print(f"|{time.time()-start_time:.1f}s| загрузка тестовых данных")

    test: pd.DataFrame = pd.read_csv("test.tsv", sep="\t")
    test = test.reset_index(drop=True)
    n = len(test)

    print(f"|{time.time()-start_time:.1f}s| тест")

    category_preds: np.ndarray = np.full(n, -1, dtype=np.int64)

    rule_pred: np.ndarray = apply_rules(test, vender_code_map, SCN_category_map, title_vender_name_map, title_SCN_map)
    rule_mask: pd.Series = rule_pred.notna().values
    category_preds[rule_mask] = rule_pred[rule_mask].values.astype(np.int64)

    print(f"|{time.time()-start_time:.1f}s| rule-based {rule_mask.sum()}/{n} = {rule_mask.mean():.1%}")

    # KNN + SC prior для непокрытых
    fb_mask = ~rule_mask
    if fb_mask.sum() > 0:
        df_fb: pd.DataFrame = test[fb_mask].reset_index(drop=True)
        print(f"|{time.time()-start_time:.1f}s| KNN + SC prior для {fb_mask.sum()} строк")
        
        X_category_fb = tf_idf_category.transform(build_category_text(df_fb))
        X_word_fb = t_word_department.transform(df_fb.apply(
            build_description_word_text, axis=1))
        X_char_fb = t_char_department.transform(df_fb.apply(
            build_description_char_text, axis=1))
        X_department_fb = hstack([X_word_fb, X_char_fb])
        SCN_fb = df_fb["shop_category_name"].values
        category_fb = predict_bayes_knn(
            X_category_fb, X_department_fb, SVC_department,
            X_norm_train, category_labels_train, train_department_idx_train,
            SCN_fb,
            SCN_prior, uniform_prior, all_category_ids, category_to_idx,
            fallback, temperature=temperature, beta=beta
        )
        category_preds[fb_mask] = category_fb

    department_preds = np.array([category_to_department.get(int(c), -1)
                         for c in category_preds], dtype=np.int64)

    # маска для неизвестных значений
    unk = department_preds == -1
    if unk.sum() > 0:
        print(f"|{time.time()-start_time:.1f}s| department fallback для {unk.sum()} строк")

        # векторизация текстовых данных для неизвестных
        Xw = t_word_department.transform(test[unk].apply(
            build_description_word_text, axis=1))
        Xc = t_char_department.transform(test[unk].apply(
            build_description_char_text, axis=1))
        

        # предсказание отдела для неизвестных через SVC
        department_preds[unk] = SVC_department.predict(hstack([Xw, Xc])).astype(int)

        # для категорий, которые остались неизвестными, ставим значение fallback
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
