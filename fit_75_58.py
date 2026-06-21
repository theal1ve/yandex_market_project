import pandas as pd
import numpy as np
import pickle
import gzip
import re
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
from scipy.special import softmax
import pymorphy3


# засекаем время
start_time = time.time()

NO_BRAND = {
    "нет бренда", "без бренда", "no brand", "нет брендаs", "без брендаs", "",
    "没有品牌", "无品牌", "другие бренды", "другие", "другойбренд",
    "универсальный", "универсальная", "н/а", "н.а", "n/a", "na", "unknown",
    "jiemiwl", "romiky", "jiemi", "джи чонг", "juxiangying", "linglingmaoyi",
    "muzimaoyi", "qingyemaoyi", "nobrand", "no_brand", "oem", "generic", "прочие",
}

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
    title_lower = str(title).lower()
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


def build_rule_dicts(df: pd.DataFrame) -> tuple:
    vender_code = df.dropna(subset=["vendor_code"])
    vender_code_nu = vender_code.groupby(
        "vendor_code")["category_id"].nunique()
    
    vender_code_map = vender_code.groupby("vendor_code")["category_id"].first(
    ).loc[vender_code_nu[vender_code_nu == 1].index].to_dict()

    SCN = df[df["shop_category_name"].notna() & (df["shop_category_name"].str.strip(
    ) != "") & (df["shop_category_name"].str.strip() != "-")]

    SCN_nu = SCN.groupby("shop_category_name")["category_id"].nunique()

    SCN_category_map = SCN.groupby("shop_category_name")[
        "category_id"].first().loc[SCN_nu[SCN_nu == 1].index].to_dict()
    
    SCN_department_nu = SCN.groupby("shop_category_name")["department_id"].nunique()

    SCN_department_map = SCN.groupby("shop_category_name")["department_id"].first(
    ).loc[SCN_department_nu[SCN_department_nu == 1].index].to_dict()

    t_vender_name_key = df["title"].fillna("").str.lower().str.strip(
    ) + "|||" + df["vendor_name"].fillna("").str.lower().str.strip()

    t_vender_name_nu = df.groupby(t_vender_name_key)["category_id"].nunique()
    t_vender_name_map = df.groupby(t_vender_name_key)["category_id"].first(
    ).loc[t_vender_name_nu[t_vender_name_nu == 1].index].to_dict()

    t_SCN_key = df["title"].fillna("").str.lower().str.strip(
    ) + "|||" + df["shop_category_name"].fillna("").str.lower().str.strip()
    t_SCN_nu = df.groupby(t_SCN_key)["category_id"].nunique()
    t_SCN_map = df.groupby(t_SCN_key)["category_id"].first(
    ).loc[t_SCN_nu[t_SCN_nu == 1].index].to_dict()
    category_to_department = df.groupby("category_id")["department_id"].first().to_dict()
    return vender_code_map, SCN_category_map, SCN_department_map, t_vender_name_map, t_SCN_map, category_to_department


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
    print(
        f"  KNN: {len(df)} train vectors  |{time.time()-start_time:.1f}s|")
    return X_norm, category_labels, department_labels, department_classes, department_to_idx, train_department_idx, fallback


def full_predict(df: pd.DataFrame, tf_idf_category,
                 X_norm_train: np.ndarray, category_labels_train: list, department_labels_train: list,
                 department_classes: list, department_to_idx: dict, train_department_idx_train: list, fallback: int,
                 t_word_department, t_char_department, SVC_department: dict,
                 vender_code_map: dict, SCN_category_map: dict, SCN_deparment_map: dict, title_vender_name_map: dict, 
                 title_SCN_map: dict, category_to_department: dict,
                 temperature: int = 1.0, batch_size = 64):
    
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
        X_word_fb = t_word_department.transform(df_fb.apply(
            build_description_word_text, axis=1))
        X_char_fb = t_char_department.transform(df_fb.apply(
            build_description_char_text, axis=1))
        X_dep_fb = hstack([X_word_fb, X_char_fb])

        X_val_norm = normalize(
            X_cat_fb, norm="l2").toarray().astype(np.float32)
        dep_scores = SVC_department.decision_function(X_dep_fb)
        dep_proba = softmax(dep_scores / temperature, axis=1)

        fb_cat_preds = np.empty(n_fb, dtype=np.int64)
        for start in range(0, n_fb, batch_size):
            end = min(start + batch_size, n_fb)
            sims = X_val_norm[start:end] @ X_norm_train.T
            weighted = sims * dep_proba[start:end][:, train_department_idx_train]
            fb_cat_preds[start:end] = category_labels_train[np.argmax(
                weighted, axis=1)]
            if start % 256 == 0:
                print(
                    f"    KNN {end}/{n_fb}  |{time.time()-start_time:.0f}s|", end="\r")
        print()
        category_preds[fb_mask] = fb_cat_preds

    dep_preds = np.array([category_to_department.get(int(c), -1)
                         for c in category_preds], dtype=np.int64)

    unk = dep_preds == -1
    if unk.sum() > 0:
        Xw = t_word_department.transform(df[unk].apply(
            build_description_word_text, axis=1))
        Xc = t_char_department.transform(df[unk].apply(
            build_description_char_text, axis=1))
        dep_preds[unk] = SVC_department.predict(hstack([Xw, Xc])).astype(int)

    return category_preds, dep_preds


def train_full(df: pd.DataFrame, tag: str = ""):
    df = df.reset_index(drop=True)

    print(f"|{time.time()-start_time:.1f}s| {tag} rule-based")
    vc_map, sc_cat_map, sc_dep_map, tvn_map, tsc_map, cat_to_dep = build_rule_dicts(
        df)
    print(
        f"  rule покрытие: {apply_rules(df, vc_map, sc_cat_map, tvn_map, tsc_map).notna().mean():.1%}")

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
    X_word = tw_dep.fit_transform(
        df.apply(build_description_word_text, axis=1))
    X_char = tc_dep.fit_transform(
        df.apply(build_description_char_text, axis=1))
    svc_dep = LinearSVC(C=1.0, max_iter=5000, dual=True,
                        class_weight="balanced")
    svc_dep.fit(hstack([X_word, X_char]), df["department_id"].values)

    return (tfidf_cat,
            X_norm_tr, cat_labels_tr, dep_labels_tr,
            dep_classes, dep_to_idx, train_dep_idx_tr, fallback,
            tw_dep, tc_dep, svc_dep,
            vc_map, sc_cat_map, sc_dep_map, tvn_map, tsc_map, cat_to_dep)


print(f"|{time.time()-start_time:.1f}s| загрузка")
df = pd.read_csv("train.tsv", sep="\t")
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
