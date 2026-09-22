"""Признаки для TF-IDF"""
import re

import pandas as pd

from .preprocessing import clean_text, clean_desc, make_vendor, get_SCN_last


def title_features(title: str) -> str:
    if not title or pd.isna(title):
        return ""
    title_lower = str(title).lower()
    tokens: list = []
    words: list = re.sub(r'[^a-zа-я0-9\s]', ' ', title_lower).split()
    if words:
        tokens.append(f"__fw_{words[0]}__")
    if len(words) >= 2:
        tokens.append(f"__bg_{words[0]}_{words[1]}__")

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


def build_description_word_text(row) -> str:
    SCN = str(row.get("shop_category_name", "") or "").strip()
    SCN = "" if SCN in ["-", "nan", ""] else SCN
    if re.search(r'[\u4e00-\u9fff]', SCN):
        SCN = ""

    SCN_last = get_SCN_last(row.get("shop_category_name", ""))

    title = clean_text(row.get("title", ""))
    desc = clean_desc(str(row.get("description", "") or ""), 800)
    vendor = clean_text(str(row.get("vendor_name", "") or ""))
    vendor_code = str(row.get("vendor_code", "") or "")

    if re.search(r'jiemiwl|jiemi|romiky|juxiang|linglin|muzimao|qingye', vendor_code.lower()):
        vendor_code = ""

    title_fe = title_features(row.get("title", ""))

    return (f"{clean_text(SCN)} " * 6 + f"{clean_text(SCN_last)} " * 3 +
            f"{title} " * 2 + f"{title_fe} " + f"{vendor} " + f"{desc} " + vendor_code)


def build_description_char_text(row) -> str:
    SCN = str(row.get("shop_category_name", "") or "").strip()
    SCN = "" if SCN in ["-", "nan", ""] else SCN
    if re.search(r'[\u4e00-\u9fff]', SCN):
        SCN = ""
    title = str(row.get("title", "") or "").lower()[:300]
    vendor = str(row.get("vendor_name", "") or "").lower()[:50]
    vender_code = str(row.get("vendor_code", "") or "").lower()[:30]
    return f"{SCN.lower()} {title} {vendor} {vender_code}"