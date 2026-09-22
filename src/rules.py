"""Rule-based слой"""
import numpy as np
import pandas as pd


def apply_rules(df: pd.DataFrame, vendor_code_map: dict, SCN_category_map: dict,
                title_vendor_name_map: dict, title_SCN_map: dict) -> pd.Series:
    result = pd.Series(np.nan, index=range(len(df)))
    df = df.reset_index(drop=True)

    title_SCN_k = df["title"].fillna("").str.lower().str.strip(
    ) + "|||" + df["shop_category_name"].fillna("").str.lower().str.strip()
    result = result.where(title_SCN_k.map(title_SCN_map).isna(),
                          title_SCN_k.map(title_SCN_map))

    title_vendor_name_k = df["title"].fillna("").str.lower().str.strip(
    ) + "|||" + df["vendor_name"].fillna("").str.lower().str.strip()
    result = result.where(title_vendor_name_k.map(title_vendor_name_map).isna(),
                          title_vendor_name_k.map(title_vendor_name_map))

    result = result.where(df["shop_category_name"].map(
        SCN_category_map).isna(), df["shop_category_name"].map(SCN_category_map))

    result = result.where(df["vendor_code"].map(
        vendor_code_map).isna(), df["vendor_code"].map(vendor_code_map))

    return result


def build_rule_dicts(df: pd.DataFrame) -> tuple:
    vender_code = df.dropna(subset=["vendor_code"])
    vender_code_nu = vender_code.groupby("vendor_code")["category_id"].nunique()
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

def rule_sources(df: pd.DataFrame, vendor_code_map: dict, SCN_category_map: dict,
                 title_vendor_name_map: dict, title_SCN_map: dict) -> pd.Series:
    """Метка сработавшего правила для каждой строки"""
    df = df.reset_index(drop=True)
    src = pd.Series(np.nan, index=range(len(df)), dtype=object)

    title_SCN_k = df["title"].fillna("").str.lower().str.strip(
    ) + "|||" + df["shop_category_name"].fillna("").str.lower().str.strip()
    m = title_SCN_k.map(title_SCN_map).notna()
    src[m] = "title+SCN"

    title_vendor_name_k = df["title"].fillna("").str.lower().str.strip(
    ) + "|||" + df["vendor_name"].fillna("").str.lower().str.strip()
    m = title_vendor_name_k.map(title_vendor_name_map).notna() & src.isna()
    src[m] = "title+vendor_name"

    m = df["shop_category_name"].map(SCN_category_map).notna() & src.isna()
    src[m] = "SCN"

    m = df["vendor_code"].map(vendor_code_map).notna() & src.isna()
    src[m] = "vendor_code"

    return src