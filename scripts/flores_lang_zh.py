"""FLORES-200 风格语言码 -> 中文语言名（Alpaca instruction 用）；未命中则返回原码。"""


FLORES_LANG_ZH: dict[str, str] = {
    "eng_Latn": "英语",
    "zho_Hans": "简体中文",
    "zho_Hant": "繁体中文",
    "spa_Latn": "西班牙语",
    "fra_Latn": "法语",
    "deu_Latn": "德语",
    "ita_Latn": "意大利语",
    "por_Latn": "葡萄牙语",
    "rus_Cyrl": "俄语",
    "ukr_Cyrl": "乌克兰语",
    "pol_Latn": "波兰语",
    "nld_Latn": "荷兰语",
    "swe_Latn": "瑞典语",
    "dan_Latn": "丹麦语",
    "nob_Latn": "挪威语（书面）",
    "fin_Latn": "芬兰语",
    "ces_Latn": "捷克语",
    "ell_Grek": "希腊语",
    "heb_Hebr": "希伯来语",
    "tur_Latn": "土耳其语",
    "arb_Arab": "阿拉伯语",
    "fas_Arab": "波斯语",
    "hin_Deva": "印地语",
    "ben_Beng": "孟加拉语",
    "urd_Arab": "乌尔都语",
    "ind_Latn": "印度尼西亚语",
    "vie_Latn": "越南语",
    "tha_Thai": "泰语",
    "tgl_Latn": "他加禄语",
    "jpn_Jpan": "日语",
    "kor_Hang": "韩语",
    "zsm_Latn": "马来语",
    "mya_Mymr": "缅甸语",
    "khm_Khmr": "高棉语",
    "lao_Laoo": "老挝语",
    "cmn_Hans": "普通话（简体）",
    "yue_Hant": "粤语",
}


def flores_code_to_zh_name(code: str) -> str:
    return FLORES_LANG_ZH.get(code.strip(), code.strip())
