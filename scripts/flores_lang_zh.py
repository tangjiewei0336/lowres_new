"""FLORES-200 风格语言码 -> 语言名；未命中则返回原码。"""


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


FLORES_LANG_EN: dict[str, str] = {
    "eng_Latn": "English",
    "zho_Hans": "Simplified Chinese",
    "zho_Hant": "Traditional Chinese",
    "spa_Latn": "Spanish",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "ukr_Cyrl": "Ukrainian",
    "pol_Latn": "Polish",
    "nld_Latn": "Dutch",
    "swe_Latn": "Swedish",
    "dan_Latn": "Danish",
    "nob_Latn": "Norwegian Bokmal",
    "fin_Latn": "Finnish",
    "ces_Latn": "Czech",
    "ell_Grek": "Greek",
    "heb_Hebr": "Hebrew",
    "tur_Latn": "Turkish",
    "arb_Arab": "Arabic",
    "fas_Arab": "Persian",
    "hin_Deva": "Hindi",
    "ben_Beng": "Bengali",
    "urd_Arab": "Urdu",
    "ind_Latn": "Indonesian",
    "vie_Latn": "Vietnamese",
    "tha_Thai": "Thai",
    "tgl_Latn": "Tagalog",
    "kor_Hang": "Korean",
    "zsm_Latn": "Malay",
    "mya_Mymr": "Burmese",
    "khm_Khmr": "Khmer",
    "lao_Laoo": "Lao",
    "cmn_Hans": "Mandarin Chinese (Simplified)",
    "yue_Hant": "Cantonese",
}


def flores_code_to_en_name(code: str) -> str:
    return FLORES_LANG_EN.get(code.strip(), code.strip())


def english_translation_instruction(src_lang: str, tgt_lang: str) -> str:
    src = flores_code_to_en_name(src_lang)
    tgt = flores_code_to_en_name(tgt_lang)
    return f"Translate the following {src} text into {tgt}. Output only the translation."
