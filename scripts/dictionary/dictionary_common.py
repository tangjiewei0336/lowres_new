#!/usr/bin/env python3
"""Shared helpers for dictionary-based MT data preparation."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


LOW_RESOURCE_LANGS = ("spa_Latn", "ind_Latn", "vie_Latn", "tha_Thai", "tgl_Latn")
PIVOT_LANGS = ("eng_Latn", "zho_Hans")

EN_PIVOT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}

LANG_EN_NAMES = {
    "eng_Latn": "English",
    "zho_Hans": "Simplified Chinese",
    "spa_Latn": "Spanish",
    "ind_Latn": "Indonesian",
    "vie_Latn": "Vietnamese",
    "tha_Thai": "Thai",
    "tgl_Latn": "Tagalog",
}

MUSE_CODES = {
    "eng_Latn": "en",
    "zho_Hans": "zh",
    "spa_Latn": "es",
    "ind_Latn": "id",
    "vie_Latn": "vi",
    "tha_Thai": "th",
    "tgl_Latn": "tl",
}

TRADITIONAL_CHARS = set(
    "與萬專業東絲丟兩嚴喪個豐臨為麗舉麼義烏樂喬習鄉書買亂爭於虧雲亞產畝親褻褸億僅從侖倉儀們價眾優夥會傘偉傳傷倫偽佇體餘傭傾僂僑僕僥僨價儉償兒兌黨蘭關興內岡冊寫軍農馮衝決況凍淨淒準涼減湊凜幾鳳憑凱擊鑿芻劃劉則剛創刪別剎劑剝劇勸辦務動勵勁勞勢勳勝區醫華協單賣盧衛卻廠廳歷厲壓厭厙參雙發變敘臺葉號嘆嘰嚇嗎啟吳員喪喬單嘩響問啞啟啣國圍園圓圖團聖場壞塊堅壇壩墳墜壟壠壢壯聲壺壽夠夢夾奪奮奧婦媽媧嫵嬌嬰學孫寧寶實寵審寫將專尋對導屆屍層屬岡峴島峽崗崢嶄嶇嶺嶼巋巒幣帥師帳帶幀幫幹幾庫廁廂廄廈廚廝廟廠廣廢廳弒張強彈彌彎彙彥徑從復徵徹恆恥悅悞惡惱惲愛愜愨愷慘慚慟慣慪慫慮慳慶憂憊憐憑懇應懣懶懷懸懺懼懾戀戇戔戧戰戲戶拋挾捨捫掃掄掗掙掛採揀換揮損搖搗搶摑摜摟摯摳摶撈撏撐撓撥撫撲撻撾撿擁擄擇擊擋擔據擠擬擯擰擱擲擴擷擺擻擼擾攄攆攏攔攖攙攢敵斂數齋斕鬥斬斷於時晉晝暈暉暢暫曄曆曇曉曏曖曠曨會朧東柵桿梔梘條梟棄棖棗棟棧棲棶椏楊楓楨業極榪榮榿構槍槤槧槨槳樁樂樅樓標樞樣樸樹樺橈橋機橢橫檁檉檔檜檟檢檣檮檯檳檸檻櫃櫓櫚櫛櫞櫟櫥櫧櫨櫪櫫櫬櫻欄權欏欒欖歡歐殲殼毀毆氈氌氣氫氬氳汙決沈沖沒況洶浹涇涼淚淥淨淩淪淵淶淺渙減渦測渾湊湞湯溈準溝溫滄滅滌滎滬滯滲滷滸滾滿漁漚漢漣漬漲漵漸漿潁潑潔潛潤潯潰潷潿澀澆澇澗澤澠澩澮澱濁濃濟濤濫濰濱濺濾瀅瀆瀉瀋瀏瀕瀘瀝瀟瀠瀦瀧瀨瀰瀲瀾灃灄灑灘灝灣灤灧災為烏烴無煉煒煙煢煥煩煬熒熗熱熾燁燈燉燒燙燜營燦燭燴燶燼燾爍爐爛爭爺爾牆牘牽犖犢狀狹狽猙猛猶猻獁獄獅獎獨獪獫獰獲獵獸獺獻獼玀現琺琿瑋瑣瑤瑩瑪瑯璉璣璦環璽瓊瓏甌產畢畫異當疇疊痙痠瘂瘋瘍瘓瘞瘡瘧瘮瘲瘺瘻療癆癇癉癒癘癟癡癢癤癥癧癩癬癭癮癰癱癲發皚皺盞盡監盤盧盜盞眥眾睏睜睞瞘瞞瞭瞼矚矯硃硤硨硯碩碭確碼磚磣磧磯礎礙礦礪礫礬祿禍禎禕禡禦禪禮禰禱禿稈稅稜稟種稱穀穌積穎穠穡穢穩穫竄竅竇竊競筆筍筧箋箏節範築篋篤篩簀簍簞簡簣簫簷簽簾籃籌籙籜籟籠籤籩籪糝糞糧糲糴糶糾紀紂約紅紆紇紈紉紋納紐紓純紕紗紙級紛紜紡紮細紱紲紳紵紹紺紼絀終組絆絎結絕絛絝絞絡絢給絨絰統絲絳絹綁綃綆綈綉綌綏經綜綠綢綣綫綬維綱網綴綵綸綹綺綻綽綾綿緄緇緊緋緒緔緗緘緙線緝緞締緡緣緦編緩緬緯緱緲練緶緹緻縈縉縊縋縐縑縛縝縞縟縣縧縫縭縮縱縲縳縴縵縶縷縹總績繃繅繆繈繒織繕繚繞繡繢繩繪繫繭繮繯繰繳繹繼繽纈纏纓纖纜缽罈罌罰罵罷羅羆羈羋羥義習翹耬聖聞聯聰聲聳職聽聾肅脅脈脛脫脹腎腖腡腳腸膃膚膠膩膽膾膿臉臍臏臘臚臟臠臢臨臺與興舉艙艤艦艫艷藝節芻莊莖莢莧華萇萊萬萵葉葒著葦葷蒓蒔蒞蒼蓀蓋蓮蓯蓽蔔蔞蔣蔥蔦蔭蕁蕆蕎蕒蕕蕘蕢蕩榮蕭薈薊薌薔薘薟薦薩薺藍藎藝藥藪藶藹藺蘄蘆蘇蘊蘋蘚蘞蘢蘭蘺蘿處虛虜號虧蟲蛺蛻蜆蝕蝟蝦蝸蠅蠆蠍蠐蠑蠔蠟蠣蠱蠶蠻術衛衝袞裊裡補裝裡製複褲褳褸襖襝襠襤襪襯襲見規覓視覘覡覦親覬覯覲覷覺覽觀觴觶觸訂訃計訊訌討訐訓訕訖託記訛訝訟訣訥訪設許訴訶診註詁詆詎詐詒詔評詖詗詘詛詞詠詡詢詣試詩詫詬詭詮詰話該詳詵詼誄誅誆誇誌認誑誒誕誘誚語誠誡誣誤誥誦誨說誰課誶誹誼調諂諄談諉請諍諏諑諒論諗諛諜諞諦諧諫諭諮諱諳諶諷諸諺諾謀謁謂謄謅謊謎謐謔謗謙講謝謠謨謫謬謭謳謹謾譁證譎譏譖識譙譚譜譫譯議譴護譸譽讀變讎讒讓讖讜讞豈豎豐豬貓貝貞負財貢貧貨販貪貫責貯貰貴貶買貸費貼貽貿賀賁賂賃賄資賈賊賑賒賓賕賙賚賜賞賠賡賢賣賤賦質賬賭賴賺賻購賽贅贇贈贊贍贏贓贔贖贛趙趕趨趲跡踐踴蹌蹕蹣蹤蹺躂躉躊躋躍躑躒躓躕躚軀車軋軌軍軒軔軛軟軤軫軲軸軹軺軻軼載輊較輅輇輈輕輒輔輛輜輝輟輥輦輩輪輯輸輻輾輿轀轂轄轅轆轉轍轎轔轟轡轢轤辦辭辯農迴這連週進遊運過達違遙遜遞遠適遲遷選遺遼邁還邇邊邏邐郟郵鄆鄉鄒鄔鄖鄧鄭鄰鄲鄴鄶鄺酈醞醬醱釀釁釃釅釋釐鈀鈁鈃鈄鈈鈉鈔鈕鈞鈣鈥鈦鈧鈮鈰鈳鈴鈷鈸鈹鈺鈽鈾鈿鉀鐵"
)

TRAD_TO_SIMP = str.maketrans(
    {
        "歲": "岁",
        "隊": "队",
        "長": "长",
        "侶": "侣",
        "國": "国",
        "學": "学",
        "會": "会",
        "說": "说",
        "這": "这",
        "個": "个",
        "們": "们",
        "來": "来",
        "時": "时",
        "對": "对",
        "發": "发",
        "過": "过",
        "還": "还",
        "後": "后",
        "裡": "里",
        "為": "为",
        "與": "与",
        "開": "开",
        "關": "关",
        "點": "点",
        "種": "种",
        "體": "体",
        "頭": "头",
        "無": "无",
        "業": "业",
        "東": "东",
        "車": "车",
        "門": "门",
        "問": "问",
        "間": "间",
        "現": "现",
        "電": "电",
        "話": "话",
        "書": "书",
        "買": "买",
        "賣": "卖",
        "貓": "猫",
        "馬": "马",
        "鳥": "鸟",
        "魚": "鱼",
        "風": "风",
        "飛": "飞",
        "愛": "爱",
        "聽": "听",
        "讀": "读",
        "寫": "写",
        "線": "线",
        "網": "网",
        "義": "义",
        "氣": "气",
        "漢": "汉",
        "語": "语",
        "萬": "万",
    }
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def preview_path_for(path: Path, n: int = 50) -> Path:
    return path.parent / "previews" / f"{path.stem}.preview_{n}.jsonl"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", clean_text(text)).casefold()
    return re.sub(r"\s+", " ", text)


def looks_simplified_chinese(text: str) -> bool:
    return not any(ch in TRADITIONAL_CHARS for ch in text)


def to_simplified_light(text: str) -> str:
    return text.translate(TRAD_TO_SIMP)


def lang_name(lang: str) -> str:
    return LANG_EN_NAMES.get(lang, lang)


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl_with_preview(path: Path, records: Iterable[dict[str, Any]], preview_n: int = 50) -> int:
    ensure_dir(path.parent)
    preview_lines: list[str] = []
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            js = json.dumps(rec, ensure_ascii=False)
            f.write(js + "\n")
            if count < preview_n:
                preview_lines.append(js)
            count += 1
    prev = preview_path_for(path, preview_n)
    ensure_dir(prev.parent)
    prev.write_text("\n".join(preview_lines) + ("\n" if preview_lines else ""), encoding="utf-8")
    return count


def unique_keep_order(items: Iterable[str], limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = clean_text(item)
        if not item:
            continue
        key = normalize_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def dict_record(
    *,
    source: str,
    src_lang: str,
    tgt_lang: str,
    source_text: str,
    target_candidates: list[str],
    confidence: float,
    source_url: str,
    license_note: str,
    pivot_text: str | None = None,
) -> dict[str, Any]:
    candidates = unique_keep_order(target_candidates, limit=8)
    if not candidates:
        raise ValueError("empty target_candidates")
    rec: dict[str, Any] = {
        "source": source,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "source_text": clean_text(source_text),
        "target_text": candidates[0],
        "target_candidates": candidates,
        "confidence": confidence,
        "relation": "translation",
        "source_url": source_url,
        "license_note": license_note,
    }
    if pivot_text:
        rec["pivot_text"] = clean_text(pivot_text)
    return rec
