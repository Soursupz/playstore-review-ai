import re

SLANG_MAP = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "ngga": "tidak", "tdk": "tidak", "engga": "tidak", "enggak": "tidak",
    "kagak": "tidak", "kaga": "tidak", "ndak": "tidak", "nda": "tidak",
    "bgt": "banget", "bgtt": "banget", "bngt": "banget",
    "skl": "sekali",
    "apk": "aplikasi", "app": "aplikasi", "apps": "aplikasi",
    "dr": "dari", "drpd": "daripada", "dgn": "dengan", "dg": "dengan",
    "sm": "sama", "brsm": "bersama", "pd": "pada", "utk": "untuk",
    "tuk": "untuk", "buat": "untuk",
    "krn": "karena", "karna": "karena", "krna": "karena",
    "yg": "yang", "tp": "tapi", "tpi": "tapi", "ttp": "tetap",
    "ttpi": "tetapi", "ttg": "tentang",
    "sy": "saya", "gw": "saya", "gue": "saya",
    "km": "kamu", "lo": "kamu", "lu": "kamu",
    "udah": "sudah", "udh": "sudah", "dah": "sudah", "sdh": "sudah",
    "blm": "belum", "blum": "belum",
    "skrg": "sekarang", "skrng": "sekarang",
    "lg": "lagi", "lgi": "lagi",
    "trs": "terus", "trus": "terus",
    "msh": "masih", "masi": "masih",
    "hbs": "habis",
    "aja": "saja", "aj": "saja",
    "emg": "memang", "emang": "memang",
    "nih": "ini", "tuh": "itu",
    "bkn": "bukan",
    "klo": "kalau", "klu": "kalau", "kl": "kalau", "klau": "kalau",
    "gimana": "bagaimana", "gmn": "bagaimana",
    "knp": "kenapa",
    "lbh": "lebih",
    "kyk": "seperti", "kyak": "seperti", "kayak": "seperti",
    "jd": "jadi", "jdi": "jadi",
    "bs": "bisa", "bsa": "bisa",
    "hrs": "harus",
    "jg": "juga",
    "mau": "mau", "mo": "mau",
    "dpt": "dapat", "dpat": "dapat",
    "sdkt": "sedikit",
    "sampe": "sampai", "ampe": "sampai",
    "bentar": "sebentar",
    "pake": "pakai", "pk": "pakai",
    "nunggu": "menunggu",
    "nyari": "mencari",
    "tmn": "teman",
    "ok": "oke",
    "mantap": "bagus", "mantul": "bagus", "kece": "bagus",
    "jelek": "jelek", "buruk": "buruk", "parah": "parah",
    "rugi": "rugi", "kecewa": "kecewa", "puas": "puas",
}

EMOJI_MAP = {
    "\U0001F44D": " bagus ", "\U0001F44E": " jelek ", "\U0001F621": " kecewa ", "\U0001F62D": " sedih ",
    "\U0001F60A": " senang ", "\U0001F60D": " suka ", "❤️": " cinta ", "\U0001F31F": " mantap ",
    "\U0001F622": " sedih ", "\U0001F620": " marah ", "\U0001F612": " kesal ", "\U0001F618": " suka ",
    "\U0001F601": " senang ", "\U0001F44C": " oke ", "\U0001F44F": " bagus ", "\U0001F496": " cinta ",
    "\U0001F389": " senang ", "\U0001F610": " biasa ", "\U0001F914": " biasa ", "\U0001F937": " biasa ",
}


def replace_emojis(text: str) -> str:
    for emo, txt in EMOJI_MAP.items():
        text = text.replace(emo, txt)
    return text


def normalize_repeated_chars(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def replace_slang(text: str) -> str:
    tokens = text.split()
    return " ".join([SLANG_MAP.get(tok, tok) for tok in tokens])


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = replace_emojis(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^\w\s.,!?;:'\"()\-/%]", " ", text, flags=re.UNICODE)
    text = normalize_repeated_chars(text)
    text = replace_slang(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
