"""Name and address normalisation — the single source of truth for preprocessing.

Notebook 00 runs these on the training data and notebook 05 runs the *same* functions
on the test data, so train/test features are computed on identically cleaned text.

Everything here is country-agnostic: the training data has US + India but the test set
adds France, so nothing is keyed on a country label. The only per-country knowledge is
lists of state/province names, which are treated as optional removable tokens and
simply never fire for an unseen country.
"""
from __future__ import annotations

import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from anyascii import anyascii  # ISC licence; transliterates Devanagari/Kannada/Telugu/accents

# --------------------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------------------
# Legal-form tokens -> canonical form. After punctuation removal "L.L.C." becomes
# "l l c", which _collapse_initials() turns into "llc" before this lookup.
LEGAL_MAP = {
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt", "pvtltd": "pvt ltd",
    "company": "co", "co": "co", "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc",
    "pc": "pc", "plc": "plc", "public": "public", "sa": "sa", "sas": "sas", "sasu": "sasu",
    "sarl": "sarl", "sci": "sci", "eurl": "eurl", "snc": "snc", "gmbh": "gmbh", "opc": "opc",
    "lc": "lc", "ltda": "ltda", "pa": "pa", "pty": "pty",
}
# Honorifics / filler tokens with no identity value.
NAME_STOP = {"the", "mr", "mrs", "ms", "smt", "m s", "messrs", "of", "and", "et", "le", "la", "les", "des", "du", "de"}
# Alias markers: "X fka Y", "X dba Y", "X aka Y", "formerly X".
ALIAS_RE = re.compile(r"\b(?:f[\s/.]?k[\s/.]?a|d[\s/.]?b[\s/.]?a|a[\s/.]?k[\s/.]?a|formerly(?: known as)?|trading as)\b\.?")
POSTCODE_RE = re.compile(r"\b(\d{5,6})[\s,]*$")
URL_TAIL_RE = re.compile(r"\|\s*(?:https?://)?(?:www\.)?\S+\s*$", re.I)
DOMAIN_RE = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|in|co\.in|co|fr|io|biz|us|info)\b", re.I)
ID_TAG_RE = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)", re.I)
NON_LATIN_RE = re.compile("[^\u0000-ɏḀ-ỿ]+")  # anything outside Latin blocks
PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
WS_RE = re.compile(r"\s+")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})

ADDR_ABBR = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "bvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle",
    "pl": "place", "pkwy": "parkway", "hwy": "highway", "sq": "square", "ter": "terrace", "trl": "trail",
    "cres": "crescent", "mt": "mount", "ft": "fort", "twp": "township", "hts": "heights", "pt": "point",
    "n": "north", "s": "south", "e": "east", "w": "west", "so": "south", "no": "no",
    "r": "rue", "ch": "chemin", "imp": "impasse", "fbg": "faubourg", "rte": "route", "all": "allee",
    "bldg": "building", "apts": "apartment", "appt": "apartment", "opp": "opposite", "nr": "near",
    "fl": "floor", "flr": "floor", "flt": "flat", "flta": "flat", "rly": "railway", "stn": "station",
    "extn": "extension", "ext": "extension", "sec": "sector", "mkt": "market", "cplx": "complex",
    "bombay": "mumbai", "gurgaon": "gurugram", "bangalore": "bengaluru", "calcutta": "kolkata",
    "madras": "chennai", "poona": "pune",
}
# Tokens that are pure formatting noise in addresses.
ADDR_STOP = {
    "no", "door", "dor", "doro", "h", "hno", "hn", "house", "number", "num", "null", "none",
    "na", "unit", "apt", "apartment", "suite", "ste", "floor", "flat", "the", "of", "near", "opposite",
    "de", "du", "la", "le", "des", "cdp", "city", "town", "village", "region", "dist", "district",
}
PO_BOX_RE = re.compile(r"\b(?:p\s?o\s?box|pmb|post box)\s*#?\s*\d+\b")

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh", "jh": "jharkhand",
    "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur",
    "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "orissa", "pb": "punjab",
    "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana", "ts": "telangana",
    "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand", "ut": "uttarakhand", "wb": "west bengal",
    "dl": "delhi", "jk": "jammu and kashmir", "ch": "chandigarh", "py": "puducherry", "la": "ladakh",
    "an": "andaman and nicobar islands", "dn": "dadra and nagar haveli", "ld": "lakshadweep",
}
# Every full name / code a whole comma-separated address component can equal.
STATE_TOKENS = set()
for _d in (US_STATES, IN_STATES):
    STATE_TOKENS |= set(_d) | set(_d.values())
STATE_TOKENS |= {"uttaranchal", "orissa", "pondicherry", "nct of delhi", "new delhi"} - {"new delhi"}
# "orissa"/"odisha", "telangana"/"andhra pradesh" etc. are kept distinct on purpose: the
# state is dropped from addr_clean and only exposed as a weak side feature.


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _ascii_lower(s: str) -> str:
    return anyascii(s).lower() if s else ""


def _collapse_initials(tokens: list[str]) -> list[str]:
    """['l','l','c'] -> ['llc'];  ['p','c'] -> ['pc'].  Runs of >=2 single letters merge."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0])
            run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    return out


def _dedupe_consecutive(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def _fix_leet(tok: str) -> str:
    """'n0rth' -> 'north', '5ons' -> 'sons'; leaves pure numbers and short codes alone."""
    if tok.isdigit() or tok.isalpha():
        return tok
    letters = sum(c.isalpha() for c in tok)
    return tok.translate(LEET) if letters >= 2 else tok


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------
def _name_tokens(s: str) -> list[str]:
    s = s.replace("&", " and ").replace("+", " plus ")
    s = PUNCT_RE.sub(" ", s)
    toks = [_fix_leet(t) for t in s.split()]
    toks = _collapse_initials(toks)
    return _dedupe_consecutive(toks)


def _core(tokens: list[str]) -> tuple[str, str]:
    legal, core = [], []
    for t in tokens:
        if t in LEGAL_MAP:
            legal.append(LEGAL_MAP[t])
        elif t not in NAME_STOP:
            core.append(t)
    return " ".join(core), " ".join(sorted(set(" ".join(legal).split())))


def clean_name(raw: str) -> dict:
    """Return the name-derived fields for one record.

    name_clean : full normalised name (legal forms canonicalised, kept)
    name_core  : identity-bearing tokens only (legal forms + honorifics removed)
    name_alt   : the *other* side of an alias marker (fka/dba/aka), '' if none
    legal      : sorted set of canonical legal-form tokens ("llc", "pvt ltd", ...)
    name_nonlatin : 1 if the raw name was mostly non-Latin script (transliterated)
    name_is_domain: 1 if the name was a bare web domain
    """
    raw = raw or ""
    nonlatin = int(len(NON_LATIN_RE.sub("", raw)) < 0.5 * len(raw)) if raw else 0
    s = ID_TAG_RE.sub(" ", raw)
    s = URL_TAIL_RE.sub(" ", s)
    s = _ascii_lower(s).strip()
    is_domain = 0
    m = DOMAIN_RE.match(s)
    if m and " " not in s.strip():
        s, is_domain = m.group(1).replace("-", " "), 1
    parts = ALIAS_RE.split(s)
    toks_full = _name_tokens(s if len(parts) == 1 else " ".join(parts))
    toks_full = [LEGAL_MAP.get(t, t) for t in toks_full]
    core, legal = _core(toks_full)
    alt = ""
    if len(parts) > 1:
        # "Halotavo F/K/A Roach Beverage Corp": both sides are plausible identities.
        # name_core keeps the LAST part (the historical / legal name), name_alt the first.
        c_last, _ = _core(_name_tokens(parts[-1]))
        c_first, _ = _core(_name_tokens(parts[0]))
        core, alt = (c_last or core), c_first
    return {
        "name_clean": " ".join(toks_full),
        "name_core": core,
        "name_alt": alt,
        "legal": legal,
        "name_nonlatin": nonlatin,
        "name_is_domain": is_domain,
    }


# --------------------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------------------
def clean_address(raw: str, country: str = "") -> dict:
    """Return address-derived fields for one record.

    addr_clean : normalised address tokens (state removed, abbreviations expanded,
                 leading zeros stripped, formatting noise like "Door No"/"#" removed)
    addr_nums  : space-separated numeric tokens in order of appearance
    state      : canonical state/province token if a whole component was one, else ''
    postcode   : 5-6 digit postal/PIN/ZIP code if present, else ''
    """
    raw = raw or ""
    s = NON_LATIN_RE.sub(" ", raw)  # native-script state names ("महाराष्ट्र") -> dropped
    s = _ascii_lower(s)
    state = ""
    comps = []
    for comp in s.split(","):
        c = WS_RE.sub(" ", PUNCT_RE.sub(" ", comp)).strip()
        if not c:
            continue
        if c in STATE_TOKENS and not state:
            # Codes are ambiguous across countries ("tn" = Tennessee / Tamil Nadu), so
            # resolve with the record's own country; unknown countries keep the raw token.
            table = {"us": US_STATES, "india": IN_STATES}.get(country, {})
            state = table.get(c, c)
            continue
        comps.append(comp)
    s = " , ".join(comps)
    s = PO_BOX_RE.sub(" ", s)
    # Postal code = a 5-6 digit number that ends the address (US ZIP / India PIN /
    # French CP). A 5-digit number elsewhere is usually a US house number.
    postcode = ""
    m = POSTCODE_RE.search(s)
    if m:
        postcode = m.group(1).lstrip("0") or "0"
    s = PUNCT_RE.sub(" ", s)
    toks = []
    for t in s.split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            t = ADDR_ABBR.get(t, t)
            # "2nd"/"3rd"/"7th" -> "2"/"3"/"7" so ordinal formats agree
            m2 = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", t)
            if m2:
                t = m2.group(1)
        if t in ADDR_STOP:
            continue
        toks.append(t)
    toks = _dedupe_consecutive(toks)
    nums = [t for t in toks if any(ch.isdigit() for ch in t)]
    return {
        "addr_clean": " ".join(toks),
        "addr_nums": " ".join(nums),
        "state": state,
        "postcode": postcode,
    }


# --------------------------------------------------------------------------------------
# Frame-level API
# --------------------------------------------------------------------------------------
CLEAN_COLUMNS = [
    "entity_id", "country", "name_clean", "name_core", "name_alt", "legal", "name_nonlatin",
    "name_is_domain", "addr_clean", "addr_nums", "state", "postcode", "name_len", "addr_len",
    "name_ntok", "addr_ntok", "name_phon", "addr_phon",
]


def _clean_block(args) -> pd.DataFrame:
    ids, countries, names, addrs = args
    n = pd.DataFrame([clean_name(x) for x in names])
    a = pd.DataFrame([clean_address(x, c) for x, c in zip(addrs, countries)])
    out = pd.concat([n, a], axis=1)
    out.insert(0, "country", countries)
    out.insert(0, "entity_id", ids)
    return out


def clean_frame(df: pd.DataFrame, n_jobs: int = 1, chunk: int = 500_000) -> pd.DataFrame:
    """Clean a raw source frame (entity_id, business_name, business_address, country)."""
    df = df.fillna("")
    # country: normalise case/whitespace only - it's an open set (France appears in test)
    countries = df["country"].astype(str).str.strip().str.lower().tolist()
    blocks = [
        (
            df["entity_id"].iloc[i : i + chunk].tolist(),
            countries[i : i + chunk],
            df["business_name"].iloc[i : i + chunk].astype(str).tolist(),
            df["business_address"].iloc[i : i + chunk].astype(str).tolist(),
        )
        for i in range(0, len(df), chunk)
    ]
    if n_jobs > 1 and len(blocks) > 1:
        with Pool(min(n_jobs, len(blocks))) as pool:
            parts = pool.map(_clean_block, blocks)
    else:
        parts = [_clean_block(b) for b in blocks]
    out = pd.concat(parts, ignore_index=True)
    out["name_len"] = out["name_core"].str.len().astype(np.int16)
    out["addr_len"] = out["addr_clean"].str.len().astype(np.int16)
    out["name_ntok"] = out["name_core"].str.count(" ").add(1).where(out["name_core"] != "", 0).astype(np.int8)
    out["addr_ntok"] = out["addr_clean"].str.count(" ").add(1).where(out["addr_clean"] != "", 0).astype(np.int8)
    for c in ("name_nonlatin", "name_is_domain"):
        out[c] = out[c].astype(np.int8)
    # phonetic skeletons (transliteration-robust keys used by retrieval and features)
    nm = out["name_core"].where(out["name_core"] != "", out["name_clean"])
    out["name_phon"] = [phonetic(x) for x in nm.tolist()]
    out["addr_phon"] = [phonetic(x) for x in out["addr_clean"].tolist()]
    return out[CLEAN_COLUMNS]


# --------------------------------------------------------------------------------------
# Phonetic skeleton (transliteration-robust key)
# --------------------------------------------------------------------------------------
# Many S2/S3 names were written in an Indic script and transliterated phonetically
# ("brait prodyusr praivet limited" for "Bright Producer Private Limited"). Character
# n-grams of the two spellings barely overlap, but their consonant skeletons do. Rules are
# applied identically to both sides, so they only need to map sound-alike spellings to the
# same key, not to be linguistically exact.
_PH_MULTI = [("ght", "t"), ("ph", "f"), ("sh", "s"), ("ch", "c"), ("th", "t"), ("dh", "d"),
             ("bh", "b"), ("kh", "k"), ("gh", "g"), ("ck", "k"), ("wh", "v")]
_PH_SOFT_C = re.compile(r"c(?=[eiy])")
_PH_SOFT_G = re.compile(r"g(?=[eiy])")
_PH_MAP = str.maketrans({"c": "k", "q": "k", "x": "k", "z": "s", "w": "v", "y": "i", "d": "t", "m": "n", "j": "j"})
_PH_VOWEL = re.compile(r"(?<!^)[aeiou]+")
_PH_REPEAT = re.compile(r"(.)\1+")


def phonetic_word(w: str) -> str:
    if not w or any(ch.isdigit() for ch in w):
        return w
    for a, b in _PH_MULTI:
        w = w.replace(a, b)
    w = _PH_SOFT_C.sub("s", w)
    w = _PH_SOFT_G.sub("j", w)
    w = w.translate(_PH_MAP)
    w = _PH_VOWEL.sub("", w)
    return _PH_REPEAT.sub(r"\1", w)


def phonetic(s: str) -> str:
    """Space-separated phonetic skeleton of an already-cleaned (ascii, lower-case) string."""
    return " ".join(phonetic_word(w) for w in s.split()) if s else ""
