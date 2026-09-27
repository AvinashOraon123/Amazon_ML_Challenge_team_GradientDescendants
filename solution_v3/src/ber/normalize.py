"""Country-agnostic text normalisation, vectorised with polars.

Adds columns:
    name_n   cleaned name (ascii, lowercase, punctuation removed, domains unwrapped)
    core_n   name_n without legal-form / filler tokens
    skel     consonant skeleton of core_n (bridges transliteration: 'constructions' ~ 'knstrksns')
    addr_n   cleaned address (abbreviations expanded, leading zeros dropped, state names -> codes)
"""
import numpy as np
import polars as pl
from anyascii import anyascii

LEGAL = {
    # generic / US
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "llp", "lp", "ltd",
    "limited", "plc", "pc", "pllc", "pa", "the", "and", "of", "dba",
    # India
    "pvt", "private", "pvtltd", "opc",
    # France
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "cie", "et", "de", "du", "des", "la", "le", "les",
}

ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "dr": "drive", "ln": "lane", "blvd": "boulevard", "bd": "boulevard", "ct": "court", "pl": "place",
    "cir": "circle", "hwy": "highway", "pkwy": "parkway", "ste": "suite", "apt": "apartment",
    "fl": "floor", "flr": "floor", "bldg": "building", "nr": "near", "opp": "opposite",
    "sq": "square", "ter": "terrace", "trl": "trail", "pt": "point", "mt": "mount", "ft": "fort",
    "no": "no", "num": "no", "hno": "house no", "h": "house",
    # French
    "r": "rue", "ch": "chemin", "chem": "chemin", "imp": "impasse", "rte": "route", "fbg": "faubourg",
    "all": "allee", "pce": "place", "qu": "quai", "crs": "cours", "sq": "square",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
}
# Note: a few two-letter codes collide across countries (e.g. 'tn'); harmless, the country is the same on both sides.
STATE_PHRASES = {f" {k} ": f" {v} " for k, v in {**US_STATES, **IN_STATES}.items()}

_NONASCII = r"[^\x00-\x7F]"


def translit(s: pl.Series) -> pl.Series:
    """anyascii only on rows that need it (about 10-15% of S2/S3), everything else untouched."""
    mask = s.str.contains(_NONASCII).to_numpy()
    if not mask.any():
        return s
    vals = s.to_numpy().astype(object)
    vals[mask] = [anyascii(v) for v in vals[mask]]
    return pl.Series(s.name, vals, dtype=pl.String)


def _squash(e):
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def _map_tokens(e, mapping):
    old, new = list(mapping), list(mapping.values())
    return e.str.split(" ").list.eval(pl.element().replace(old, new)).list.join(" ")


def _drop_tokens(e, tokens):
    return e.str.split(" ").list.eval(pl.element().filter(~pl.element().is_in(list(tokens)))).list.join(" ")


def name_expr(c):
    e = c.str.to_lowercase()
    # web domains -> bare label: 'www.indriyaclub.com' -> 'indriyaclub'
    e = e.str.replace_all(r"(https?://)?(www\.)?([a-z0-9\-]+)\.(com|net|org|co\.in|in|co|fr|biz|info|us)\b", " $3 ")
    e = e.str.replace_all(r"#\s*[0-9]+", " ")            # junk reference numbers: 'sppropertiescom #61545'
    e = e.str.replace_all("&", " and ").str.replace_all(r"[\'\.]", "")
    e = e.str.replace_all(r"[^a-z0-9]+", " ")
    return _squash(e)


def skeleton_expr(c):
    """Phonetic consonant skeleton; tolerant to transliteration, vowel choice, OCR digit swaps."""
    e = c
    for a, b in [("0", "o"), ("1", "l"), ("3", "e"), ("4", "a"), ("5", "s"), ("6", "g"), ("8", "b")]:
        # only digits glued to letters are OCR noise; standalone numbers stay
        e = e.str.replace_all(rf"([a-z]){a}|{a}([a-z])", f"${{1}}{b}${{2}}")
    for a, b in [("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"),
                 ("sh", "s"), ("ch", "c"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("z", "s"),
                 ("c", "k"), ("w", "v"), ("v", "b")]:
        e = e.str.replace_all(a, b, literal=True)
    e = e.str.replace_all(r"[aeiouyh]", "")
    for ch in "bdfgjklmnprstv":  # collapse doubled consonants (rust regex has no backreferences)
        e = e.str.replace_all(f"{ch}{ch}+", ch)
    return _squash(e)


def addr_expr(c):
    e = c.str.to_lowercase()
    e = e.str.replace_all(r"\b(null|none|nan|n/a)\b", " ")
    e = e.str.replace_all(r"[^a-z0-9/\-]+", " ")          # punctuation (incl. '#', '.', ',') -> space
    e = e.str.replace_all(r"(^|[^0-9])0+([0-9])", "${1}${2}")  # '0701' -> '701', '0083-231' -> '83-231'
    e = e.str.replace_all(r"[/\-]+", " ")
    e = _squash(e)
    e = (" " + e + " ").str.replace_many(list(STATE_PHRASES), list(STATE_PHRASES.values()))
    e = _map_tokens(_squash(e), ADDR_ABBR)
    return _squash(e)


LEGAL_FORMS = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "ltd": "ltd", "limited": "ltd", "plc": "plc", "pc": "pc",
    "pllc": "pllc", "pa": "pa", "pvt": "pvt", "private": "pvt", "opc": "opc", "sarl": "sarl", "sas": "sas",
    "sasu": "sasu", "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc", "cie": "co",
}
# words the noise process appends to names ('Wood Yield' -> 'Wood Yield Group'); stripped in the clean core
FILLER = {"group", "partners", "center", "centre", "holdings", "enterprises", "services", "solutions",
          "associates", "ventures", "international", "global", "www", "com"}
# place-name junk the noise process appends to addresses ('Darby CITY', 'Baltimore ICTY', 'Louisville CDP')
ADDR_JUNK = {"city", "icty", "ctiy", "cdp", "county", "hq", "region"}


def clean_addr_raw(c):
    """Remove record-side noise that never appears in Source 1: URLs / e-mails, PO boxes / PMBs."""
    e = c.str.to_lowercase()
    e = e.str.replace_all(r"\S*@\S+|(https?://)?www\.\S+|\S+\.(com|net|org|co\.in|in|fr|biz|info)\b", " ")
    e = e.str.replace_all(r"\b(p\.?\s*o\.?\s*box|pmb|post box)\s*#?\s*[0-9]+", " ")
    return e


def noise_markers(name, addr):
    """Record-level traces of the noise process on the RAW text. Noisy copies of real entities carry them
    more often than decoy records (e.g. stray accents: 83% matched vs 74% overall), so they act as a prior."""
    tok = name.str.to_lowercase().str.extract_all(r"[a-z]+")
    m = {
        "mk_dup": tok.list.len() > tok.list.unique().list.len(),
        "mk_accent": name.str.contains(r"[\u00c0-\u00ff]"),
        "mk_ocr": name.str.contains(r"[A-Za-z][0-9][A-Za-z]"),
        "mk_junk": name.str.contains(r"^\W"),
        "mk_brackets": name.str.contains(r"[\[\(]"),
        "mk_script": name.str.contains(r"[\u0900-\u0dff]"),
        "mk_domain": name.str.contains(r"\.com|www|\.in\b|\.net"),
        "mk_upper_name": name == name.str.to_uppercase(),
        "mk_zeropad": addr.str.contains(r"\b0+[1-9][0-9]*"),
        "mk_hash": addr.str.contains(r"##|#[0-9]"),
        "mk_pobox": addr.str.to_lowercase().str.contains(r"po box|pmb"),
        "mk_upper_addr": (addr == addr.str.to_uppercase()) & (addr != ""),
    }
    return [e.cast(pl.Float32).alias(k) for k, e in m.items()]


MARKERS = ("mk_dup", "mk_accent", "mk_ocr", "mk_junk", "mk_brackets", "mk_script", "mk_domain", "mk_upper_name",
           "mk_zeropad", "mk_hash", "mk_pobox", "mk_upper_addr")


def normalize(recs: pl.DataFrame) -> pl.DataFrame:
    recs = recs.with_columns(noise_markers(pl.col("name"), pl.col("addr")))
    recs = recs.with_columns(translit(recs["name"]).alias("_n"), translit(recs["addr"]).alias("_a"))
    recs = recs.with_columns(name_expr(pl.col("_n")).alias("name_n"), addr_expr(pl.col("_a")).alias("addr_n"))
    recs = recs.with_columns(_squash(_drop_tokens(pl.col("name_n"), LEGAL)).alias("core_n"))
    # if a name is nothing but legal words keep the full name as core
    recs = recs.with_columns(
        pl.when(pl.col("core_n") == "").then(pl.col("name_n")).otherwise(pl.col("core_n")).alias("core_n")
    )
    recs = recs.with_columns(skeleton_expr(pl.col("core_n")).alias("skel"))
    # cleaned variants used only by the matcher's comparison features (encoder inputs stay unchanged)
    recs = recs.with_columns(
        _squash(_drop_tokens(addr_expr(clean_addr_raw(pl.col("_a"))), ADDR_JUNK)).alias("addr_c"),
        _squash(_drop_tokens(pl.col("core_n").str.replace_all(r"\b[0-9]{5,}\b", " "), FILLER)).alias("core_c"),
        pl.col("name_n").str.split(" ").list.eval(
            pl.element().filter(pl.element().is_in(list(LEGAL_FORMS)))
            .replace(list(LEGAL_FORMS), list(LEGAL_FORMS.values()))
        ).list.unique().list.sort().list.join(" ").alias("legal"),
    )
    recs = recs.with_columns(
        pl.when(pl.col("core_c") == "").then(pl.col("core_n")).otherwise(pl.col("core_c")).alias("core_c"))
    return recs.drop("_n", "_a")


def to_bytes(s: pl.Series, length: int, chunk: int = 1_000_000) -> np.ndarray:
    """Fixed-width uint8 matrix (N, length); 0 = padding. Input must be ascii (post-normalisation)."""
    out = np.zeros((len(s), length), dtype=np.uint8)
    col = np.arange(length, dtype=np.int64)
    for a in range(0, len(s), chunk):
        part = s[a:a + chunk].str.slice(0, length)
        lens = part.str.len_bytes().to_numpy().astype(np.int64)
        buf = np.frombuffer("".join(part.to_list()).encode("ascii", "replace"), dtype=np.uint8)
        starts = np.cumsum(lens) - lens
        valid = col[None, :] < lens[:, None]
        out[a:a + chunk][valid] = buf[(starts[:, None] + col[None, :])[valid]]
    return out
