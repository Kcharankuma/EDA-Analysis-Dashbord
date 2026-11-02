"""Read Excel/CSV files and turn them into a clean, privacy-safe orders table."""
import re
from pathlib import Path

import pandas as pd

ALIASES = {
    "price": ["purchase price", "price", "amount", "total", "order value", "order total", "revenue", "sales"],
    "provider": ["cc provider", "card provider", "payment method", "card type", "provider"],
    "am_pm": ["am or pm", "am/pm", "ampm"],
    "job": ["job", "occupation", "profession"],
    "language": ["language", "lang"],
    "email": ["email", "e-mail", "email address"],
    "address": ["address"],
    "browser": ["browser info", "user agent", "browser"],
    "date": ["order date", "purchase date", "date", "created at", "created_at", "timestamp", "datetime"],
}
EXCEL_EXT = {".xlsx", ".xlsm", ".xls"}


def norm(name) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def find_col(columns, key):
    lookup = {norm(c): c for c in columns}
    for alias in ALIASES[key]:
        if norm(alias) in lookup:
            return lookup[norm(alias)]
    return None


def is_excel(path) -> bool:
    return Path(path).suffix.lower() in EXCEL_EXT


def sheet_names(path) -> list[str]:
    return pd.ExcelFile(path).sheet_names if is_excel(path) else []


def read_any(path, sheet=None) -> pd.DataFrame:
    if is_excel(path):
        df = pd.read_excel(path, sheet_name=sheet if sheet else 0)
    else:
        df = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("Could not decode the CSV file.")
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all")


def derive_browser(ua: str) -> str:
    if ua.startswith("Opera"):
        return "Opera"
    if "MSIE" in ua or "Trident" in ua:
        return "Internet Explorer"
    for name in ("Firefox", "Chrome", "Safari"):
        if name in ua:
            return name
    return "Other"


def derive_os(ua: str) -> str:
    if any(k in ua for k in ("iPhone", "iPad", "iPod")):
        return "iOS"
    if "Windows" in ua:
        return "Windows"
    if "Macintosh" in ua or "Mac OS" in ua:
        return "macOS"
    if "Linux" in ua:
        return "Linux"
    return "Other"


def prepare(raw: pd.DataFrame, price_col: str, date_col: str = "None"):
    """Return (clean dataframe, number of rows dropped for invalid price).

    Sensitive columns (card number, security code, IP, email, address) are read
    only to derive safe features (email domain, state, customer id) and are
    never kept.
    """
    df = raw.reset_index(drop=True)
    price = df[price_col]
    if not pd.api.types.is_numeric_dtype(price):
        price = pd.to_numeric(price.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
    out = pd.DataFrame({"source_row": df.index + 2, "purchase_price": price.astype(float)})

    def text(key, default="Unknown"):
        col = find_col(df.columns, key)
        return df[col].fillna(default).astype(str).str.strip() if col else pd.Series(default, index=df.index)

    out["cc_provider"] = text("provider")
    out["job"] = text("job")
    out["language"] = text("language")

    dates = pd.to_datetime(df[date_col], errors="coerce") if date_col and date_col != "None" else None
    out["order_date"] = dates if dates is not None else pd.NaT

    am_col = find_col(df.columns, "am_pm")
    if am_col:
        out["am_pm"] = df[am_col].fillna("Unknown").astype(str).str.strip().str.upper()
    elif dates is not None:
        out["am_pm"] = dates.dt.hour.map(lambda h: "Unknown" if pd.isna(h) else ("AM" if h < 12 else "PM"))
    else:
        out["am_pm"] = "Unknown"

    email = find_col(df.columns, "email")
    if email:
        out["email_domain"] = df[email].astype(str).str.split("@").str[-1].str.lower()
        out["customer_id"] = pd.factorize(df[email])[0] + 1
    else:
        out["email_domain"], out["customer_id"] = "Unknown", pd.NA

    addr = find_col(df.columns, "address")
    out["state"] = (df[addr].astype(str).str.extract(r".*\b([A-Z]{2})\s+\d{5}")[0].fillna("Unknown")
                    if addr else "Unknown")

    ua = find_col(df.columns, "browser")
    uas = df[ua].fillna("").astype(str) if ua else None
    out["browser"] = uas.map(derive_browser) if uas is not None else "Unknown"
    out["os"] = uas.map(derive_os) if uas is not None else "Unknown"

    before = len(out)
    out = out[out["purchase_price"].notna()].reset_index(drop=True)
    return out, before - len(out)