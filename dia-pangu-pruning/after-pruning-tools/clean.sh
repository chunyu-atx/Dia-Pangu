python - <<'PY'
import os
import pandas as pd

IN_CSV  = "/media/t1/sym/dia-pangu/results/dia-pangu_v0112_ft_c17_33.csv"
OUT_CSV = "/media/t1/sym/dia-pangu/results/dia-pangu_v0112_ft_c17_33_utf8.csv"

encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk"]
last_err = None
df = None

for enc in encodings:
    try:
        df = pd.read_csv(IN_CSV, encoding=enc)
        print(f"[OK] Read CSV with encoding: {enc}")
        break
    except UnicodeDecodeError as e:
        last_err = e

if df is None:
    print("[WARN] Common encodings failed, fallback to utf-8(errors=replace)")
    with open(IN_CSV, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8", errors="replace")
    from io import StringIO
    df = pd.read_csv(StringIO(text))
    print(f"[OK] Read CSV with fallback decode; bad bytes replaced with '�'")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"[DONE] Wrote UTF-8-SIG CSV to: {OUT_CSV}")
PY