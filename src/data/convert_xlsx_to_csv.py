"""
批量将 data/raw/ 下的所有 .xlsx 文件的工作表导出为 CSV。

输出目录: data/processed/{xlsx_name}/
每个工作表 → 一个 CSV 文件。

用法:
    python src/data/convert_xlsx_to_csv.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"


def sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符。"""
    # 移除或替换 Windows/Unix 文件名中不允许的字符
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip().strip(".")
    if not name:
        name = "unnamed"
    return name


def convert_xlsx_to_csv(xlsx_path: Path) -> dict[str, Path]:
    """将单个 xlsx 的所有工作表导出为 CSV。

    Returns:
        {sheet_name: output_csv_path}
    """
    stem = sanitize_filename(xlsx_path.stem)
    out_subdir = OUT_DIR / stem
    out_subdir.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(xlsx_path)
    sheet_names = xl.sheet_names
    results: dict[str, Path] = {}

    for sheet in sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        safe_sheet = sanitize_filename(sheet)
        csv_path = out_subdir / f"{safe_sheet}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        results[sheet] = csv_path
        print(f"  [{len(sheet_names)} sheets] Exported: {sheet} -> {csv_path.name}  ({df.shape[0]} rows x {df.shape[1]} cols)")

    return results


def main() -> None:
    xlsx_files = sorted(RAW_DIR.glob("*.xlsx"))
    if not xlsx_files:
        print(f"[ERROR] No .xlsx files found in {RAW_DIR}")
        return

    print(f"Found {len(xlsx_files)} Excel file(s), converting...\n")

    total_sheets = 0
    for xf in xlsx_files:
        print(f"[FILE] {xf.name}")
        try:
            out = convert_xlsx_to_csv(xf)
            total_sheets += len(out)
        except Exception as e:
            print(f"   [WARN] Failed: {e}")

    print(f"\n[DONE] {len(xlsx_files)} file(s) -> {total_sheets} sheet(s) -> {OUT_DIR}")


if __name__ == "__main__":
    main()
