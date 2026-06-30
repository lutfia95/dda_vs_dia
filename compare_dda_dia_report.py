#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FRACTION_RE = re.compile(r"Fraction_([^./\\]+)", re.IGNORECASE)


def clean_sample_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("\\", "/").split("/")[-1]
    text = re.sub(r"^interact-", "", text)
    text = re.sub(r"\.(raw|mzml|mzxml|pep\.xml)$", "", text, flags=re.IGNORECASE)
    return text


def extract_fraction(value: object) -> str:
    text = clean_sample_name(value)
    match = FRACTION_RE.search(text)
    if match:
        return match.group(1)
    return text


def fraction_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    match = re.search(r"\d+", text)
    if match:
        return (int(match.group(0)), text)
    return (10**9, text)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def log2_positive(series: pd.Series) -> pd.Series:
    values = safe_numeric(series)
    return np.log2(values.where(values > 0))


def key_from_columns(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = []
    for col in cols:
        parts.append(df[col].fillna("").astype(str))
    out = parts[0]
    for part in parts[1:]:
        out = out + "|" + part
    return out


def normalize_fixed_carbamidomethyl(modified_sequence: pd.Series) -> pd.Series:
    """Remove common fixed Cys carbamidomethyl annotation for a fairer mod key."""
    return (
        modified_sequence.fillna("")
        .astype(str)
        .str.replace(r"C\[(57\.0215|57\.02146|57)\]", "C", regex=True)
    )


def fmt_int(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def fmt_float(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def jaccard(left: set[str], right: set[str]) -> float:
    union = len(left | right)
    if union == 0:
        return float("nan")
    return len(left & right) / union


def html_table(df: pd.DataFrame, max_rows: int | None = None, classes: str = "") -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    out = [f'<table class="{html.escape(classes)}">']
    out.append("<thead><tr>")
    for col in df.columns:
        out.append(f"<th>{html.escape(str(col))}</th>")
    out.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        out.append("<tr>")
        for val in row:
            if isinstance(val, float):
                text = fmt_float(val, 3)
            else:
                text = "" if pd.isna(val) else str(val)
            out.append(f"<td>{html.escape(text)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def card(title: str, value: str, subtitle: str = "") -> str:
    return (
        '<div class="card">'
        f'<div class="card-title">{html.escape(title)}</div>'
        f'<div class="card-value">{html.escape(value)}</div>'
        f'<div class="card-subtitle">{html.escape(subtitle)}</div>'
        "</div>"
    )


def grouped_bar_svg(
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    width: int = 1180,
    height: int = 360,
) -> str:
    if not labels or not series:
        return '<div class="empty">No data available.</div>'
    margin = dict(left=68, right=18, top=26, bottom=86)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_y = max(max(vals) if vals else 0 for _, vals, _ in series)
    max_y = max_y * 1.08 if max_y > 0 else 1
    group_w = plot_w / len(labels)
    bar_w = min(18, group_w / (len(series) + 1.2))
    y_ticks = 5
    items = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Grouped bar chart">'
    ]
    items.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    for i in range(y_ticks + 1):
        y_val = max_y * i / y_ticks
        y = margin["top"] + plot_h - (y_val / max_y) * plot_h
        items.append(
            f'<line x1="{margin["left"]}" x2="{width - margin["right"]}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        items.append(
            f'<text x="{margin["left"] - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#475569">{fmt_int(y_val)}</text>'
        )
    for i, label in enumerate(labels):
        cx = margin["left"] + i * group_w + group_w / 2
        for j, (_, values, color) in enumerate(series):
            value = values[i] if i < len(values) and not pd.isna(values[i]) else 0
            h = (value / max_y) * plot_h
            x = cx - (len(series) * bar_w) / 2 + j * bar_w
            y = margin["top"] + plot_h - h
            items.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 1:.1f}" height="{h:.1f}" fill="{color}"/>'
            )
        if len(labels) <= 40 or i % 2 == 0:
            items.append(
                f'<text x="{cx:.1f}" y="{height - 48}" text-anchor="end" transform="rotate(-45 {cx:.1f} {height - 48})" font-size="10" fill="#334155">{html.escape(label)}</text>'
            )
    legend_x = margin["left"]
    for name, _, color in series:
        items.append(f'<rect x="{legend_x}" y="8" width="11" height="11" fill="{color}"/>')
        items.append(
            f'<text x="{legend_x + 16}" y="18" font-size="12" fill="#0f172a">{html.escape(name)}</text>'
        )
        legend_x += 130
    items.append("</svg>")
    return "\n".join(items)


def line_svg(
    labels: list[str],
    values: list[float],
    width: int = 1180,
    height: int = 280,
) -> str:
    clean = [float(v) if not pd.isna(v) else float("nan") for v in values]
    if not labels or all(math.isnan(v) for v in clean):
        return '<div class="empty">No data available.</div>'
    margin = dict(left=58, right=20, top=20, bottom=72)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    y_max = max(1.0, max(v for v in clean if not math.isnan(v)) * 1.05)
    group_w = plot_w / max(len(labels) - 1, 1)
    points = []
    items = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Line chart">']
    items.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    for tick in np.linspace(0, y_max, 5):
        y = margin["top"] + plot_h - (tick / y_max) * plot_h
        items.append(
            f'<line x1="{margin["left"]}" x2="{width - margin["right"]}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        items.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#475569">{tick:.2f}</text>'
        )
    for i, val in enumerate(clean):
        if math.isnan(val):
            continue
        x = margin["left"] + i * group_w
        y = margin["top"] + plot_h - (val / y_max) * plot_h
        points.append((x, y))
    if len(points) > 1:
        point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        items.append(f'<polyline points="{point_string}" fill="none" stroke="#2563eb" stroke-width="2"/>')
    for x, y in points:
        items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#2563eb"/>')
    for i, label in enumerate(labels):
        if len(labels) <= 40 or i % 2 == 0:
            x = margin["left"] + i * group_w
            items.append(
                f'<text x="{x:.1f}" y="{height - 38}" text-anchor="end" transform="rotate(-45 {x:.1f} {height - 38})" font-size="10" fill="#334155">{html.escape(label)}</text>'
            )
    items.append("</svg>")
    return "\n".join(items)


def scatter_svg(
    x_values: Iterable[float],
    y_values: Iterable[float],
    width: int = 560,
    height: int = 420,
    max_points: int = 4500,
) -> str:
    xy = pd.DataFrame({"x": list(x_values), "y": list(y_values)}).replace([np.inf, -np.inf], np.nan).dropna()
    if xy.empty:
        return '<div class="empty">No matched positive intensities available.</div>'
    if len(xy) > max_points:
        xy = xy.sample(max_points, random_state=7)
    margin = dict(left=58, right=18, top=18, bottom=48)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    x_min, x_max = float(xy["x"].min()), float(xy["x"].max())
    y_min, y_max = float(xy["y"].min()), float(xy["y"].max())
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    items = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Intensity scatter plot">']
    items.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    for tick in np.linspace(x_min, x_max, 5):
        x = margin["left"] + (tick - x_min) / (x_max - x_min) * plot_w
        items.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{margin["top"]}" y2="{height - margin["bottom"]}" stroke="#eef2f7"/>')
        items.append(f'<text x="{x:.1f}" y="{height - 24}" text-anchor="middle" font-size="10" fill="#475569">{tick:.1f}</text>')
    for tick in np.linspace(y_min, y_max, 5):
        y = margin["top"] + plot_h - (tick - y_min) / (y_max - y_min) * plot_h
        items.append(f'<line x1="{margin["left"]}" x2="{width - margin["right"]}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eef2f7"/>')
        items.append(f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" fill="#475569">{tick:.1f}</text>')
    for row in xy.itertuples(index=False):
        x = margin["left"] + (row.x - x_min) / (x_max - x_min) * plot_w
        y = margin["top"] + plot_h - (row.y - y_min) / (y_max - y_min) * plot_h
        items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="1.7" fill="#0f766e" opacity="0.32"/>')
    items.append(f'<text x="{width / 2:.1f}" y="{height - 6}" text-anchor="middle" font-size="12" fill="#0f172a">DIA log2 peptide quantity</text>')
    items.append(f'<text x="14" y="{height / 2:.1f}" text-anchor="middle" font-size="12" fill="#0f172a" transform="rotate(-90 14 {height / 2:.1f})">DDA log2 PSM intensity</text>')
    items.append("</svg>")
    return "\n".join(items)


def venn_svg(
    title: str,
    dia_total: int,
    dda_total: int,
    overlap: int,
    width: int = 520,
    height: int = 310,
) -> str:
    dia_only = max(dia_total - overlap, 0)
    dda_only = max(dda_total - overlap, 0)
    union = dia_only + overlap + dda_only
    jacc = overlap / union if union else float("nan")
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)} Venn diagram">
  <rect width="{width}" height="{height}" fill="white"/>
  <text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="16" font-weight="650" fill="#0f172a">{html.escape(title)}</text>
  <circle cx="210" cy="145" r="92" fill="#2563eb" opacity="0.30" stroke="#2563eb" stroke-width="2"/>
  <circle cx="310" cy="145" r="92" fill="#0f766e" opacity="0.30" stroke="#0f766e" stroke-width="2"/>
  <text x="155" y="92" text-anchor="middle" font-size="13" font-weight="650" fill="#1d4ed8">DIA</text>
  <text x="365" y="92" text-anchor="middle" font-size="13" font-weight="650" fill="#0f766e">DDA</text>
  <text x="170" y="150" text-anchor="middle" font-size="22" font-weight="700" fill="#0f172a">{fmt_int(dia_only)}</text>
  <text x="260" y="150" text-anchor="middle" font-size="22" font-weight="700" fill="#0f172a">{fmt_int(overlap)}</text>
  <text x="350" y="150" text-anchor="middle" font-size="22" font-weight="700" fill="#0f172a">{fmt_int(dda_only)}</text>
  <text x="170" y="172" text-anchor="middle" font-size="12" fill="#334155">DIA-only</text>
  <text x="260" y="172" text-anchor="middle" font-size="12" fill="#334155">shared</text>
  <text x="350" y="172" text-anchor="middle" font-size="12" fill="#334155">DDA-only</text>
  <text x="{width / 2:.1f}" y="258" text-anchor="middle" font-size="12" fill="#475569">DIA total {fmt_int(dia_total)} | DDA total {fmt_int(dda_total)} | Jaccard {fmt_float(jacc, 3)}</text>
</svg>
"""


def read_diann(path: Path, q_threshold: float) -> pd.DataFrame:
    cols = [
        "Run",
        "Precursor.Id",
        "Modified.Sequence",
        "Stripped.Sequence",
        "Precursor.Charge",
        "Protein.Group",
        "Protein.Ids",
        "Precursor.Quantity",
        "Q.Value",
        "PG.Q.Value",
        "Decoy",
    ]
    df = pd.read_parquet(path, columns=cols)
    df["Q.Value"] = safe_numeric(df["Q.Value"])
    df["PG.Q.Value"] = safe_numeric(df["PG.Q.Value"])
    df["Decoy"] = safe_numeric(df["Decoy"]).fillna(0)
    df = df[(df["Decoy"] == 0) & (df["Q.Value"] <= q_threshold)].copy()
    df["fraction"] = df["Run"].map(extract_fraction)
    df["Precursor.Quantity"] = safe_numeric(df["Precursor.Quantity"])
    df["Precursor.Charge"] = df["Precursor.Charge"].astype("Int64").astype(str)
    df["precursor_key"] = key_from_columns(df, ["Modified.Sequence", "Precursor.Charge"])
    df["stripped_charge_key"] = key_from_columns(df, ["Stripped.Sequence", "Precursor.Charge"])
    return df


def read_dda_psm(path: Path, min_probability: float) -> pd.DataFrame:
    cols = [
        "Spectrum",
        "Spectrum File",
        "Peptide",
        "Modified Peptide",
        "Charge",
        "Probability",
        "Intensity",
        "Protein",
    ]
    df = pd.read_csv(path, sep="\t", usecols=cols)
    df["Probability"] = safe_numeric(df["Probability"])
    df = df[df["Probability"] >= min_probability].copy()
    df["sample"] = df["Spectrum File"].map(clean_sample_name)
    df["fraction"] = df["Spectrum File"].map(extract_fraction)
    df["Intensity"] = safe_numeric(df["Intensity"]).fillna(0)
    df["Charge"] = df["Charge"].astype("Int64").astype(str)
    df["modified_for_key"] = df["Modified Peptide"].fillna(df["Peptide"]).astype(str)
    df["precursor_key"] = key_from_columns(df, ["modified_for_key", "Charge"])
    df["stripped_charge_key"] = key_from_columns(df, ["Peptide", "Charge"])
    return df


def read_dda_global(ion_path: Path, peptide_path: Path, protein_path: Path) -> dict[str, pd.DataFrame]:
    ion = pd.read_csv(
        ion_path,
        sep="\t",
        usecols=["Peptide Sequence", "Modified Sequence", "Charge", "Intensity", "Protein"],
    )
    ion["Charge"] = ion["Charge"].astype("Int64").astype(str)
    ion["Intensity"] = safe_numeric(ion["Intensity"]).fillna(0)
    ion["modified_no_fixed_c"] = normalize_fixed_carbamidomethyl(ion["Modified Sequence"])
    ion["precursor_key"] = key_from_columns(ion, ["modified_no_fixed_c", "Charge"])
    ion["stripped_charge_key"] = key_from_columns(ion, ["Peptide Sequence", "Charge"])

    peptide = pd.read_csv(peptide_path, sep="\t", usecols=["Peptide", "Intensity", "Spectral Count"])
    peptide["Intensity"] = safe_numeric(peptide["Intensity"]).fillna(0)

    protein = pd.read_csv(protein_path, sep="\t", usecols=["Protein", "Total Intensity", "Unique Peptides"])
    protein["Total Intensity"] = safe_numeric(protein["Total Intensity"]).fillna(0)
    return {"ion": ion, "peptide": peptide, "protein": protein}


def per_fraction_summary(diann: pd.DataFrame, dda_psm: pd.DataFrame) -> pd.DataFrame:
    fractions = sorted(set(diann["fraction"]) | set(dda_psm["fraction"]), key=fraction_sort_key)
    rows = []
    for fraction in fractions:
        dia_f = diann[diann["fraction"] == fraction]
        dda_f = dda_psm[dda_psm["fraction"] == fraction]
        dia_pep = set(dia_f["Stripped.Sequence"].dropna().astype(str))
        dda_pep = set(dda_f["Peptide"].dropna().astype(str))
        overlap = dia_pep & dda_pep
        rows.append(
            {
                "fraction": fraction,
                "DIA rows": len(dia_f),
                "DDA PSMs": len(dda_f),
                "DIA stripped peptides": dia_f["Stripped.Sequence"].nunique(),
                "DDA stripped peptides": dda_f["Peptide"].nunique(),
                "DIA precursors": dia_f["precursor_key"].nunique(),
                "DDA precursors": dda_f["precursor_key"].nunique(),
                "DIA protein groups": dia_f["Protein.Group"].nunique(),
                "DDA proteins": dda_f["Protein"].nunique(),
                "stripped peptide overlap": len(overlap),
                "DIA-only stripped": len(dia_pep - dda_pep),
                "DDA-only stripped": len(dda_pep - dia_pep),
                "stripped Jaccard": jaccard(dia_pep, dda_pep),
            }
        )
    return pd.DataFrame(rows)


def global_overlap_table(diann: pd.DataFrame, dda_global: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ion = dda_global["ion"]
    protein = dda_global["protein"]
    levels = [
        (
            "Stripped peptide",
            set(diann["Stripped.Sequence"].dropna().astype(str)),
            set(ion["Peptide Sequence"].dropna().astype(str)),
        ),
        (
            "Stripped peptide + charge",
            set(diann["stripped_charge_key"].dropna().astype(str)),
            set(ion["stripped_charge_key"].dropna().astype(str)),
        ),
        (
            "Modified peptide",
            set(diann["Modified.Sequence"].dropna().astype(str)),
            set(ion["modified_no_fixed_c"].dropna().astype(str)),
        ),
        (
            "Modified peptide + charge",
            set(diann["precursor_key"].dropna().astype(str)),
            set(ion["precursor_key"].dropna().astype(str)),
        ),
        (
            "Protein/group",
            set(diann["Protein.Group"].dropna().astype(str)),
            set(protein["Protein"].dropna().astype(str)),
        ),
    ]
    rows = []
    for level, dia_set, dda_set in levels:
        rows.append(
            {
                "level": level,
                "DIA": len(dia_set),
                "DDA": len(dda_set),
                "overlap": len(dia_set & dda_set),
                "DIA-only": len(dia_set - dda_set),
                "DDA-only": len(dda_set - dia_set),
                "union": len(dia_set | dda_set),
                "Jaccard": jaccard(dia_set, dda_set),
            }
        )
    return pd.DataFrame(rows)


def intensity_by_fraction(diann: pd.DataFrame, dda_psm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dia_pep = (
        diann.groupby(["fraction", "Stripped.Sequence"], as_index=False)["Precursor.Quantity"]
        .sum()
        .rename(columns={"Stripped.Sequence": "peptide", "Precursor.Quantity": "dia_intensity"})
    )
    dda_positive = dda_psm[dda_psm["Intensity"] > 0].copy()
    dda_pep = (
        dda_positive.groupby(["fraction", "Peptide"], as_index=False)["Intensity"]
        .sum()
        .rename(columns={"Peptide": "peptide", "Intensity": "dda_intensity"})
    )
    matched = dia_pep.merge(dda_pep, on=["fraction", "peptide"], how="inner")
    matched["dia_log2"] = log2_positive(matched["dia_intensity"])
    matched["dda_log2"] = log2_positive(matched["dda_intensity"])
    matched = matched.dropna(subset=["dia_log2", "dda_log2"])
    rows = []
    for fraction, group in matched.groupby("fraction", sort=False):
        if len(group) >= 3:
            pearson = group["dia_log2"].corr(group["dda_log2"], method="pearson")
            spearman = group["dia_log2"].corr(group["dda_log2"], method="spearman")
        else:
            pearson = np.nan
            spearman = np.nan
        rows.append(
            {
                "fraction": fraction,
                "matched positive peptides": len(group),
                "Pearson log2 intensity": pearson,
                "Spearman log2 intensity": spearman,
                "median DIA log2": group["dia_log2"].median(),
                "median DDA log2": group["dda_log2"].median(),
            }
        )
    corr = pd.DataFrame(rows).sort_values("fraction", key=lambda s: s.map(fraction_sort_key))
    return matched, corr


def global_intensity_match(diann: pd.DataFrame, dda_global: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dia = (
        diann.groupby("Stripped.Sequence", as_index=False)["Precursor.Quantity"]
        .sum()
        .rename(columns={"Stripped.Sequence": "peptide", "Precursor.Quantity": "dia_intensity"})
    )
    dda = dda_global["peptide"][["Peptide", "Intensity"]].rename(
        columns={"Peptide": "peptide", "Intensity": "dda_intensity"}
    )
    merged = dia.merge(dda, on="peptide", how="inner")
    merged = merged[(merged["dia_intensity"] > 0) & (merged["dda_intensity"] > 0)].copy()
    merged["dia_log2"] = log2_positive(merged["dia_intensity"])
    merged["dda_log2"] = log2_positive(merged["dda_intensity"])
    return merged.dropna(subset=["dia_log2", "dda_log2"])


def detection_frequency(diann: pd.DataFrame, dda_psm: pd.DataFrame) -> pd.DataFrame:
    dia = diann[["fraction", "Stripped.Sequence"]].drop_duplicates()
    dia_counts = dia.groupby("Stripped.Sequence")["fraction"].nunique().rename("DIA fractions")
    dda = dda_psm[["fraction", "Peptide"]].drop_duplicates()
    dda_counts = dda.groupby("Peptide")["fraction"].nunique().rename("DDA fractions")
    out = pd.concat([dia_counts, dda_counts], axis=1).fillna(0).astype(int)
    out.index.name = "peptide"
    out["method"] = np.select(
        [
            (out["DIA fractions"] > 0) & (out["DDA fractions"] > 0),
            out["DIA fractions"] > 0,
            out["DDA fractions"] > 0,
        ],
        ["both", "DIA-only", "DDA-only"],
        default="none",
    )
    return out.reset_index()


def write_report(
    output: Path,
    diann: pd.DataFrame,
    dda_psm: pd.DataFrame,
    dda_global: dict[str, pd.DataFrame],
    per_fraction: pd.DataFrame,
    overlap: pd.DataFrame,
    matched_fraction: pd.DataFrame,
    corr_fraction: pd.DataFrame,
    matched_global: pd.DataFrame,
    detection: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    fractions = per_fraction["fraction"].astype(str).tolist()
    dia_peptides = per_fraction["DIA stripped peptides"].astype(float).tolist()
    dda_peptides = per_fraction["DDA stripped peptides"].astype(float).tolist()
    jaccards = per_fraction["stripped Jaccard"].astype(float).tolist()

    stripped_row = overlap[overlap["level"] == "Stripped peptide"].iloc[0]
    precursor_row = overlap[overlap["level"] == "Stripped peptide + charge"].iloc[0]
    total_unique_peptides = int(stripped_row["union"])
    global_pearson = matched_global["dia_log2"].corr(matched_global["dda_log2"]) if len(matched_global) >= 3 else np.nan
    global_spearman = (
        matched_global["dia_log2"].corr(matched_global["dda_log2"], method="spearman")
        if len(matched_global) >= 3
        else np.nan
    )

    detection_summary = (
        detection.groupby("method")
        .size()
        .reindex(["both", "DIA-only", "DDA-only"], fill_value=0)
        .reset_index(name="peptides")
    )

    top_dia_only = (
        detection[detection["method"] == "DIA-only"]
        .sort_values(["DIA fractions", "peptide"], ascending=[False, True])
        .head(25)
    )
    top_dda_only = (
        detection[detection["method"] == "DDA-only"]
        .sort_values(["DDA fractions", "peptide"], ascending=[False, True])
        .head(25)
    )

    cards = "\n".join(
        [
            card("DIA-NN runs", fmt_int(diann["fraction"].nunique()), f'{fmt_int(len(diann))} report rows after Q <= {args.diann_q}'),
            card("DDA fractions", fmt_int(dda_psm["fraction"].nunique()), f"{fmt_int(len(dda_psm))} PSM rows"),
            card("Stripped peptide overlap", fmt_int(stripped_row["overlap"]), f"Jaccard {fmt_float(stripped_row['Jaccard'], 3)}"),
            card("Total unique peptides", fmt_int(total_unique_peptides), "DIA or DDA stripped sequence"),
            card("Precursor overlap", fmt_int(precursor_row["overlap"]), "stripped sequence + charge"),
            card("Matched intensity peptides", fmt_int(len(matched_global)), f"Pearson {fmt_float(global_pearson, 3)}; Spearman {fmt_float(global_spearman, 3)}"),
        ]
    )

    def venn_for(level: str, title: str) -> str:
        row = overlap[overlap["level"] == level].iloc[0]
        return venn_svg(
            title,
            int(row["DIA"]),
            int(row["DDA"]),
            int(row["overlap"]),
        )

    venns = "\n".join(
        [
            venn_for("Stripped peptide + charge", "Stripped peptide + charge"),
            venn_for("Protein/group", "Protein/groups"),
        ]
    )

    unique_peptide_venn = venn_for("Stripped peptide", "Total unique stripped peptides")

    unique_per_run = per_fraction[
        [
            "fraction",
            "DIA stripped peptides",
            "DDA stripped peptides",
            "stripped peptide overlap",
            "DIA-only stripped",
            "DDA-only stripped",
        ]
    ].copy()
    unique_per_run["total unique peptides"] = (
        per_fraction["DIA stripped peptides"]
        + per_fraction["DDA stripped peptides"]
        - per_fraction["stripped peptide overlap"]
    )
    unique_per_run = unique_per_run.rename(
        columns={
            "fraction": "run/fraction",
            "DIA stripped peptides": "DIA unique peptides",
            "DDA stripped peptides": "DDA unique peptides",
            "stripped peptide overlap": "shared unique peptides",
            "DIA-only stripped": "DIA-only unique peptides",
            "DDA-only stripped": "DDA-only unique peptides",
        }
    )
    unique_per_run_display = unique_per_run.copy()
    for col in unique_per_run_display.columns:
        if col != "run/fraction":
            unique_per_run_display[col] = unique_per_run_display[col].map(fmt_int)

    per_fraction_display = per_fraction.copy()
    for col in per_fraction_display.columns:
        if col != "fraction" and col != "stripped Jaccard":
            per_fraction_display[col] = per_fraction_display[col].map(fmt_int)
    per_fraction_display["stripped Jaccard"] = per_fraction_display["stripped Jaccard"].map(lambda x: fmt_float(x, 3))

    overlap_display = overlap.copy()
    for col in ["DIA", "DDA", "overlap", "DIA-only", "DDA-only", "union"]:
        overlap_display[col] = overlap_display[col].map(fmt_int)
    overlap_display["Jaccard"] = overlap_display["Jaccard"].map(lambda x: fmt_float(x, 3))

    corr_display = corr_fraction.copy()
    if not corr_display.empty:
        corr_display["matched positive peptides"] = corr_display["matched positive peptides"].map(fmt_int)
        for col in ["Pearson log2 intensity", "Spearman log2 intensity", "median DIA log2", "median DDA log2"]:
            corr_display[col] = corr_display[col].map(lambda x: fmt_float(x, 3))

    css = """
    :root { color-scheme: light; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #0f172a;
      background: #f8fafc;
      line-height: 1.45;
    }
    header {
      background: #0f172a;
      color: white;
      padding: 28px 34px 24px;
    }
    main { padding: 26px 34px 42px; max-width: 1280px; margin: 0 auto; }
    h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
    h2 { margin: 34px 0 12px; font-size: 21px; letter-spacing: 0; }
    h3 { margin: 22px 0 10px; font-size: 16px; letter-spacing: 0; }
    p { max-width: 980px; }
    .muted { color: #64748b; }
    header .muted { color: #cbd5e1; }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin: 18px 0 24px;
    }
    .card {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 14px 16px;
    }
    .card-title { color: #475569; font-size: 13px; }
    .card-value { font-size: 27px; font-weight: 700; margin-top: 4px; }
    .card-subtitle { color: #64748b; font-size: 12px; margin-top: 3px; }
    section {
      background: white;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 18px 18px 20px;
      margin: 18px 0;
      overflow-x: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: white;
    }
    th, td {
      padding: 7px 9px;
      border-bottom: 1px solid #e5e7eb;
      text-align: right;
      white-space: nowrap;
    }
    th:first-child, td:first-child { text-align: left; }
    th { color: #334155; background: #f8fafc; font-weight: 650; }
    .two-col {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 16px;
    }
    .venn-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
      align-items: start;
    }
    .venn-grid svg {
      width: 100%;
      height: auto;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
    }
    .note {
      border-left: 4px solid #2563eb;
      padding: 10px 12px;
      background: #eff6ff;
      color: #1e3a8a;
      margin: 12px 0;
    }
    .empty { padding: 18px; color: #64748b; background: #f8fafc; border-radius: 8px; }
    code { background: #f1f5f9; padding: 1px 4px; border-radius: 4px; }
    @media (max-width: 850px) {
      header, main { padding-left: 18px; padding-right: 18px; }
      .two-col { grid-template-columns: 1fr; }
    }
    """

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DDA vs DIA comparison report</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>DDA vs DIA comparison report</h1>
    <div class="muted">Generated {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</div>
  </header>
  <main>
    <p>
      This report compares DIA-NN <code>{html.escape(str(args.diann_report))}</code>
      against FragPipe/MSFragger outputs in <code>{html.escape(str(args.dda_dir))}</code>.
      Stripped peptide overlap is the primary peptide comparison because this DIA-NN file has
      no modification annotations in <code>Modified.Sequence</code>, while MSFragger reports
      fixed and variable mass annotations.
    </p>
    <div class="cards">{cards}</div>

    <section>
      <h2>Global Identification Overlap</h2>
      <div class="note">
        Modified peptide rows are notation-sensitive. DDA fixed Cys carbamidomethyl
        annotations were removed for that key, but variable modifications remain method-specific.
      </div>
      {html_table(overlap_display)}
    </section>

    <section>
      <h2>Venn Overlap Summary</h2>
      <p class="muted">
        These diagrams show DIA-only, shared, and DDA-only counts for the main comparison levels.
      </p>
      <div class="venn-grid">{venns}</div>
    </section>

    <section>
      <h2>Total Unique Peptide Venn</h2>
      <p class="muted">
        This Venn uses unique stripped peptide sequences pooled across all runs/fractions.
        The union is the total number of unique peptide sequences found by either DIA or DDA.
      </p>
      <div class="venn-grid">{unique_peptide_venn}</div>
    </section>

    <section>
      <h2>Unique Peptides Per Run</h2>
      <p class="muted">
        Counts are distinct stripped peptide sequences within each matched run/fraction.
        Shared unique peptides are sequences found by both DIA and DDA in the same run/fraction.
      </p>
      {grouped_bar_svg(fractions, [("DIA", dia_peptides, "#2563eb"), ("DDA", dda_peptides, "#0f766e")])}
      {html_table(unique_per_run_display)}
    </section>

    <section>
      <h2>Stripped Peptide Overlap Per Fraction</h2>
      {line_svg(fractions, jaccards)}
      {html_table(per_fraction_display)}
    </section>

    <section>
      <h2>Intensity Agreement</h2>
      <p class="muted">
        Intensities are compared only for stripped peptides detected by both methods with positive signal.
        DIA values are summed <code>Precursor.Quantity</code>; DDA values are summed PSM <code>Intensity</code>.
        Absolute scales are not expected to match, so log2 correlation is the useful metric.
      </p>
      <div class="two-col">
        <div>
          <h3>Global matched peptide intensity</h3>
          {scatter_svg(matched_global["dia_log2"], matched_global["dda_log2"])}
        </div>
        <div>
          <h3>Per-fraction correlations</h3>
          {html_table(corr_display)}
        </div>
      </div>
    </section>

    <section>
      <h2>Detection Frequency Across Fractions</h2>
      <p class="muted">
        These are fraction-detection counts, not biological replicate reproducibility.
      </p>
      {html_table(detection_summary)}
      <div class="two-col">
        <div>
          <h3>Top DIA-only peptides by fraction count</h3>
          {html_table(top_dia_only[["peptide", "DIA fractions", "DDA fractions"]])}
        </div>
        <div>
          <h3>Top DDA-only peptides by fraction count</h3>
          {html_table(top_dda_only[["peptide", "DIA fractions", "DDA fractions"]])}
        </div>
      </div>
    </section>

    <section>
      <h2>Inputs And Filters</h2>
      <table>
        <tbody>
          <tr><th>DIA-NN report</th><td>{html.escape(str(args.diann_report))}</td></tr>
          <tr><th>DIA precursor Q filter</th><td>Q.Value &lt;= {html.escape(str(args.diann_q))}</td></tr>
          <tr><th>DDA PSM table</th><td>{html.escape(str(args.dda_dir / "psm.tsv"))}</td></tr>
          <tr><th>DDA PSM probability filter</th><td>Probability &gt;= {html.escape(str(args.dda_min_probability))}</td></tr>
          <tr><th>Output</th><td>{html.escape(str(output))}</td></tr>
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_doc, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diann-report", type=Path, default=Path("diann_out/report.parquet"))
    parser.add_argument("--dda-dir", type=Path, default=Path("MSfraggerResults"))
    parser.add_argument("--output", type=Path, default=Path("dda_dia_comparison_report.html"))
    parser.add_argument("--diann-q", type=float, default=0.01, help="DIA-NN precursor Q.Value threshold")
    parser.add_argument(
        "--dda-min-probability",
        type=float,
        default=0.0,
        help="Minimum FragPipe PSM Probability; default keeps the filtered psm.tsv as exported",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    psm_path = args.dda_dir / "psm.tsv"
    ion_path = args.dda_dir / "ion.tsv"
    peptide_path = args.dda_dir / "peptide.tsv"
    protein_path = args.dda_dir / "protein.tsv"
    required = [args.diann_report, psm_path, ion_path, peptide_path, protein_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files: " + ", ".join(missing))

    diann = read_diann(args.diann_report, args.diann_q)
    dda_psm = read_dda_psm(psm_path, args.dda_min_probability)
    dda_global = read_dda_global(ion_path, peptide_path, protein_path)

    per_fraction = per_fraction_summary(diann, dda_psm)
    overlap = global_overlap_table(diann, dda_global)
    matched_fraction, corr_fraction = intensity_by_fraction(diann, dda_psm)
    matched_global = global_intensity_match(diann, dda_global)
    detection = detection_frequency(diann, dda_psm)

    write_report(
        args.output,
        diann,
        dda_psm,
        dda_global,
        per_fraction,
        overlap,
        matched_fraction,
        corr_fraction,
        matched_global,
        detection,
        args,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
