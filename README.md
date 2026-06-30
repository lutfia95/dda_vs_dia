# DDA vs DIA comparison report

Generate an HTML report comparing DIA-NN DIA search results with FragPipe/MSFragger DDA outputs.

The script summarizes peptide, precursor, protein/group, fraction-level, and intensity agreement between the two workflows. It is intended for quick QC and method comparison after both searches have already been completed.

## Installation

Requirements:

- Python 3.10 or newer
- `numpy`
- `pandas`
- `pyarrow` or another pandas-compatible Parquet engine

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas pyarrow
```

No package build step is required. The report generator is a standalone Python script.

## Required input files

The default command expects this layout:

```text
.
├── compare_dda_dia_report.py
├── diann_out/
│   └── report.parquet
└── MSfraggerResults/
    ├── psm.tsv
    ├── ion.tsv
    ├── peptide.tsv
    └── protein.tsv
```

You can also pass custom paths with command-line options.

### DIA-NN input

`report.parquet` must be a DIA-NN report table in Parquet format. The script reads these columns:

- `Run`
- `Precursor.Id`
- `Modified.Sequence`
- `Stripped.Sequence`
- `Precursor.Charge`
- `Protein.Group`
- `Protein.Ids`
- `Precursor.Quantity`
- `Q.Value`
- `PG.Q.Value`
- `Decoy`

Rows are filtered before comparison. The script excludes DIA-NN decoy rows with `Decoy != 0` and keeps only targets with `Q.Value <= 0.01` by default.

### FragPipe/MSFragger inputs

The DDA directory must contain FragPipe/MSFragger tab-separated output files:

- `psm.tsv`
- `ion.tsv`
- `peptide.tsv`
- `protein.tsv`

Required columns:

`psm.tsv`

- `Spectrum`
- `Spectrum File`
- `Peptide`
- `Modified Peptide`
- `Charge`
- `Probability`
- `Intensity`
- `Protein`

`ion.tsv`

- `Peptide Sequence`
- `Modified Sequence`
- `Charge`
- `Intensity`
- `Protein`

`peptide.tsv`

- `Peptide`
- `Intensity`
- `Spectral Count`

`protein.tsv`

- `Protein`
- `Total Intensity`
- `Unique Peptides`

DDA PSMs are kept as exported by default. Use `--dda-min-probability` to apply an additional minimum PSM probability filter.

## Usage

Run with the default paths:

```bash
python compare_dda_dia_report.py
```

Run with explicit input and output paths:

```bash
python compare_dda_dia_report.py \
  --diann-report diann_out/report.parquet \
  --dda-dir MSfraggerResults \
  --output dda_dia_comparison_report.html
```

Optional filters:

```bash
python compare_dda_dia_report.py \
  --diann-q 0.01 \
  --dda-min-probability 0.9
```

Command-line options:

```text
--diann-report PATH          DIA-NN report Parquet file
--dda-dir PATH               Directory containing psm.tsv, ion.tsv, peptide.tsv, and protein.tsv
--output PATH                Output HTML report path
--diann-q FLOAT              DIA-NN precursor Q.Value threshold, default 0.01
--dda-min-probability FLOAT  Minimum FragPipe PSM Probability, default 0.0
```

## Output

The script writes one self-contained HTML file, `dda_dia_comparison_report.html` by default.

The report includes:

- Global identification overlap for stripped peptides, stripped peptide plus charge, modified peptides, precursors, and proteins/protein groups
- Venn-style overlap summaries for key comparison levels
- Unique peptide counts per run or fraction
- Per-fraction peptide overlap and Jaccard similarity
- DIA vs DDA peptide intensity agreement using log2-transformed positive intensities
- Detection frequency tables showing peptides found by both methods, DIA only, or DDA only
- Input paths and filter settings used to generate the report

DIA-NN decoys are skipped and are not included in any counts, overlaps, intensity comparisons, or detection frequency tables.

Fractions are inferred from input file or run names using `Fraction_<value>` when present. If that pattern is not present, the cleaned run or spectrum file name is used as the fraction label.

## Notes

- Modified peptide overlap is sensitive to modification notation differences between DIA-NN and MSFragger.
- Fixed cysteine carbamidomethyl annotations are normalized in the DDA ion table for the modified peptide key, but variable modifications remain method-specific.
- Intensity values are useful for relative agreement checks, not direct absolute scale matching between DIA and DDA workflows.

## License

MIT License. See [LICENSE](LICENSE).
