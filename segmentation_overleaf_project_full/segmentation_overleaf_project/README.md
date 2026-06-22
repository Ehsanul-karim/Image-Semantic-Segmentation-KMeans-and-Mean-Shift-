# Overleaf LaTeX Project - Image Segmentation Report

This folder is ready to upload to Overleaf as a ZIP project.

## Main files
- `main.tex`: full report source. It includes the full 478-image result gallery by default.
- `main_quick_preview.pdf`: compiled preview PDF with the 478-image gallery disabled for speed.
- `outputs/`: generated output images, self GT, peer GT, and generated evaluation CSVs.
- `numerical_results/`: clean CSV copies for own-GT summaries and peer-GT evaluation.
- `materials/programs/`: Python source files and helper evaluation scripts.
- `annotations/`: self and peer LabelMe-style JSON annotations.

## Overleaf compile note
The full appendix contains 478 result figures. If Overleaf compilation becomes slow, change this line in `main.tex`:

```tex
\includeallresultgallerytrue
```

to:

```tex
\includeallresultgalleryfalse
```

Use `\includeallresultgallerytrue` for the final submitted report if you want the PDF to contain every output result figure.

## Replace outputs later
You can replace the `outputs/` folder with your updated full output folder, regenerate the CSVs with the scripts in `materials/programs/`, and keep the same LaTeX structure.
