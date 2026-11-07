# TextSwap Studio

Private Colab notebook for replacing text inside scanned documents, images and PDFs while matching the original font, size, colour, case, weight, width and edge softness. Watermark / logo / stamp pixels behind the text are detected and preserved.

## Run in Colab

Open the notebook directly:

`https://colab.research.google.com/github/maxico714/textswap-studio/blob/main/TextSwap_Studio.ipynb`

Colab may ask you to authorize GitHub access because this repository is private.

## What it does

- Finds text with OCR and replaces it without touching the rest of the page.
- Splits the source word into per-character cells before drawing.
- Locks every replacement character to the exact source pixel height.
- Matches font family, bold/italic, source case, colour, letter width and scan softness automatically.
- Detects underlays (watermark, emblem, stamp) and protects their visible pixels while erasing old glyph strokes.
- Supports batch images and scanned PDFs, OCR-only mode, ZIP downloads and page previews.
- Polished Hinglish form UI with advanced controls collapsed by default.

## Notes

- Run the single hidden form cell in Colab; the UI appears below it.
- First run installs Tesseract and Python dependencies, so it can take a minute.
- Files live only in the Colab session temp folder and are cleared on reset/reload.
