# Traditional Chinese Character Mapping

## Coverage

- CNN output classes: 13,065
- Mapped character candidates: 13,065
- Unique character candidates: 13,065

## Evidence Levels

- Direct consistent: 12978
- Reconstructed candidates: 85
- Visually supported candidates: 2

## Two Special Cases

- Class 2781: U+6FC8, Big5 E9EF; the original filename additionally contains ASCII 'u'.
- Class 12332: U+5D92, Big5 E565; the original filename additionally contains ASCII 'p'.

The handwriting and encoded glyph bytes support
these candidates, but the extra bytes have not
been explained by the source dataset author.

## Important Limitations

- This is a reconstructed mapping, not a fully
  authoritative original label dictionary.
- Model accuracy metrics remain unchanged.
- Inference code must verify whether model output
  indices are zero-based before using this mapping.
