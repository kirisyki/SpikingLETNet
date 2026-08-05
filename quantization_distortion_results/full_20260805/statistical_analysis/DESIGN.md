# Visual contract

## Audience and decision

- Audience: technical ML/SNN researchers preparing an ICLR submission.
- Decision: determine what kind of Integer-LIF code distortion is present and which matched training comparison should be run next.
- Claim boundary: frozen-checkpoint descriptive and inferential audit; no causal claim that QAD repairs the observed pattern.

## Semantic roles

- FP-QIF: neutral baseline.
- W4: blue categorical series.
- A4-input: purple categorical series.
- W4A4-input: orange categorical series.
- Positive/negative movements: semantic movement styling in audit tables.
- Rates: stored as decimal fractions and rendered as percentages.

## Chart map

| Question | Grain | Chart | Encodings | Reason |
|---|---|---|---|---|
| Where is code distortion concentrated? | stage × perturbation | grouped bar | stage → x, disagreement → y, perturbation → color | stage comparison is the core localization result |
| Does representation distortion co-occur with output instability? | 35 source images | scatter | disagreement → x, prediction flip → y, source id → label | preserves the independent cluster unit |
| Which classes are most sensitive? | class × mode | grouped bar | class → x, IoU → y, mode → color | exposes non-uniform downstream impact |

## Tables

- Stage morphology: signed error, zero delta, saturation delta, and contribution share.
- Top layers: exact layer names and all three perturbation rates for auditability.

## Accessibility and layout

- Legends identify every multi-series encoding.
- No single-series legend or redundant color-by-axis mapping.
- Tables provide exact values behind every major visual claim.
- Report uses full-width technical evidence blocks and a semantic HTML fallback.
