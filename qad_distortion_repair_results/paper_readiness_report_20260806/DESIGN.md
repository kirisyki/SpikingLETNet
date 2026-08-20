# Visual contract

## Audience and decision

- Audience: technical ML/SNN researchers preparing the ICLR 2027 submission.
- Decision: determine whether current evidence supports saying that QAD repairs W4A4 Integer-LIF quantization distortion.
- Claim boundary: single-seed historical QAD versus 16-epoch STE-QAT; descriptive representation evidence, not a fair algorithmic causal comparison.

## Semantic roles

- QAD: blue categorical series.
- STE-QAT: orange categorical series.
- FP-QIF teacher: neutral reference.
- Rates are stored as decimal fractions and rendered as percentages.
- Positive QAD-minus-STE code disagreement means QAD is farther from the reference.
- Positive feature repair means STE normalized MSE minus QAD normalized MSE, so larger is better for QAD.

## Chart map

| Report segment | Analytical question | Grain | Chart | Fields | Supported claim |
|---|---|---|---|---|---|
| Global code fidelity | Does QAD reduce QIF code mismatch? | reference design × method | grouped bar | reference, method, disagreement | QAD has higher code disagreement under both references |
| KD target alignment | Which distilled features are closer to the teacher? | feature × method | grouped bar | feature, method, normalized MSE | QAD improves normalized MSE at all six KD targets |
| Layer localization | Is the code-fidelity result localized? | stage | signed bar | stage, QAD-minus-STE disagreement | every stage has higher QAD mismatch |
| Mechanism-task closure | Does greater feature repair predict greater source-level mIoU gain? | 35 source images | scatter | feature repair, mIoU gain, source id | no stable source-level association is observed |

## Tables

- Feature audit: exact normalized-MSE and cosine-distance effects for all six KD targets.
- Claim audit: allowed and disallowed paper statements with evidence status.
- Per-class task results: exact FP, QAD, and STE IoU values.

## Accessibility and layout

- Every multi-series chart has a legend; the signed single-series stage chart does not.
- All major visuals are full width with adjacent interpretation.
- Tables preserve exact values and caveats behind headline claims.
- The portable report includes a semantic HTML fallback.

