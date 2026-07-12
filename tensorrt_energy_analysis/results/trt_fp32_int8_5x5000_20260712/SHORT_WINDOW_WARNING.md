# Short-window validation warning

This run is retained as a diagnostic record and must not be used as the formal energy result.

Each 5,000-image active window lasted only about four seconds. The NVML cumulative energy counter
was 10.0% to 15.2% higher than bounded 100 ms power integration, whereas a diagnostic 50,000-image
window reduced the discrepancy to 1.20% for FP32 and 0.72% for INT8. The stable CV therefore shows
repeatability but does not establish counter accuracy for this short window.

Use the corresponding `5x50000` long-window run as the formal result.
