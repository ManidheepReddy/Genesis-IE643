# Genesis — quantitative results

Mean over the prompt sequences of each set (± = standard deviation across sequences). Arrows: ↑ higher is better, ↓ lower is better. Seam jump: 1.0 = seam is as smooth as a normal frame step.

## day3  (balloon, boat, car, fox)

| method | n | CLIP-T x100 ↑ | Prompt top-1 ↑ | Prompt progress ↑ | Subject cons. (DINO) ↑ | Background cons. (CLIP) ↑ | Start-end identity ↑ | Seam jump (pixel) ↓ | Seam jump (DINO) ↓ | Seam percentile ↓ | Colour drift dE (last clip) ↓ | Overexposed % (last clip) ↓ | CLIP-IQA ↑ | CLIP-IQA drop first->last ↓ | Sharpness ↑ | Motion (px/frame) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 4 | 29.28 ± 2.64 | 0.450 ± 0.100 | 1.20 ± 0.76 | 0.817 ± 0.074 | 0.919 ± 0.014 | 0.618 ± 0.160 | 48.18 ± 63.54 | 204.08 ± 290.30 | 99.71 ± 0.57 | 18.99 ± 9.49 | 12.48 ± 16.84 | 0.771 ± 0.137 | 0.168 ± 0.208 | 365 ± 200 | 2.32 ± 1.65 |
| A_lastframe | 4 | 29.50 ± 3.35 | 0.300 ± 0.115 | -0.00 ± 0.76 | 0.865 ± 0.087 | 0.941 ± 0.022 | 0.685 ± 0.181 | 2.02 ± 1.55 | 2.98 ± 2.00 | 61.10 ± 19.73 | 17.08 ± 13.85 | 7.27 ± 7.84 | 0.788 ± 0.062 | 0.206 ± 0.165 | 368 ± 265 | 3.94 ± 4.05 |
| B_m2 | 4 | 28.71 ± 2.51 | 0.300 ± 0.115 | -0.34 ± 1.68 | 0.865 ± 0.075 | 0.939 ± 0.016 | 0.669 ± 0.150 | 1.79 ± 0.68 | 3.05 ± 0.89 | 73.35 ± 13.96 | 19.55 ± 17.22 | 6.79 ± 8.72 | 0.818 ± 0.073 | 0.096 ± 0.155 | 427 ± 314 | 3.47 ± 2.73 |
| C_anchor | 4 | 29.19 ± 2.58 | 0.250 ± 0.100 | -0.26 ± 0.59 | 0.873 ± 0.077 | 0.939 ± 0.017 | 0.700 ± 0.140 | 2.47 ± 1.42 | 3.25 ± 1.23 | 82.67 ± 9.52 | 6.87 ± 1.03 | 3.31 ± 6.35 | 0.797 ± 0.087 | 0.147 ± 0.083 | 599 ± 653 | 2.87 ± 2.21 |
| B_m2_lora | 4 | 29.31 ± 2.84 | 0.300 ± 0.115 | -0.52 ± 1.07 | 0.887 ± 0.083 | 0.950 ± 0.022 | 0.761 ± 0.157 | 1.55 ± 0.72 | 1.44 ± 0.37 | 64.32 ± 13.02 | 18.37 ± 14.32 | 1.82 ± 2.56 | 0.858 ± 0.080 | 0.092 ± 0.162 | 318 ± 224 | 2.80 ± 2.43 |
| C_anchor_lora | 4 | 29.07 ± 3.20 | 0.300 ± 0.115 | -0.64 ± 0.75 | 0.895 ± 0.081 | 0.947 ± 0.021 | 0.777 ± 0.131 | 2.75 ± 2.74 | 2.35 ± 1.26 | 80.51 ± 9.92 | 6.93 ± 1.94 | 2.78 ± 4.22 | 0.830 ± 0.108 | 0.143 ± 0.244 | 448 ± 482 | 2.42 ± 2.51 |

## day5  (cutdemo, fox20b)

| method | n | CLIP-T x100 ↑ | Prompt top-1 ↑ | Prompt progress ↑ | Subject cons. (DINO) ↑ | Background cons. (CLIP) ↑ | Start-end identity ↑ | Seam jump (pixel) ↓ | Seam jump (DINO) ↓ | Seam percentile ↓ | Colour drift dE (last clip) ↓ | Overexposed % (last clip) ↓ | CLIP-IQA ↑ | CLIP-IQA drop first->last ↓ | Sharpness ↑ | Motion (px/frame) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_anchor | 1 | 29.62 | 0.000 | -0.69 | 0.840 | 0.943 | 0.682 | 1.36 | 1.34 | 58.31 | 8.54 | 7.66 | 0.514 | -0.134 | 1107 | 2.77 |
| C_anchor_lora | 2 | 28.05 ± 0.88 | 0.083 ± 0.118 | -0.23 ± 1.95 | 0.887 ± 0.068 | 0.953 ± 0.013 | 0.641 ± 0.024 | 2.80 ± 1.80 | 3.97 ± 3.09 | 73.29 ± 9.98 | 5.79 ± 3.09 | 7.45 ± 1.99 | 0.574 ± 0.214 | -0.109 ± 0.062 | 693 ± 39 | 1.36 ± 0.24 |
