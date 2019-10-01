# Description of Geolithe datasets

The table below summarizes for all the 3D point clouds provided by Geolithe the number of points contained into each cloud, as well as the definition domain for the three cartesian coordinates (`x`, `y` and `z`).

| file        | point nb  |    x    |    y    |    z    |
|-------------|-----------|---------|---------|---------|
| Beausoleil1 | 4 468 552   | 56.44   | 68.07   | 31.52   |
| Beausoleil2 | 1 035 672   | 57.11   | 87.94   | 32.28   |
| castillon   | 13 590 290  | 103.75  | 107.81  | 119.91  |
| Chambon     | 2 442 218   | 11.86   | 9.52    | 13.48   |
| escale      | 1 069 569   | 39.06   | 54.45   | 32.80   |
| Nuage-XYZRGB-2018-09-17 (geocliff) | 107 903 504 | 135.59 | 134.55 | 118.04 |
| kansera     | 32 946 371  | 730.17  | 339.34  | 256.73  |
| Malaussene  | 9 840 758   | 613.70  | 691.40  | 364.80  |
| Pombourg    | 672 193     | 43.66   | 56.45   | 51.00   |
| Reyvroz     | 8 628 452   | 570.09  | 358.86  | 159.62  |
| Tancarville | 2 534 384   | 15.32   | 11.01   | 13.78   |

In order to define more finely the bin features, the second table below
proposes a bin size for each dataset, computed as the maximum 2D radius with
10-point neighborhoods. These values may be supplied to the `cluster` geo3d command
in the configuration file with the `bin` parameter name.

| file         | bin size |
|--------------|----------|
| Beausoleil_1 |  9       |
| Beausoleil_2 | 10       |
| castillon    | 12       |
| Chambon      |          |
| escale       |  3       |
| geocliff36   |  8       |
| kansera      | 26       |
| Malaussene   | 21       |
| Pombourg     |          |
| Reyvroz      | 23       |
| Tancarville  |  1       |
