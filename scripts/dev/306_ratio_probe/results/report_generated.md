### A. Ratio matrix -- median over 8 samples per class

Ratio = uncompressed / compressed; > 1 means the codec found something.

| asset class | layout | lzma-fast | zlib-6 | zstd-19 | zstd-19-mt16 | zstd-3 | zstd-3-chunk4M-x8 | zstd-3-mt16 |
|---|---|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | nibble | 0.9999 | 1.0005 | 1.0000 | -- | 1.0000 | -- | -- |
| `dsv4f_ud_iq3xxs_iq2_xs` | plane | 0.9999 | 1.0014 | 1.0000 | -- | 1.0000 | -- | -- |
| `dsv4f_ud_iq3xxs_iq2_xs` | raw | 0.9999 | 1.0011 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `dsv4f_ud_iq3xxs_iq2_xs` | stride74 | 1.0223 | 1.0294 | 1.0299 | -- | 1.0255 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_s` | nibble | 0.9999 | 1.0007 | 1.0004 | -- | 1.0000 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_s` | plane | 1.0011 | 1.0069 | 1.0092 | -- | 1.0000 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_s` | raw | 0.9999 | 1.0031 | 1.0054 | 1.0054 | 1.0000 | 1.0000 | 1.0000 |
| `dsv4f_ud_iq3xxs_iq3_s` | stride110 | 1.0163 | 1.0260 | 1.0285 | -- | 1.0175 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_xxs` | nibble | 0.9999 | 1.0006 | 1.0000 | -- | 1.0000 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_xxs` | plane | 1.0002 | 1.0044 | 1.0068 | -- | 1.0001 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_xxs` | raw | 0.9999 | 1.0043 | 1.0064 | 1.0064 | 1.0000 | 1.0000 | 1.0000 |
| `dsv4f_ud_iq3xxs_iq3_xxs` | stride98 | 1.0207 | 1.0279 | 1.0276 | -- | 1.0289 | -- | -- |
| `dsv4f_ud_iq3xxs_mxfp4` | plane | 1.0601 | 1.0671 | 1.0719 | -- | 1.0686 | -- | -- |
| `dsv4f_ud_iq3xxs_mxfp4` | raw | 1.0345 | 1.0451 | 1.0494 | 1.0494 | 1.0487 | 1.0489 | 1.0491 |
| `dsv4f_ud_iq3xxs_mxfp4` | stride17 | 1.0772 | 1.0883 | 1.0934 | -- | 1.0877 | -- | -- |
| `dsv4f_ud_q3kxl_iq3_xxs` | nibble | 0.9999 | 1.0007 | 1.0001 | -- | 1.0000 | -- | -- |
| `dsv4f_ud_q3kxl_iq3_xxs` | plane | 1.0002 | 1.0049 | 1.0074 | -- | 1.0001 | -- | -- |
| `dsv4f_ud_q3kxl_iq3_xxs` | raw | 0.9999 | 1.0047 | 1.0071 | 1.0071 | 1.0000 | 1.0000 | 1.0000 |
| `dsv4f_ud_q3kxl_iq3_xxs` | stride98 | 1.0216 | 1.0287 | 1.0290 | -- | 1.0297 | -- | -- |
| `dsv4f_ud_q3kxl_mxfp4` | plane | 1.0387 | 1.0439 | 1.0478 | -- | 1.0456 | -- | -- |
| `dsv4f_ud_q3kxl_mxfp4` | raw | 1.0338 | 1.0433 | 1.0472 | 1.0472 | 1.0454 | 1.0456 | 1.0467 |
| `dsv4f_ud_q3kxl_mxfp4` | stride17 | 1.0751 | 1.0862 | 1.0908 | -- | 1.0846 | -- | -- |
| `hibernate_img` | nibble | 1.0322 | 1.0406 | 1.0456 | -- | 1.0434 | -- | -- |
| `hibernate_img` | raw | 1.0359 | 1.0352 | 1.0389 | 1.0389 | 1.0353 | 1.0356 | 1.0359 |
| `hibernate_img` | stride2 | 1.0367 | 1.0402 | 1.0456 | -- | 1.0410 | -- | -- |
| `hibernate_img` | stride4 | 1.0340 | 1.0408 | 1.0444 | -- | 1.0417 | -- | -- |
| `qwen27b_fp8` | nibble | 1.1399 | 1.1555 | 1.1811 | -- | 1.1806 | -- | -- |
| `qwen27b_fp8` | raw | 1.1559 | 1.2084 | 1.2104 | 1.2104 | 1.2108 | 1.2109 | 1.2109 |
| `qwen27b_fp8` | stride2 | 1.1556 | 1.2081 | 1.2101 | -- | 1.2105 | -- | -- |
| `qwen27b_fp8` | stride4 | 1.1553 | 1.2079 | 1.2096 | -- | 1.2099 | -- | -- |
| `qwen27b_int8` | nibble | 1.0993 | 1.1222 | 1.1239 | -- | 1.1302 | -- | -- |
| `qwen27b_int8` | raw | 1.1060 | 1.1295 | 1.1302 | 1.1302 | 1.1307 | 1.1307 | 1.1307 |
| `qwen27b_int8` | stride2 | 1.1057 | 1.1290 | 1.1301 | -- | 1.1306 | -- | -- |
| `qwen27b_int8` | stride4 | 1.1052 | 1.1286 | 1.1300 | -- | 1.1305 | -- | -- |
| `qwen35ba3b_ud_q3km_iq3_xxs` | nibble | 1.0007 | 1.0015 | 1.0015 | -- | 1.0008 | -- | -- |
| `qwen35ba3b_ud_q3km_iq3_xxs` | plane | 1.0012 | 1.0051 | 1.0073 | -- | 1.0013 | -- | -- |
| `qwen35ba3b_ud_q3km_iq3_xxs` | raw | 1.0010 | 1.0050 | 1.0072 | 1.0072 | 1.0011 | 1.0012 | 1.0011 |
| `qwen35ba3b_ud_q3km_iq3_xxs` | stride98 | 1.0196 | 1.0264 | 1.0258 | -- | 1.0261 | -- | -- |
| `qwen35ba3b_ud_q3km_iq4_xs` | nibble | 1.0039 | 1.0126 | 1.0081 | -- | 1.0016 | -- | -- |
| `qwen35ba3b_ud_q3km_iq4_xs` | plane | 1.0061 | 1.0128 | 1.0023 | -- | 1.0010 | -- | -- |
| `qwen35ba3b_ud_q3km_iq4_xs` | raw | 1.0056 | 1.0127 | 1.0016 | 1.0016 | 1.0016 | 1.0016 | 1.0014 |
| `qwen35ba3b_ud_q3km_iq4_xs` | stride136 | 1.0181 | 1.0275 | 1.0165 | -- | 1.0236 | -- | -- |
| `qwen35ba3b_ud_q3km_q3_k` | nibble | 0.9999 | 0.9997 | 1.0000 | -- | 1.0000 | -- | -- |
| `qwen35ba3b_ud_q3km_q3_k` | plane | 0.9999 | 1.0015 | 1.0022 | -- | 1.0000 | -- | -- |
| `qwen35ba3b_ud_q3km_q3_k` | raw | 0.9999 | 0.9997 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `qwen35ba3b_ud_q3km_q3_k` | stride110 | 1.0089 | 1.0132 | 1.0139 | -- | 1.0107 | -- | -- |
| `qwen35ba3b_ud_q3km_q4_k` | nibble | 1.0158 | 1.0222 | 1.0264 | -- | 1.0234 | -- | -- |
| `qwen35ba3b_ud_q3km_q4_k` | plane | 1.0160 | 1.0220 | 1.0248 | -- | 1.0231 | -- | -- |
| `qwen35ba3b_ud_q3km_q4_k` | raw | 1.0155 | 1.0216 | 1.0242 | 1.0242 | 1.0231 | 1.0233 | 1.0237 |
| `qwen35ba3b_ud_q3km_q4_k` | stride144 | 1.0364 | 1.0488 | 1.0518 | -- | 1.0502 | -- | -- |
| `qwen35ba3b_ud_q3km_q6_k` | nibble | 0.9999 | 1.0018 | 1.0025 | -- | 1.0000 | -- | -- |
| `qwen35ba3b_ud_q3km_q6_k` | plane | 1.0100 | 1.0145 | 1.0162 | -- | 1.0159 | -- | -- |
| `qwen35ba3b_ud_q3km_q6_k` | raw | 0.9999 | 1.0024 | 1.0047 | 1.0047 | 1.0000 | 1.0000 | 1.0000 |
| `qwen35ba3b_ud_q3km_q6_k` | stride210 | 1.0209 | 1.0268 | 1.0288 | -- | 1.0281 | -- | -- |

### B. Decompress rate matrix -- median MB/s (1e6 B/s), inverse permutation included

| asset class | layout | lzma-fast | zlib-6 | zstd-19 | zstd-19-mt16 | zstd-3 | zstd-3-chunk4M-x8 | zstd-3-mt16 |
|---|---|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | nibble | 974 | 444 | 1147 | -- | 1155 | -- | -- |
| `dsv4f_ud_iq3xxs_iq2_xs` | plane | 1597 | 339 | 2114 | -- | 2161 | -- | -- |
| `dsv4f_ud_iq3xxs_iq2_xs` | raw | 4016 | 388 | 12830 | 11797 | 12881 | 15330 | 11213 |
| `dsv4f_ud_iq3xxs_iq2_xs` | stride74 | 194 | 357 | 693 | -- | 838 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_s` | nibble | 953 | 435 | 1005 | -- | 1111 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_s` | plane | 60 | 328 | 926 | -- | 2116 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_s` | raw | 3906 | 383 | 1487 | 1468 | 12825 | 16524 | 12157 |
| `dsv4f_ud_iq3xxs_iq3_s` | stride110 | 36 | 270 | 594 | -- | 865 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_xxs` | nibble | 819 | 406 | 944 | -- | 976 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_xxs` | plane | 617 | 320 | 936 | -- | 1871 | -- | -- |
| `dsv4f_ud_iq3xxs_iq3_xxs` | raw | 3131 | 370 | 1513 | 1541 | 11019 | 11326 | 9816 |
| `dsv4f_ud_iq3xxs_iq3_xxs` | stride98 | 32 | 275 | 545 | -- | 542 | -- | -- |
| `dsv4f_ud_iq3xxs_mxfp4` | plane | 27 | 280 | 788 | -- | 773 | -- | -- |
| `dsv4f_ud_iq3xxs_mxfp4` | raw | 27 | 323 | 1460 | 1452 | 1420 | 4935 | 1436 |
| `dsv4f_ud_iq3xxs_mxfp4` | stride17 | 27 | 273 | 695 | -- | 664 | -- | -- |
| `dsv4f_ud_q3kxl_iq3_xxs` | nibble | 799 | 402 | 925 | -- | 961 | -- | -- |
| `dsv4f_ud_q3kxl_iq3_xxs` | plane | 655 | 327 | 951 | -- | 2176 | -- | -- |
| `dsv4f_ud_q3kxl_iq3_xxs` | raw | 3408 | 371 | 1478 | 1474 | 10936 | 11739 | 10393 |
| `dsv4f_ud_q3kxl_iq3_xxs` | stride98 | 32 | 274 | 542 | -- | 539 | -- | -- |
| `dsv4f_ud_q3kxl_mxfp4` | plane | 27 | 290 | 868 | -- | 841 | -- | -- |
| `dsv4f_ud_q3kxl_mxfp4` | raw | 27 | 338 | 1529 | 1504 | 1464 | 4812 | 1482 |
| `dsv4f_ud_q3kxl_mxfp4` | stride17 | 28 | 311 | 896 | -- | 838 | -- | -- |
| `hibernate_img` | nibble | 34 | 283 | 766 | -- | 781 | -- | -- |
| `hibernate_img` | raw | 34 | 368 | 1507 | 1467 | 1497 | 4802 | 1483 |
| `hibernate_img` | stride2 | 32 | 334 | 967 | -- | 1045 | -- | -- |
| `hibernate_img` | stride4 | 51 | 337 | 967 | -- | 1034 | -- | -- |
| `qwen27b_fp8` | nibble | 71 | 374 | 934 | -- | 937 | -- | -- |
| `qwen27b_fp8` | raw | 31 | 324 | 1341 | 1324 | 1336 | 4443 | 1348 |
| `qwen27b_fp8` | stride2 | 31 | 287 | 891 | -- | 908 | -- | -- |
| `qwen27b_fp8` | stride4 | 31 | 288 | 871 | -- | 876 | -- | -- |
| `qwen27b_int8` | nibble | 65 | 391 | 842 | -- | 905 | -- | -- |
| `qwen27b_int8` | raw | 30 | 329 | 1478 | 1475 | 1508 | 4891 | 1509 |
| `qwen27b_int8` | stride2 | 30 | 295 | 962 | -- | 976 | -- | -- |
| `qwen27b_int8` | stride4 | 30 | 290 | 926 | -- | 960 | -- | -- |
| `qwen35ba3b_ud_q3km_iq3_xxs` | nibble | 461 | 399 | 926 | -- | 937 | -- | -- |
| `qwen35ba3b_ud_q3km_iq3_xxs` | plane | 398 | 323 | 942 | -- | 1982 | -- | -- |
| `qwen35ba3b_ud_q3km_iq3_xxs` | raw | 715 | 370 | 1454 | 1413 | 8728 | 9446 | 8921 |
| `qwen35ba3b_ud_q3km_iq3_xxs` | stride98 | 33 | 294 | 621 | -- | 628 | -- | -- |
| `qwen35ba3b_ud_q3km_iq4_xs` | nibble | 25 | 265 | 738 | -- | 880 | -- | -- |
| `qwen35ba3b_ud_q3km_iq4_xs` | plane | 26 | 302 | 1568 | -- | 1637 | -- | -- |
| `qwen35ba3b_ud_q3km_iq4_xs` | raw | 26 | 349 | 6164 | 6069 | 6035 | 10321 | 6150 |
| `qwen35ba3b_ud_q3km_iq4_xs` | stride136 | 25 | 247 | 692 | -- | 598 | -- | -- |
| `qwen35ba3b_ud_q3km_q3_k` | nibble | 723 | 709 | 911 | -- | 915 | -- | -- |
| `qwen35ba3b_ud_q3km_q3_k` | plane | 1227 | 631 | 1347 | -- | 1776 | -- | -- |
| `qwen35ba3b_ud_q3km_q3_k` | raw | 3012 | 2302 | 9699 | 9863 | 10343 | 10588 | 9425 |
| `qwen35ba3b_ud_q3km_q3_k` | stride110 | 238 | 355 | 604 | -- | 697 | -- | -- |
| `qwen35ba3b_ud_q3km_q4_k` | nibble | 27 | 282 | 675 | -- | 684 | -- | -- |
| `qwen35ba3b_ud_q3km_q4_k` | plane | 27 | 327 | 1068 | -- | 1041 | -- | -- |
| `qwen35ba3b_ud_q3km_q4_k` | raw | 27 | 357 | 1486 | 1483 | 1430 | 4864 | 1482 |
| `qwen35ba3b_ud_q3km_q4_k` | stride144 | 27 | 265 | 634 | -- | 630 | -- | -- |
| `qwen35ba3b_ud_q3km_q6_k` | nibble | 1052 | 312 | 914 | -- | 1259 | -- | -- |
| `qwen35ba3b_ud_q3km_q6_k` | plane | 70 | 529 | 1653 | -- | 1649 | -- | -- |
| `qwen35ba3b_ud_q3km_q6_k` | raw | 4409 | 399 | 1542 | 1529 | 13895 | 17771 | 13659 |
| `qwen35ba3b_ud_q3km_q6_k` | stride210 | 69 | 424 | 832 | -- | 823 | -- | -- |

### C. Best achievable ratio per asset class (any method)

| asset class | n | best method | ratio median | ratio min-max | decompress MB/s | compress MB/s | kill criterion (< 1.08) |
|---|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | 8 | stride74/zstd-19 | **1.0299** | 1.0295-1.0305 | 693 | 7 | **DEAD** |
| `dsv4f_ud_iq3xxs_iq3_s` | 8 | stride110/zstd-19 | **1.0285** | 1.0283-1.0287 | 594 | 6 | **DEAD** |
| `dsv4f_ud_iq3xxs_iq3_xxs` | 8 | stride98/zstd-3 | **1.0289** | 1.0283-1.0301 | 542 | 643 | **DEAD** |
| `dsv4f_ud_iq3xxs_mxfp4` | 8 | stride17/zstd-19 | **1.0934** | 1.0892-1.0999 | 695 | 5 | alive |
| `dsv4f_ud_q3kxl_iq3_xxs` | 8 | stride98/zstd-3 | **1.0297** | 1.0291-1.0306 | 539 | 633 | **DEAD** |
| `dsv4f_ud_q3kxl_mxfp4` | 8 | stride17/zstd-19 | **1.0908** | 1.0895-1.0915 | 896 | 6 | alive |
| `hibernate_img` | 8 | nibble/zstd-19 | **1.0456** | 1.0101-20189.1889 | 766 | 7 | **DEAD** |
| `qwen27b_fp8` | 8 | raw/zstd-3-mt16 | **1.2109** | 1.2032-1.2172 | 1348 | 1203 | alive |
| `qwen27b_int8` | 8 | raw/zstd-3 | **1.1307** | 1.1190-1.1438 | 1508 | 1149 | alive |
| `qwen35ba3b_ud_q3km_iq3_xxs` | 8 | stride98/zlib-6 | **1.0264** | 1.0249-1.0435 | 294 | 55 | **DEAD** |
| `qwen35ba3b_ud_q3km_iq4_xs` | 8 | stride136/zlib-6 | **1.0275** | 1.0253-1.0294 | 247 | 51 | **DEAD** |
| `qwen35ba3b_ud_q3km_q3_k` | 8 | stride110/zstd-19 | **1.0139** | 1.0121-1.0144 | 604 | 5 | **DEAD** |
| `qwen35ba3b_ud_q3km_q4_k` | 8 | stride144/zstd-19 | **1.0518** | 1.0340-1.0521 | 634 | 7 | **DEAD** |
| `qwen35ba3b_ud_q3km_q6_k` | 8 | stride210/zstd-19 | **1.0288** | 1.0283-1.0292 | 832 | 7 | **DEAD** |

### D. Cell verdicts -- serial speedup per (asset class, link)

Each cell: the SERIAL speedup of the method that maximises it for that link (pipelined bound in brackets). The no-compression baseline is 1.000x by definition, so > 1.000x is a win and < 1.000x means storing the asset RAW is strictly faster. The method is re-chosen per link, so a cell is the best this probe can do there, not the best-ratio method forced onto a link it does not suit.

| asset class | best ratio (any method) | T3 local NVMe / disk image 1.80 GB/s | T4 remote rig-2 over 40G 2.07 GB/s | T4 remote rig-2 over 40G 2.83 GB/s | T2 host RAM -> card 6.40 GB/s | T2 host RAM -> card 13.00 GB/s | verdict |
|---|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | 1.0299 | 0.895x [1.000x] raw/zstd-3-chunk4M-x8 | 0.881x [1.000x] raw/zstd-3-chunk4M-x8 | 0.844x [1.000x] raw/zstd-3-chunk4M-x8 | 0.705x [1.000x] raw/zstd-3-chunk4M-x8 | 0.541x [1.000x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `dsv4f_ud_iq3xxs_iq3_s` | 1.0285 | 0.902x [1.000x] raw/zstd-3-chunk4M-x8 | 0.889x [1.000x] raw/zstd-3-chunk4M-x8 | 0.854x [1.000x] raw/zstd-3-chunk4M-x8 | 0.721x [1.000x] raw/zstd-3-chunk4M-x8 | 0.560x [1.000x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `dsv4f_ud_iq3xxs_iq3_xxs` | 1.0289 | 0.863x [1.000x] raw/zstd-3-chunk4M-x8 | 0.845x [1.000x] raw/zstd-3-chunk4M-x8 | 0.800x [1.000x] raw/zstd-3-chunk4M-x8 | 0.639x [1.000x] raw/zstd-3-chunk4M-x8 | 0.466x [0.871x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `dsv4f_ud_iq3xxs_mxfp4` | 1.0934 | 0.759x [1.049x] raw/zstd-3-chunk4M-x8 | 0.728x [1.049x] raw/zstd-3-chunk4M-x8 | 0.655x [1.049x] raw/zstd-3-chunk4M-x8 | 0.444x [0.771x] raw/zstd-3-chunk4M-x8 | 0.279x [0.380x] raw/zstd-3-chunk4M-x8 | no win |
| `dsv4f_ud_q3kxl_iq3_xxs` | 1.0297 | 0.867x [1.000x] raw/zstd-3-chunk4M-x8 | 0.850x [1.000x] raw/zstd-3-chunk4M-x8 | 0.806x [1.000x] raw/zstd-3-chunk4M-x8 | 0.647x [1.000x] raw/zstd-3-chunk4M-x8 | 0.474x [0.903x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `dsv4f_ud_q3kxl_mxfp4` | 1.0908 | 0.752x [1.046x] raw/zstd-3-chunk4M-x8 | 0.721x [1.046x] raw/zstd-3-chunk4M-x8 | 0.647x [1.046x] raw/zstd-3-chunk4M-x8 | 0.437x [0.752x] raw/zstd-3-chunk4M-x8 | 0.273x [0.370x] raw/zstd-3-chunk4M-x8 | no win |
| `hibernate_img` | 1.0456 | 0.746x [1.036x] raw/zstd-3-chunk4M-x8 | 0.716x [1.036x] raw/zstd-3-chunk4M-x8 | 0.643x [1.036x] raw/zstd-3-chunk4M-x8 | 0.435x [0.750x] raw/zstd-3-chunk4M-x8 | 0.272x [0.369x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `qwen27b_fp8` | 1.2109 | 0.812x [1.211x] raw/zstd-3-chunk4M-x8 | 0.774x [1.211x] raw/zstd-3-chunk4M-x8 | 0.684x [1.211x] raw/zstd-3-chunk4M-x8 | 0.441x [0.694x] raw/zstd-3-chunk4M-x8 | 0.267x [0.342x] raw/zstd-3-chunk4M-x8 | no win |
| `qwen27b_int8` | 1.1307 | 0.798x [1.131x] raw/zstd-3-chunk4M-x8 | 0.765x [1.131x] raw/zstd-3-chunk4M-x8 | 0.683x [1.131x] raw/zstd-3-chunk4M-x8 | 0.456x [0.764x] raw/zstd-3-chunk4M-x8 | 0.282x [0.376x] raw/zstd-3-chunk4M-x8 | no win |
| `qwen35ba3b_ud_q3km_iq3_xxs` | 1.0264 | 0.841x [1.001x] raw/zstd-3-chunk4M-x8 | 0.821x [1.001x] raw/zstd-3-chunk4M-x8 | 0.770x [1.001x] raw/zstd-3-chunk4M-x8 | 0.597x [1.001x] raw/zstd-3-chunk4M-x8 | 0.421x [0.727x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `qwen35ba3b_ud_q3km_iq4_xs` | 1.0275 | 0.853x [1.002x] raw/zstd-3-chunk4M-x8 | 0.834x [1.002x] raw/zstd-3-chunk4M-x8 | 0.786x [1.002x] raw/zstd-3-chunk4M-x8 | 0.618x [1.002x] raw/zstd-3-chunk4M-x8 | 0.443x [0.794x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `qwen35ba3b_ud_q3km_q3_k` | 1.0139 | 0.855x [1.000x] raw/zstd-3-chunk4M-x8 | 0.836x [1.000x] raw/zstd-3-chunk4M-x8 | 0.789x [1.000x] raw/zstd-3-chunk4M-x8 | 0.623x [1.000x] raw/zstd-3-chunk4M-x8 | 0.449x [0.814x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `qwen35ba3b_ud_q3km_q4_k` | 1.0518 | 0.742x [1.023x] raw/zstd-3-chunk4M-x8 | 0.713x [1.023x] raw/zstd-3-chunk4M-x8 | 0.641x [1.023x] raw/zstd-3-chunk4M-x8 | 0.436x [0.760x] raw/zstd-3-chunk4M-x8 | 0.274x [0.374x] raw/zstd-3-chunk4M-x8 | **DEAD** |
| `qwen35ba3b_ud_q3km_q6_k` | 1.0288 | 0.908x [1.000x] raw/zstd-3-chunk4M-x8 | 0.896x [1.000x] raw/zstd-3-chunk4M-x8 | 0.863x [1.000x] raw/zstd-3-chunk4M-x8 | 0.735x [1.000x] raw/zstd-3-chunk4M-x8 | 0.578x [1.000x] raw/zstd-3-chunk4M-x8 | **DEAD** |

#### D.1 Required ratio `r_min = D/(D-L)` at the fastest decompress arm

`r_min` is the smallest ratio that could make a serial win, given the decompress rate actually measured. Compare it against the best ratio column above.

| asset class | fastest decompress arm | D (MB/s) | r_min @ 1.80 GB/s | r_min @ 2.07 GB/s | r_min @ 2.83 GB/s | r_min @ 6.40 GB/s | r_min @ 13.00 GB/s |
|---|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | raw/zstd-3-chunk4M-x8 | 15330 | 1.133 | 1.156 | 1.226 | 1.717 | 6.579 |
| `dsv4f_ud_iq3xxs_iq3_s` | raw/zstd-3-chunk4M-x8 | 16524 | 1.122 | 1.143 | 1.207 | 1.632 | 4.689 |
| `dsv4f_ud_iq3xxs_iq3_xxs` | raw/zstd-3-chunk4M-x8 | 11326 | 1.189 | 1.224 | 1.333 | 2.299 | impossible |
| `dsv4f_ud_iq3xxs_mxfp4` | raw/zstd-3-chunk4M-x8 | 4935 | 1.574 | 1.722 | 2.344 | impossible | impossible |
| `dsv4f_ud_q3kxl_iq3_xxs` | raw/zstd-3-chunk4M-x8 | 11739 | 1.181 | 1.214 | 1.318 | 2.199 | impossible |
| `dsv4f_ud_q3kxl_mxfp4` | raw/zstd-3-chunk4M-x8 | 4812 | 1.598 | 1.755 | 2.428 | impossible | impossible |
| `hibernate_img` | raw/zstd-3-chunk4M-x8 | 4802 | 1.600 | 1.758 | 2.435 | impossible | impossible |
| `qwen27b_fp8` | raw/zstd-3-chunk4M-x8 | 4443 | 1.681 | 1.872 | 2.755 | impossible | impossible |
| `qwen27b_int8` | raw/zstd-3-chunk4M-x8 | 4891 | 1.582 | 1.734 | 2.373 | impossible | impossible |
| `qwen35ba3b_ud_q3km_iq3_xxs` | raw/zstd-3-chunk4M-x8 | 9446 | 1.235 | 1.281 | 1.428 | 3.101 | impossible |
| `qwen35ba3b_ud_q3km_iq4_xs` | raw/zstd-3-chunk4M-x8 | 10321 | 1.211 | 1.251 | 1.378 | 2.632 | impossible |
| `qwen35ba3b_ud_q3km_q3_k` | raw/zstd-3-chunk4M-x8 | 10588 | 1.205 | 1.243 | 1.365 | 2.528 | impossible |
| `qwen35ba3b_ud_q3km_q4_k` | raw/zstd-3-chunk4M-x8 | 4864 | 1.587 | 1.741 | 2.391 | impossible | impossible |
| `qwen35ba3b_ud_q3km_q6_k` | raw/zstd-3-chunk4M-x8 | 17771 | 1.113 | 1.132 | 1.189 | 1.563 | 3.725 |

### E. Multi-thread and chunked-frame arms (raw layout)

| asset class | zstd-3 1T comp MB/s | zstd-3 16T comp MB/s | zstd-19 1T comp MB/s | zstd-19 16T comp MB/s | zstd-3 1T decomp MB/s | zstd-3 4 MiB frames x8 decomp MB/s | frame-chunk ratio |
|---|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | 1021 | 1077 | 6 | 6 | 12881 | 15330 | 1.0000 |
| `dsv4f_ud_iq3xxs_iq3_s` | 1063 | 1149 | 6 | 6 | 12825 | 16524 | 1.0000 |
| `dsv4f_ud_iq3xxs_iq3_xxs` | 1078 | 1025 | 5 | 5 | 11019 | 11326 | 1.0000 |
| `dsv4f_ud_iq3xxs_mxfp4` | 683 | 856 | 5 | 5 | 1420 | 4935 | 1.0489 |
| `dsv4f_ud_q3kxl_iq3_xxs` | 1023 | 1021 | 5 | 5 | 10936 | 11739 | 1.0000 |
| `dsv4f_ud_q3kxl_mxfp4` | 713 | 946 | 6 | 6 | 1464 | 4812 | 1.0456 |
| `hibernate_img` | 606 | 875 | 7 | 7 | 1497 | 4802 | 1.0356 |
| `qwen27b_fp8` | 1122 | 1203 | 5 | 5 | 1336 | 4443 | 1.2109 |
| `qwen27b_int8` | 1149 | 1211 | 6 | 6 | 1508 | 4891 | 1.1307 |
| `qwen35ba3b_ud_q3km_iq3_xxs` | 986 | 1040 | 4 | 4 | 8728 | 9446 | 1.0012 |
| `qwen35ba3b_ud_q3km_iq4_xs` | 626 | 792 | 5 | 5 | 6035 | 10321 | 1.0016 |
| `qwen35ba3b_ud_q3km_q3_k` | 2252 | 1361 | 5 | 5 | 10343 | 10588 | 1.0000 |
| `qwen35ba3b_ud_q3km_q4_k` | 583 | 867 | 7 | 7 | 1430 | 4864 | 1.0233 |
| `qwen35ba3b_ud_q3km_q6_k` | 1103 | 1126 | 7 | 7 | 13895 | 17771 | 1.0000 |

### F. Sample provenance

| asset class | n | bytes/sample | source file(s) | example tensor | format | block bytes |
|---|---|---|---|---|---|---|
| `dsv4f_ud_iq3xxs_iq2_xs` | 8 | 16777206 | 2 files, e.g. `DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00002-of-00004.gguf` | `blk.18.ffn_up_exps.weight` | IQ2_XS | 74 |
| `dsv4f_ud_iq3xxs_iq3_s` | 8 | 16777200 | `DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00003-of-00004.gguf` | `blk.26.ffn_up_exps.weight` | IQ3_S | 110 |
| `dsv4f_ud_iq3xxs_iq3_xxs` | 8 | 16777208 | 2 files, e.g. `DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00002-of-00004.gguf` | `blk.36.ffn_down_exps.weight` | IQ3_XXS | 98 |
| `dsv4f_ud_iq3xxs_mxfp4` | 8 | 16777215 | 2 files, e.g. `DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00003-of-00004.gguf` | `blk.26.ffn_down_exps.weight` | MXFP4 | 17 |
| `dsv4f_ud_q3kxl_iq3_xxs` | 8 | 16777208 | 3 files, e.g. `DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00002-of-00004.gguf` | `blk.32.ffn_up_exps.weight` | IQ3_XXS | 98 |
| `dsv4f_ud_q3kxl_mxfp4` | 8 | 16777215 | 3 files, e.g. `DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00002-of-00004.gguf` | `blk.0.ffn_down_exps.weight` | MXFP4 | 17 |
| `hibernate_img` | 8 | 16777216 | `rank0_GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d.pt` | `<image chunk 0>` | mixed-Q4_K/Q6_K/F32 | n/a (flat 1-byte elements) |
| `qwen27b_fp8` | 8 | 16777216 | 8 files, e.g. `layers-15.safetensors` | `model.language_model.layers.5.mlp.up_proj.weight` | F8_E4M3 | n/a (flat 1-byte elements) |
| `qwen27b_int8` | 8 | 16777216 | `model.safetensors` | `model.language_model.layers.2.linear_attn.in_proj_z.weight` | I8 | n/a (flat 1-byte elements) |
| `qwen35ba3b_ud_q3km_iq3_xxs` | 8 | 16777208 | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | `blk.4.ffn_gate_exps.weight` | IQ3_XXS | 98 |
| `qwen35ba3b_ud_q3km_iq4_xs` | 8 | 16777096 | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | `blk.20.ffn_down_exps.weight` | IQ4_XS | 136 |
| `qwen35ba3b_ud_q3km_q3_k` | 8 | 16777200 | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | `blk.40.ffn_gate_exps.weight` | Q3_K | 110 |
| `qwen35ba3b_ud_q3km_q4_k` | 8 | 16777152 | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | `blk.40.ffn_down_exps.weight` | Q4_K | 144 |
| `qwen35ba3b_ud_q3km_q6_k` | 8 | 16777110 | `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | `blk.34.ffn_down_exps.weight` | Q6_K | 210 |

### G. Link-rate provenance (all MEASURED, none invented)

| link | rate | source |
|---|---|---|
| T3 local NVMe / disk image, cold read | 1.80 GB/s | `ANALYSE_389_nvme_expert_tier.md` §(b), `iflag=direct`, reproduced 3x; tier table `DESIGN_407_memory_tier_registry.md:135` |
| T4 remote rig-2 over 40G, NCCL-over-sockets | 2.07 GB/s | `NOTE_453_remote_expert_lane.md:9-10` / `INTEGRATION_R3_VALIDATION.md:5053`; tier table `DESIGN_407_memory_tier_registry.md:137` |
| T4 remote rig-2 over 40G, staged RDMA 1 MiB | 2.83 GB/s | tier table `DESIGN_407_memory_tier_registry.md:137` |
| T2 host RAM -> card, PCIe H2D pinned, gen4 x4 | 6.40 GB/s | `ANALYSE_393_ik_llama.md:301-304`; tier table `DESIGN_407_memory_tier_registry.md:136` |
| T2 host RAM -> card, PCIe H2D pinned, gen4 x8 | 13.00 GB/s | `ANALYSE_393_ik_llama.md:301-304`; tier table `DESIGN_407_memory_tier_registry.md:136` |
