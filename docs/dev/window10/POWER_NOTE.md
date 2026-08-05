# Power limits during window 10

The user restored the power limits to their record-era maxima at 17:49Z, while
ARM 1 was still loading weights and before any measured draw:

  GPU 0  RTX 3080  320 W   (was 200 W since 2026-08-05, memory note
  GPU 1  RTX 5090  525 W    "Power-Targets reduziert")
  GPU 2  RTX 3080  320 W

Every measured draw in this window therefore ran at the same limits as the
#845 / #424 record baseline, so no power caveat separates these arms from it.
`vram_power_series.csv` samples index;power.limit;power.draw;util;mem.free every
2 s for the whole window, so each arm's actual draw is evidence rather than an
assumption -- the limit being raised is not the same as the card taking it.
