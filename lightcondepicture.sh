  # Static paper figure — 9×9 symmetric, 7 explicit snapshots
  python3 make_lightcone_figure.py \
    ../extended_20260514/mi100/9x9_chi020_n50.log \
    -o fig_lightcone.png \
    --layers 1 5 10 15 20 30 40

  # GIF — 9×9 symmetric spreading (the showpiece)
  python3 make_gif.py \
    ../extended_20260514/mi100/9x9_chi020_n50.log \
    -o spreading_9x9.gif --fps 4

  # GIF — 4×4 chi=80 (asymmetric + trapped layer at L17)
  python3 make_gif.py \
    ../extended_20260514/mi100/4x4_chi080.log \
    -o spreading_4x4_chi80.gif --fps 3

  # GIF — side-by-side grid-size comparison
  python3 make_gif.py \
    ../extended_20260514/mi100/4x4_chi050.log \
    ../extended_20260514/mi100/5x5_chi060_n30.log \
    ../extended_20260514/mi100/9x9_chi020_n50.log \
    -o spreading_comparison.gif --fps 3
  
