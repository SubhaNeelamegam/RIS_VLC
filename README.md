# RIS_VLC
RIS Group Partitioning for Spatial Modulation in  Visible Light Communication
Requires: python3, numpy, scipy, matplotlib, pillow

Run in this order (same folder):
  python3 run.py     # all experiments -> res.json, A_*.npy   (about 2-3 min)
  python3 figs.py    # figures f1..f8.png into ./figs  (create folder: mkdir figs)

Files
  eng.py   exact ML BER for the scalar constellation, partitions
           (balanced, random, unequal-size, Stage 1, Stage 2), 16-PAM, RIS-GSM
  run.py   experiments: BER vs SNR, table, union bound, Monte Carlo check,
           GSM, stagger ratio r, element failures, gain-estimate error
  figs.py  plots used in the paper
