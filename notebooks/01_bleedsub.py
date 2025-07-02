# %%
#!/usr/bin/env python
"""
01_bleedsub.py
Residual bleed-through subtraction (AF488, PE/594, AF647, AF750)
+ writes a NEW composite (…_clean.ome.tif)

Run:
    mamba activate ibex_py
    python notebooks/01_bleedsub.py 1110_SFO-Armd10-03
"""
import sys, tifffile, numpy as np, yaml
from pathlib import Path

# ---------------- User-tunable ---------------------------------------------
SCALE      = 0.03                          # % carry-over to subtract
FLUOR_CH   = [1, 2, 3, 4]                  # channels to clean (0 = DAPI)
BASE_LUT   = ['#FFFFFF', '#2ECC71', '#E67E22', '#9B59B6', '#2980B9']
# ---------------------------------------------------------------------------

ROI   = sys.argv[1]
root  = Path(__file__).resolve().parents[1]
raw   = root/'raw'
aldir = root/f'aligned/{ROI}';  aldir.mkdir(exist_ok=True, parents=True)

# ---------- helpers ---------------------------------------------------------
def read_zcyx(p):
    a = tifffile.imread(p)
    if a.ndim == 3: a = a[None]          # (C,Y,X) → (Z=1,C,Y,X)
    return a.astype(np.float32)

def mip5(p):                             # Z-max first 5 channels
    s = read_zcyx(p);  return s.max(axis=0)[:5]

# ---------- clean cycles ≥2 -------------------------------------------------
ref   = read_zcyx(raw/f"cycle01/{ROI}_cyc1.ome.tif")
prev  = ref.copy()
clean_paths = []

for reg in sorted(aldir.glob("cycle*_reg.ome.tif")):   # 2,3,4 …
    stk = read_zcyx(reg)
    for ch in FLUOR_CH:
        stk[:, ch] = np.clip(stk[:, ch] - SCALE*prev[:, ch], 0, None)
    out = reg.with_stem(reg.stem + "_clean")
    tifffile.imwrite(out, stk, ome=True, metadata={'Axes': 'ZCYX'})
    clean_paths.append(out)
    prev = stk.copy()
    print("✓ cleaned", out.name)

# ---------- new composite ---------------------------------------------------
stack = np.concatenate(
        [mip5(raw/f"cycle01/{ROI}_cyc1.ome.tif"),
         *[mip5(p) for p in clean_paths]], axis=0)         # (N*5,Y,X)

n_cycles = len(stack) // 5
lut      = BASE_LUT * n_cycles

comp = aldir/f"{stack.shape[0]}ch_{ROI}_clean.ome.tif"
tifffile.imwrite(comp, stack, ome=True, metadata={'Axes': 'CYX'})
yaml.dump({'colors': lut}, open(comp.with_suffix('.yml'), 'w'))

print("\n📦  new composite written →", comp)



