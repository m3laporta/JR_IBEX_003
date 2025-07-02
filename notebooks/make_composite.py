# notebooks/make_composite.py
import sys, tifffile, numpy as np, yaml
from pathlib import Path

roi   = sys.argv[1]                 # e.g. 1110_SFO-Armd10-03
root  = Path(__file__).resolve().parents[1]   # JR_IBEX_003
aligned_dir = root/f"aligned/{roi}"

reg_paths = sorted(aligned_dir.glob("cycle*_reg.ome.tif"))
if not reg_paths:
    sys.exit(f"[❌] no registered stacks found in {aligned_dir}")

def mip5(p):
    a = tifffile.imread(p);             # (Z,C,Y,X)
    if a.ndim == 3: a = a[None]
    return a.max(axis=0).astype(np.float32)[:5]

stack = np.concatenate([mip5(root/f"raw/cycle01/{roi}_cyc1.ome.tif"),
                        *map(mip5, reg_paths)], axis=0)

out = aligned_dir/f"{stack.shape[0]}ch_{roi}.ome.tif"
tifffile.imwrite(out, stack, ome=True, metadata={'Axes':'CYX'})

base = ['#FFFFFF','#2ECC71','#E67E22','#9B59B6','#2980B9']
yaml.dump({'colors': base*len(reg_paths+['dummy'])},
          open(out.with_suffix('.yml'),'w'))
print("✓ composite written →", out)
