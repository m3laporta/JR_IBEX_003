#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

# ---- CHANGE ONLY THESE LINES --------------------------------------
ROI_ID = "1110_SFO-Armd10-03"              # ← your new tissue section
root   = Path("/Volumes/impl-k/Mikey_filer/Projects/JR_IBEX_003")

CYCLE_PATHS = { 
    1: root/"raw/cycle01/1110_SFO-Armd10-03_cyc1.ome.tif",
    2: root/"raw/cycle02/1110_SFO-Armd10-03_cyc2.ome.tif",
    3: root/"raw/cycle03/1110_SFO-Armd10-03_cyc3.ome.tif",
    4: root/"raw/cycle04/1110_SFO-Armd10-03_cyc4.ome.tif",
    # add 5,6… when available
}

OUT_DIR = root/f"aligned/{ROI_ID}"
TX_DIR  = root/f"transforms/{ROI_ID}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TX_DIR.mkdir(parents=True,  exist_ok=True)
print("🗂  Output →", OUT_DIR)


# In[2]:


import tifffile, numpy as np, SimpleITK as sitk
from pystackreg import StackReg        # conda-forge build
from skimage.registration import phase_cross_correlation

def read_stack(p):                      # always returns (Z,C,Y,X)
    a = tifffile.imread(p)
    if a.ndim == 4: return a
    if a.ndim == 3 and a.shape[0] <= 5: return a[None]        # C,Y,X
    if a.ndim == 3 and a.shape[-1] <= 5: return a[...,None].transpose(3,2,0,1)
    if a.ndim == 2:  return a[None,None]
    raise ValueError(f"shape {a.shape} not supported")

mip  = lambda z: z.max(axis=0)

def pad(a,b):
    H,W = max(a.shape[0],b.shape[0]), max(a.shape[1],b.shape[1])
    padder = lambda x: np.pad(x, ((0,H-x.shape[0]), (0,W-x.shape[1])))
    return padder(a), padder(b)

def register(ref_mip, mov_path):
    mov = mip(read_stack(mov_path)[:,0])
    ref_p, mov_p = pad(ref_mip, mov)
    M = StackReg(StackReg.RIGID_BODY).register(ref_p, mov_p)
    tx = sitk.AffineTransform(2)
    tx.SetMatrix(M[:2,:2].ravel()); tx.SetTranslation(M[:2,2])
    return tx

def save_registered_stack(src_path, tx, dst_path):
    stk = read_stack(src_path)
    refY, refX = mip(read_stack(CYCLE_PATHS[1])[:,0]).shape
    out = np.empty_like(stk, dtype=np.float32)
    for z in range(stk.shape[0]):
        for c in range(stk.shape[1]):
            img = sitk.GetImageFromArray(stk[z,c])
            res = sitk.Resample(img, img, tx, sitk.sitkLinear, 0., img.GetPixelID())
            out[z,c] = sitk.GetArrayFromImage(res)[:refY,:refX]
    tifffile.imwrite(dst_path, out, ome=True, metadata={'Axes':'ZCYX'})


# In[4]:


ref_stack = read_stack(CYCLE_PATHS[1])
ref_mip   = mip(ref_stack[:,0])                 # DAPI

for cyc, src in CYCLE_PATHS.items():
    if cyc == 1:                                # reference
        continue
    print(f"→ registering cycle {cyc}")
    tx = register(ref_mip, src)
    tx_matrix = np.array(tx.GetMatrix()).reshape(2, 2)  # 2x2
    tx_translation = np.array(tx.GetTranslation()).reshape(2, 1)  # 2x1
    tx_affine = np.hstack([tx_matrix, tx_translation])  # 2x3
    np.savetxt(TX_DIR/f"cyc{cyc}_to_cyc1.txt", tx_affine)
    save_registered_stack(src, tx, OUT_DIR/f"cycle0{cyc}_reg.ome.tif")

print("✅ all cycles aligned")


# In[5]:


from matplotlib import pyplot as plt, colors as mcolors

reg_paths = sorted(OUT_DIR.glob("cycle*_reg.ome.tif"))
stack     = np.concatenate([mip(read_stack(p)) for p in [CYCLE_PATHS[1], *reg_paths]], axis=0)

n_cycles  = len(stack)//5
tif_out   = OUT_DIR/f"{stack.shape[0]}ch_{ROI_ID}.ome.tif"
tifffile.imwrite(tif_out, stack, ome=True, metadata={'Axes':'CYX'})

# save LUT
base = ['#FFFFFF','#2ECC71','#E67E22','#9B59B6','#2980B9']
lut  = base * n_cycles
yaml_path = tif_out.with_suffix('.yml')
import yaml; yaml.dump({'colors': lut}, open(yaml_path,'w'))

# thumbnail
rgb = np.zeros(stack.shape[1:]+(3,), float)
for img,col in zip(stack,lut):
    r,g,b = mcolors.to_rgb(col); norm = img/img.max() if img.max() else img
    rgb  += np.dstack([norm*r, norm*g, norm*b])
plt.imsave(tif_out.with_suffix('.png'), np.clip(rgb,0,1))

print("📦 composite saved →", tif_out)

