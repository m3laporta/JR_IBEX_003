
---

## **JR\_IBEX\_003: Immunofluorescence Multi-Cycle Alignment & Composite Pipeline**

### **1. Clone the Repository**

```bash
# On your new computer:
git clone git@github.com:m3laporta/JR_IBEX_003.git
cd JR_IBEX_003
```

---

### **2. Set Up the Python Environment**

```bash
# Install mamba or conda if needed (optional, if not already installed)
# Then create your environment from the yml file:
mamba env create -f notebooks/ibex_py_env.yml
# OR if using conda:
conda env create -f notebooks/ibex_py_env.yml

# Activate the environment
mamba activate ibex_py
# OR
conda activate ibex_py
```

---

### **3. Prepare Your Data**

* Place your raw OME-TIFFs in the expected directory (e.g., `raw/cycle01/`, `raw/cycle02/`, etc.)
  *File naming should follow the pattern: `raw/cycleXX/SAMPLEID_cycX.ome.tif`*

---

### **4. Register (Align) Cycles for a Sample**

```bash
# From inside the JR_IBEX_003 directory, run:
python notebooks/00_alignment.py SAMPLE_ID
# Example:
python notebooks/00_alignment.py 1110_SFO-Armd10-03
```

* This will output registered/cleaned OME-TIFFs in `aligned/SAMPLE_ID/`.

---

### **5. (Optional) Run Bleed-through & Background Correction**

```bash
python notebooks/01_bleedsub.py SAMPLE_ID
# Output: cleaned/bleed-subtracted images in aligned/SAMPLE_ID/
```

---

### **6. Make Composite OME-TIFF**

```bash
python notebooks/make_composite.py SAMPLE_ID
# Output: aligned/SAMPLE_ID/20ch_SAMPLE_ID.ome.tif (or 25ch for 5 cycles, etc.)
```

---

### **7. Visualize/Analyze in QuPath, napari, or your tool of choice**

* Open the composite .ome.tif in **QuPath** for downstream analysis, cell detection, annotation, etc.
* The colors and LUTs are assigned by the scripts; refer to the provided YAML file for channel-color mapping.

---

## **Troubleshooting / Notes**

* **If you get `Permission denied (publickey)` on git:**
  Make sure you have added your SSH public key to your [GitHub SSH settings](https://github.com/settings/keys).
* **If you get `not a git repository`:**
  Double-check you’re in the JR\_IBEX\_003 project directory.

---

### **Quick Reference**

* `notebooks/ibex_py_env.yml` – Python dependencies for this workflow.
* `notebooks/00_alignment.py` – Registration/alignment script.
* `notebooks/01_bleedsub.py` – Optional bleed-through/background correction.
* `notebooks/make_composite.py` – Composite multi-channel TIFF creation.

---

**For detailed protocol and options, see the in-line comments in each script.**


