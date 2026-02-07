In this repository, we provide essential files for a submitted paper:

## Nucleotide-dependent Structural Selection Governs c-Src Phosphorylation of Oncogenic KRas4B-G12D

by Huixia Lu<sup>*</sup>, Honglin Xu, Jordi Marti, Buyong Ma, and Jordi Faraudo

#### Simulating packages used in this work for conducting MD simulations:

##### AMBER20 utilizing the CHARMM36 force field 

# Highlights from this work:

This study reveals the molecular basis of c-Src’s nucleotide-dependent recognition of oncogenic KRas4B-G12D. Using extensive all-atom molecular dynamics simulations and Markov state models (34 μs total), the work shows that c-Src preferentially binds to the most populated conformations of GTP-bound KRas, while interacting mainly with rare, low-population states of the GDP-bound form. The analysis identifies specific coil-structured regions in the Src kinase domain (residues 298-303, 411-426, and 459-467) that mediate this selective interaction. These findings provide atomic-level insight into KRas-Src selectivity and suggest new strategies for designing peptides that selectively target the active, GTP-bound KRas4B-G12D in cancer.

# pyemma-markov-state-models

**start your MSM analysis**

##Alert! Conda has removed Pyemma 2.5.7!!!



```
conda create -n pyemma-env python=3.7.12
```

```
conda activate pyemma-env
```

```
python -m pip install pyemma==2.5.7
```
OBS: If your computer has a `gcc` version 13 or greater, most probably it wont be able to build pyemma 2.5.7 from sources. If that happen you can try your luck installing older gcc/g++ compiler (like `sudo apt install gcc-12 g++-12` and then provide that version via `CC` and `CXX` environment variables: `CC=gcc-12 CXX=g++-12 python -m pip install pyemma==2.5.7`).

```
python -m pip install pandas==0.25.3
```

```
python -m pip install notebook
```

#if you want to install MDAnalysis in this pyemma conda environment, then install MDAnalaysis=2.1.0 to meet the requirments of numpy=1.21.0 and python3.7:
```
(pyemma-env) huixia@SIMCON4: conda install -c conda-forge mdanalysis=2.1.0
```
#Maybe you also want to install sklearn and seaborn as well
```
python -m pip install seaborn
```
```
python -m pip install sklearn
```

Then you are ready to go!! Have fun!

##

**More details:**

1. Initial systems used to conduct short MD simulations are stored in Folder: **initial-systems**
2. In the Folder: **toy_data** you will find an example of aaa.txt which is part of the data set I have used to construct MSMs for system KRas4B-G12D-GTP (paper in preparation). I couldn't upload the whole data because of the size limit.
3. A walk through tutorial of how to construct MSMs is shown in file: **msm-analysis.ipynb**
4. PDB files of KRas4B-GTP and KRas4B-GDP were extracted from Macrostates by PyEMMA after constructing MSMs, the corresponding results are installed in Folders: **kras4b-g12d-gtp and kras4b-g12d-gdp**.
5. Final equilibrated structures of complex of Src-KRas in its two nucleotide states, obtained from 200 ns molecular dynamics simulations performed with NAMD and initialized from HADDOCK2.4 docking results for KRas-Src macrostates with HADDOCK scores above 130, are provided in the **HADDOCK_results folder**.
