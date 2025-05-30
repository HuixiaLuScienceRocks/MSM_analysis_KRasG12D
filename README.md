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
6. Results of Docking simualtion by HADDOCK2.4 are provided in Folder: **HADDOCK_results**
