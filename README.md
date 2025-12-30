# ILs_CO2AbsorptionProject
## About
This is the code for paper "Multiple ensemble graph neural networks for the high-throughput screening of ionic liquids for carbon capture: Modeling study and experimental validation"<br>
In this study, five GNN models, including MPNN, GAT, GIN, PNA, and Attentive FP, were developed to predict the CO₂ absorption capacity and viscosity of ILs. To further enhance predictive accuracy, four ensemble strategies, namely Outlier-removed statistical averaging(ORSA), Weighted averaging (WA), stacking ensemble based on linear regression (stack_LR), and stacking ensemble based on extremely randomized trees (stack_ET), were implemented. Furthermore, traditional ML models, including XGBoost and random forest (RF), paired with classic molecular descriptors (RDKit descriptors and Morgan fingerprints), were benchmarked to contextualize the advantages of the GNN ensemble frameworks.
## Requirements
The code of ILs_CO2AbsorptionProject are implemented and tested under the following development environment:
| package  | version|
| ---------- | -----------|
| python   | 3.9.19   |
| pytorch   | 2.4.0   |
| torch-geometric   | 2.5.3   |
| rdkit   | 2024.03.5   |
| scikit-learn   | 1.5.1   |
|  numpy   | 1.26.4   |
| pandas   | 1.5.3   |
## Usage
1. Data splitting <br>
   cd my_code/data_process/ //python
   python DataSplit.py --dataset CO2_capacity --group-key smiles --index-dir ../../data/indexs/CO2_capacity //python
   
3. 
   
4. James Monroe
5. John Quincy Adams


