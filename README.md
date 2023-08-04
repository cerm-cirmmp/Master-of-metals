# Master-of-metals
Tool for the prediction of metal binding sites in protein structures

This code requires Pytorch, Pandas, Numpy libraries installed.

To run the code set the following input arguments:
* --trainingset_list  "a file containing the list of sites downloaded from MetalPDB"
* --trainingset_dir   "the path of the directory contanining the sites downloaded from MetalPDB"
* --to_predict_list   "a file containing the list of input structures to process"
* --to_predict_dir    "the path of the directory containing the input structures"



python main.py --trainingset_list [...]  --trainingset_dir [...]  --to_predict_list [...] --to_predict_dir [...]
