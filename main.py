# This is my main script
import sys, os, pathlib
sys.path.insert(0,pathlib.Path(__file__).parent)

import pathlib
import pandas as pd
from base.parallel_predictions import mbs_search
from base import configs
from base.utils import load_classifier, get_MBS_V2

import argparse


"""
df = pd.read_csv("base/training_set_F0.csv")
print(df)
training_sites = []

for _, row in df.iterrows():

    sites = list(set( row['sites'].split(';')) )

    sites = [f'{x}.site.pdb' for x in sites]

    training_sites += sites


with open('trainingset_zinc.txt', 'w') as fp:
    for item in training_sites:
           fp.write(f'{item}\n')
"""


parser = argparse.ArgumentParser(description="MasterOfMetals")

parser.add_argument(
    "--trainingset_dir",
    help="The path of the directory containing the known MBSs, downloaded from metalPDB")

parser.add_argument(
    "--trainingset_list",
    help="The path of the file containing the list of the known sites")


parser.add_argument(
    "--to_predict_dir",
    help="The path of the dir containing the input structures, for which we want to predict the MBSs")

parser.add_argument(
    "--to_predict_list",
    help="The file containing the list of the structures we want to process")


args = parser.parse_args()



def Main(args):

    trainingset_list = args.trainingset_list # 'trainingset_zinc.txt'
    trainingset_dir = args.trainingset_dir #  "/mnt/disk4Tb/Vincenzo/data/site_files"

    to_predict_list = args.to_predict_list # 'alphafold_structs.txt'
    to_predict_dir = args.to_predict_dir # '/mnt/disk4Tb/milazzo/alphafold_staph'

    print(trainingset_list)

    trainingset_dir = pathlib.Path(trainingset_dir)
    configs.SITES_PATH = trainingset_dir

    # OUTPUT PATH
    pathlib.Path("output").mkdir(parents=True, exist_ok=True)
    output_path = 'output'

    aas_list = ['HIS', 'CYS', 'ASP', 'GLU']

    # READ THE TRAINING SET LIST
    with open(trainingset_list, 'r') as fp:
        trainingset_list = fp.readlines()
        trainingset_list = [x.strip() for x in trainingset_list]


    # LOAD THE TRAINING SITES
    real_mbs = []
    for site_name in trainingset_list:
        pprot = site_name.split('.')[0].split('_')[0]
        #mbs = get_MBS_V2(pprot, configs.SITES_PATH.joinpath(site_name))
        mbs = get_MBS_V2(pprot, trainingset_dir.joinpath(site_name))
        if mbs != None:
            real_mbs.append(mbs) #


    # PATH OF THE INPUT PROTEOME #
    PROTEINS_PATH = pathlib.Path(to_predict_dir)  #  args.to_predict_path


    # READ THE INPUT STRUCTURES LIST
    with open(to_predict_list, 'r') as fp:
        filename_list = fp.readlines()
        filename_list = [x.strip() for x in filename_list]

    # LOAD THE TRAINED GRAPH NEURAL NETWORK
    net = load_classifier('base/trained_model_F0.pth')
    net.eval()

    mbs_search(
        net,
        PROTEINS_PATH,
        filename_list,
        output_path,
        real_mbs,
        aas_list)


if __name__ == '__main__':
    print(args)
    Main(args)
    print("The end")
