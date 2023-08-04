import pandas as pd
import numpy as np
import sys
from base import configs
import torch
from base.model import MBSClassifier
#from base.CBP_dataloader import MetalBindingSite
from base.metal_binding_site import MetalBindingSite


def df_to_pdb(pdb_df):
    columns = pdb_df.columns.tolist()
    pdb_df['character'] = pdb_df['character'].replace(np.nan, " ")
    FILE = []

    # f = open("demofile3.pdb", "w")
    d = {'record_name': 6,
         'atom_number': 5 + 1,
         'atom_name': 4,
         'character': 1,
         'residue_name': 3 + 1,
         'chain_id': 1,
         'residue_seq_num': 4,
         'x': 8 + 4,
         'y': 8,
         'z': 8,
         'occupancy': 6,
         'temp_factor': 6,
         'element_symbol': 2 + 10}

    for idx, r in pdb_df.iterrows():
        stringa = ""
        for col in columns:
            if r[col] != np.nan:

                if col in ['x', 'y', 'z']:
                    sub_string = str(np.round(r[col], 3))
                else:
                    sub_string = str(r[col])

                # print(sub_string, len(sub_string))

                sub_string_len = len(sub_string)

                space_to_add = d[col] - sub_string_len

                if col in ['residue_seq_num', 'x', 'y', 'z', 'occupancy', 'temp_factor', 'element_symbol']:
                    stringa += " " * space_to_add + sub_string
                else:
                    stringa += sub_string + " " * space_to_add

        #print(stringa)
        FILE.append(stringa + '\n')

    return FILE


"""
df = pd.read_csv("ttt.csv", index_col=None)
towrite = df_to_pdb(df)


f = open("demofile4.pdb", "w")
f.writelines(towrite)
f.close()
"""

def load_classifier(model_name):
    # LOAD CLASSIFIER
    N_FEAT = configs.NUM_FEATURES
    model = MBSClassifier(nfeat=N_FEAT, n_out=2, dropout=0.1)
    #model.load_state_dict(torch.load("trained_model"))
    model.load_state_dict(torch.load(model_name))
    model.eval()
    return model



def get_MBS_V2(prot, site_path):

    debug = False

    try:

        # CREATE THE mbs OBJECT ASSOCIATED TO THE "REAL SITE"
        mbs = MetalBindingSite(site_path=site_path,
                               site_name=site_path.name,
                               protein=prot, debug=debug)
        mbs_s = [x.split("_")[0] for x in mbs.sequence]

        return mbs
    except Exception as ex:
        print("--- Problem with mbs object creation")
        print(ex)
        return None