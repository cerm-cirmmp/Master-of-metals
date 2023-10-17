import numpy as np
from base.pdb_loader import load_pdb
from base.mmcif_loader import load_mmcif

import torch
from base.preprocess import get_distances
from base.features import get_features_from_aminoacid
#from base.CBP_dataloader import MetalBindingSite
from base.metal_binding_site import MetalBindingSite
import pathlib


def extract_potential_mbs(pdb_df, aas):
    pdb_CA = pdb_df[pdb_df['atom_name'] == 'CA']
    pdb_CB = pdb_df[pdb_df['atom_name'] == 'CB']
    pdb_CA = pdb_CA[pdb_CA['residue_name'].isin(aas)]  # dataframe of all CA coordinates for HIS, CYS, MET, ASP, GLU
    pdb_CB = pdb_CB[pdb_CB['residue_name'].isin(aas)]

    residues_ca_xyz = torch.from_numpy(pdb_CA[['x', 'y', 'z']].to_numpy())
    residues_cb_xyz = torch.from_numpy(pdb_CB[['x', 'y', 'z']].to_numpy())
    res_cacb = torch.cat((residues_ca_xyz, residues_cb_xyz), dim=0)

    distances = get_distances(residues_ca_xyz, residues_ca_xyz)   # distance matrix over all CAs
    distances_CACB = get_distances(res_cacb, res_cacb)

    mask = (distances > 3) & (distances <= 13)    # returns true or false for every matrix element

    POT_MBS_LIST = []
    for j_idx in range(mask.size(0)):
        aa_indices = mask[j_idx].nonzero().view(-1).tolist()    # for every row (CA), lists indices of true elements (aminoacids between 3 and 12 A)
        if len(aa_indices) >= 2:    # we need at least 3 aminoacids per MBS
            aa_indices = sorted(aa_indices + [j_idx])   # adding the first aminoacid of the group
            aa_indices = [str(x) for x in aa_indices]   # turning group to string (so I can set them later)
            POT_MBS_LIST.append('_'.join(aa_indices))

    POT_MBS_LIST = list(set(POT_MBS_LIST))   # aminoacid combinations appear only once

    return POT_MBS_LIST, pdb_CA, pdb_CB, distances_CACB, mask    # list contains indices, I'll need coordinates later


def subgroups(group, group_size):
    res = []
    lll = torch.combinations(torch.arange(len(group)), group_size)   # finds all combinations of a certain size for each group of close aminoacids

    for idxs in lll:
        sub_g = [group[i] for i in idxs]   # lll contains indices - these are used to generate the subgroups
        res.append(sub_g)

    return res


def create_mbs(protname, pdb_CA, pdb_CB, all_CACB_dist, aa_idxs):
    local_df_CA = pdb_CA.loc[aa_idxs]   # selects rows of potential ligands in the pdb dataframe
    ca_xyz = torch.from_numpy(local_df_CA[['x', 'y', 'z']].to_numpy())   # getting CA coordinates
    local_df_CB = pdb_CB.loc[aa_idxs]
    cb_xyz = torch.from_numpy(local_df_CB[['x', 'y', 'z']].to_numpy())

    cb_idxs = [x + len(pdb_CA) for x in aa_idxs]
    localCACB_idxs = aa_idxs + cb_idxs

    distances_CAandCB = all_CACB_dist[:, localCACB_idxs][localCACB_idxs, :]
    distances_CAandCB = torch.exp(-distances_CAandCB / 15)

    #distances_CAandCB, features = process_features(ca_xyz, cb_xyz, aminoacids=local_df_CA['residue_name'].tolist())

    features = get_features_from_aminoacid(local_df_CA['residue_name'].tolist())

    mbs = MetalBindingSite(features=features,
                           adj=distances_CAandCB,
                           sequence=local_df_CA['name_chain_seq'].tolist(),
                           protein=protname,
                           coord_CA=ca_xyz,
                           coord_CB=cb_xyz)
    return mbs


def check_validity(subgroup, matrix):   # checks whether all residues in the subgroup are within distance requirements
    submatrix = matrix[:, subgroup][subgroup, :]
    upper_triangular = submatrix[np.triu_indices(len(subgroup), k=1)]
    return all(upper_triangular)    # the elements of the upper triangular submatrix should be all True


def new_load_potential_sites(pdb_file, aminoacids, chain=None):
    prot_name = pathlib.Path(pdb_file).name.split('.')[0]

    if pdb_file.endswith(".pdb"):
        print("is a pdb")
        pdb_df = load_pdb(pdb_file)

    elif pdb_file.endswith(".cif"):
        print("is a cif")
        pdb_df = load_mmcif(pdb_file)

    if chain!= None:
        pdb_df = pdb_df[pdb_df['chain_id']==chain]

    pdb_df['name_chain_seq'] = pdb_df['residue_name'] + "_" + pdb_df['chain_id'] + "_" + pdb_df['residue_seq_num']  # new dataframe column

    aa_groups, pdb_CA, pdb_CB, CA_dist_matrix, mask = extract_potential_mbs(pdb_df, aminoacids)

    potential_mbs_list = []
    ALL_GROUPS = []

    for g in aa_groups:
        g_list = g.split('_')
        #print(g_list)

        if len(g_list) >= 6:
            subgroups3 = subgroups(g_list, group_size=3)
            subgroups4 = subgroups(g_list, group_size=4)
            subgroups5 = subgroups(g_list, group_size=5)
            all_subgroups = subgroups3 + subgroups4 + subgroups5
        elif len(g_list) == 5:
            subgroups3 = subgroups(g_list, group_size=3)
            subgroups4 = subgroups(g_list, group_size=4)
            all_subgroups = subgroups3 + subgroups4
        elif len(g_list) == 4:
            subgroups3 = subgroups(g_list, group_size=3)
            all_subgroups = subgroups3
        else:
            all_subgroups = [g_list]

        ALL_GROUPS += all_subgroups

    ALL_GROUPS = map(lambda x: '|'.join(x), ALL_GROUPS)
    ALL_GROUPS = list(set(ALL_GROUPS))   # combinations in subgroups need to be unique
    ALL_GROUPS = list(map(lambda x: x.split('|'), ALL_GROUPS))

    pdb_CA = pdb_CA.reset_index()
    pdb_CB = pdb_CB.reset_index()

    for subg in ALL_GROUPS:
        subg = [int(x) for x in subg]   # indices need to be turned back into integers
        if check_validity(subg, mask):
            try:
                MBS = create_mbs(prot_name, pdb_CA, pdb_CB, CA_dist_matrix, subg)
            except KeyError:
                print('problem with pdb file - missing CB coordinates')
            else:
                potential_mbs_list.append(MBS)

    return potential_mbs_list
