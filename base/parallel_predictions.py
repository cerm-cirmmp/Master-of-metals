# predicting metal binding sites

import pandas as pd
import pathlib
import torch
import os
import urllib
#from base.find_proteins_sitesV3 import load_potential_sites
from itertools import permutations
import numpy as np
import time
from base.extract_potential_sites import new_load_potential_sites
from base.postprocess import get_metal_coord
from base import configs
from base.pdb_loader import load_pdb
from base.utils import df_to_pdb
import shutil
import sys



def make_tensor(mbs_list):

    tensor_3res, tensor_4res, tensor_5res = [], [], []
    sites_3res, sites_4res, sites_5res = [], [], []
    pattern_3res, pattern_4res, pattern_5res = [], [], []

    for mbs_t in mbs_list:
        mbs_t_size = len(mbs_t.sequence)
        ptrn = '_'.join([x.split('_')[0] for x in mbs_t.sequence])

        if mbs_t_size == 3:
            tensor_3res.append(mbs_t.real_adj)
            sites_3res.append(mbs_t)
            pattern_3res.append(ptrn)
        elif mbs_t_size == 4:
            tensor_4res.append(mbs_t.real_adj)
            sites_4res.append(mbs_t)
            pattern_4res.append(ptrn)
        elif mbs_t_size == 5:
            tensor_5res.append(mbs_t.real_adj)
            sites_5res.append(mbs_t)
            pattern_5res.append(ptrn)
        else:
            print(f"ERROR: number of ligands must be 3, 4 or 5, {mbs_t.protein} has {mbs_t_size}")

    tensor_3res = torch.stack(tensor_3res, dim=0)
    tensor_4res = torch.stack(tensor_4res, dim=0)
    pattern_3res = np.array(pattern_3res)
    pattern_4res = np.array(pattern_4res)

    if tensor_5res != []:
        tensor_5res = torch.stack(tensor_5res, dim=0)
        pattern_5res = np.array(pattern_5res)

    return tensor_3res, sites_3res, pattern_3res, tensor_4res, sites_4res, pattern_4res, tensor_5res, sites_5res, pattern_5res


def download_structures(prot_path, protein):
    if '_' in protein:
        protein = protein.split('_')[0]
    full_prot_path = prot_path.joinpath(f'{protein}.pdb')
    if full_prot_path.exists():
        print(f'{protein} already exists!')
    else:
        print(f'{protein} not found, downloading it...')
        urllib.request.urlretrieve(f'http://files.rcsb.org/download/{protein}.pdb', prot_path.joinpath(f'{protein}.pdb'))


def matrix_rmsd_parallel(matrix1, matrix2, lista_siti, input_pdb, tensor_patterns, pot_pattern):    # calculates a measure of similarity between 2 MBS
    # matrix2 is the huge tensor
    training_matrix = np.array(matrix2)
    ptrn_idxs = (tensor_patterns == pot_pattern).nonzero()[0]

    #print("///////////////////////////")
    #print(tensor_patterns[ptrn_idxs])
    #print(pot_pattern)


    if len(ptrn_idxs) == 0:
        return None

    # extract the sub tensor (matrices having binding pattern = pot_pattern)
    training_matrix = training_matrix[ptrn_idxs]
    lista_siti = [lista_siti[j] for j in ptrn_idxs]

    assert len(training_matrix) == len(lista_siti)

    # permutation combiantions
    #l = list(permutations(range(int((len(matrix1)) / 2))))
    l = [list(range(int((len(matrix1)) / 2)))] # NO PERMUTATIONS FOR NOW

    # name list of the training pdbs
    protein_names = [x.protein for x in lista_siti]

    #print("protein names", protein_names)

    # to avoid comparing same pdbs
    same_pdb_idxs = [j for j, site in enumerate(protein_names) if site == input_pdb]

    #print("same_pdb_idxs", same_pdb_idxs)

    rmsd_list = []

    for permutation in l:    # I want to minimize the rmsd -> permutations of potential mbs distance matrix
        permutation = list(permutation)
        test_matrix = np.array(matrix1)
        cb_idxs = [x + int((len(test_matrix)) / 2) for x in permutation]  # maintains correspondence between CA and CB
        cacb_idxs = permutation + cb_idxs
        test_matrix[list(range(len(test_matrix))), :] = test_matrix[cacb_idxs, :]  # exchanging rows
        test_matrix[:, list(range(len(test_matrix)))] = test_matrix[:, cacb_idxs]  # exchanging columns
        
        #diff = (test_matrix-training_matrix)**2
        #ave = np.mean(diff.reshape((training_matrix.shape[0], -1)), axis=1)
        #rmsd = np.sqrt(ave)

        diff = np.abs(test_matrix-training_matrix)
        rmsd = np.mean(diff.reshape((training_matrix.shape[0], -1)), axis=1)

        #print("rmsd", rmsd)

        for same_pdb_idx in same_pdb_idxs:  # avoiding comparison between sites on the same protein
            rmsd[same_pdb_idx] = 10000.     # manually setting rmsd

        min_idx = np.argmin(rmsd)

        min_rmsd = rmsd[min_idx]
        clst_site = lista_siti[min_idx]

        rmsd_list.append((min_rmsd, clst_site, cacb_idxs))

    #print(rmsd_list)

    maxrmsd = 1000
    for (min_rmsd2, clst_site2, idxs) in rmsd_list:  # finding the closest site out of all permutations
        if min_rmsd2 < maxrmsd:
            maxrmsd = min_rmsd2
            min_rmsd, clst_site, permutation_idxs = min_rmsd2, clst_site2, idxs

    #print(min_rmsd, clst_site, permutation_idxs)
    return min_rmsd, clst_site, permutation_idxs

"""
def mbs_prediction(output_directory, training_mbs_list, test_mbs_list, pdb_path, aas):

    n = len(test_mbs_list)
    start = time.time()

    tensor_3, sites_3, tensor_4, sites_4, tensor_5, sites_5 = make_tensor(training_mbs_list)

    for j, test_site in enumerate(test_mbs_list):
    #for test_site in ['4ahb_2', '1xrf_1', '5xkq_2', '2elp_1', '4tvr_2', '3rza_4']:
    #for test_site in ['4tvr_2']:

        print(f"progress: {j}/{n}")
        prot = test_site.split('_')[0]
        #prot = test_site.protein
        if os.path.exists(f'{output_directory}/{prot}_pred.csv'):   # checking whether predictions for a certain structure already exist
            print(f'metal sites for {prot} already predicted!')
        else:
            download_structures(pdb_path, f'{prot}')
            print(f'generating potential binding sites for {prot}')
            pot_bind_sites = load_potential_sites(f'{prot}', pdb_path, aa_residues=aas, create_pickle=True)
            predicted_mbs = []
            print(f'predicting metal binding sites for {prot}')
            if pot_bind_sites != None:
                for i in pot_bind_sites:
                    size = (len(i.real_adj))/2   # adj matrices need to have the same dimensions to be compared

                    if size == 3:
                        tensor_P, lista = tensor_3, sites_3
                    elif size == 4:
                        tensor_P, lista = tensor_4, sites_4
                    elif size == 5:
                        tensor_P, lista = tensor_5, sites_5
                    else:
                        print(f'ERROR : potential binding site has {size} ligands')

                    closest_dist, closest_real_mbs = matrix_rmsd_parallel(i.real_adj, tensor_P, lista, prot)

                    if closest_dist < 1000:   # this is the rmsd threshold
                        pred_sequence = ';'.join(i.sequence)
                        real_sequence = ';'.join(closest_real_mbs.sequence)
                        predicted_mbs.append(
                            (i.protein, pred_sequence, closest_dist, closest_real_mbs.site_name, real_sequence))

                predicted_mbs.sort(key=lambda x: x[2])   # sorting by rmsd
                predicted_mbs_df = pd.DataFrame(predicted_mbs,
                                                columns=['protein', 'predicted pattern', 'rmsd', 'closest real mbs',
                                                         'real mbs pattern'])
                predicted_mbs_df.to_csv(f'{output_directory}/{prot}_pred.csv', index=None)
                print(f'predictions for {prot} successfully created!')

    end = time.time()
    print('time elapsed :', end - start)
"""

def mbs_search(net, pdb_path, file_list, output_directory, training_mbs_list, aas):

    out_dir = pathlib.Path(output_directory)

    n = len(file_list)
    start = time.time()

    # create tensors containing the distance matrices of the training mbs
    tensor_3, sites_3, ptrns_3, tensor_4, sites_4, ptrns_4, tensor_5, sites_5, ptrns_5 = make_tensor(training_mbs_list)

    final_mbs = []

    # iterate over the input sites
    for j, pdb_file in enumerate(file_list):
        print(f"progress: {j + 1}/{n}")
        prot_name = pdb_file.split('.')[0]

        chain = None

        print(f'generating potential binding sites for {prot_name}')
        potential_mbs_list = new_load_potential_sites(f'{pdb_path}/{pdb_file}', aas, chain=chain)
        print(len(potential_mbs_list), 'potential MBSs')

        # Prediction of the metal binding sites
        print(f'predicting metal binding sites for {prot_name}')
        if potential_mbs_list != None:
            for potential_mbs in potential_mbs_list:

                hidden, net_out = net(potential_mbs.x, potential_mbs.adj.float())

                if net_out[1] < 0.6: # P(mbs)
                    continue

                # residues pattern of the potential mbs
                pot_pattern = '_'.join([x.split('_')[0] for x in potential_mbs.sequence])

                size = (len(potential_mbs.real_adj)) / 2  # adj matrices need to have the same dimensions to be compared

                if size == 3:
                    tensor_P, lista, ptrns = tensor_3, sites_3, ptrns_3
                elif size == 4:
                    tensor_P, lista, ptrns = tensor_4, sites_4, ptrns_4
                elif size == 5:
                    tensor_P, lista, ptrns = tensor_5, sites_5, ptrns_5
                else:
                    print(f'ERROR : potential binding site has {size} ligands')

                if tensor_P != []:
                    results = matrix_rmsd_parallel(potential_mbs.real_adj, tensor_P, lista, prot_name, ptrns, pot_pattern)
                    if results == None:
                        continue
                    else:
                        closest_dist, closest_real_mbs, coord_idxs = results

                else:
                    closest_dist = 1000

                if closest_dist < 0.35:  # this is the rmsd threshold
                    pot_coord_ca_cb = np.concatenate((potential_mbs.coord_CA, potential_mbs.coord_CB), axis=0)
                    ordered_coord = pot_coord_ca_cb[coord_idxs]  # potential mbs coordinates ordered according to best permutation
                    known_coord_ca_cb = np.concatenate((closest_real_mbs.coord_CA, closest_real_mbs.coord_CB), axis=0)
                    clst_metal_coord = closest_real_mbs.metals[['x', 'y', 'z']].to_numpy()

                    # the closest real mbs is superimposed to the predicted one
                    new_metal_coord, new_known_coord, rot, tran = get_metal_coord(ordered_coord, known_coord_ca_cb, clst_metal_coord)

                    closest_known_site_path = configs.SITES_PATH.joinpath(closest_real_mbs.site_name)

                    site_df, _ = load_pdb(closest_known_site_path, is_site=True, return_metals=True, remove_metals=False)

                    loaded_site_coord = site_df[['x', 'y', 'z']].to_numpy()

                    aligned_site = np.dot(loaded_site_coord, rot) + tran

                    site_df[['x', 'y', 'z']] = aligned_site.round(3)

                    aligned_site_pdb = df_to_pdb(site_df)
                    new_name = str(np.round(closest_dist, 3)) + '#' + '_'.join(potential_mbs.sequence) + '=' + closest_real_mbs.site_name

                    out_dir.joinpath(prot_name).mkdir(exist_ok=True)
                    if not out_dir.joinpath(prot_name).joinpath(f'{prot_name}.pdb').exists():
                        shutil.copy(f'{pdb_path}/{pdb_file}', out_dir.joinpath(prot_name))

                    with open(out_dir.joinpath(prot_name).joinpath(new_name), 'w') as f:
                        f.writelines(aligned_site_pdb)

                    final_mbs.append((prot_name, ';'.join(potential_mbs.sequence), closest_dist,
                                      closest_real_mbs.site_name, ';'.join(closest_real_mbs.sequence)))

            print(f'predictions for {prot_name} successfully created!')

    final_mbs_df = pd.DataFrame(final_mbs,
                                    columns=['protein', 'predicted pattern', 'rmsd', 'closest real mbs',
                                             'real mbs pattern'])
    final_mbs_df.to_csv(f'{output_directory}/output_summary.csv', index=None)
    end = time.time()
    print('time elapsed :', end - start)
