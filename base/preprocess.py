"""
Come estrarre i potenziali siti dalle proteine???

- Abbiamo delle statistiche sulle distanze ...

dato un certo aminoacido.. che si fa???
* Vediamo a quanti aminoacidi è connesso che rispettano le statistiche ...
* Tra tutti i risultanti, prendiamo tutti i gruppi a 3,4,5,S-1

- Calcola matrice di adiacenza
- Per site_size S = [3,...,6]
-- Estrai gli indici degli aminoacidi che 'almeno' S aminoacidi connessi
(la cui distanza rispetta le statistiche)
-- Per quelli che ne hanno di più, raggruppali a gruppi di S

"""


import torch
import sys
import os
import numpy as np
from base.pdb_loader import load_pdb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def combinations(x, group_size=3, keep_dim=False):

    dim_1 = x.size(1)

    # index combinations

    idxs = torch.combinations(torch.arange(dim_1), group_size)

    # combinations over rows
    x = x.T[idxs]

    # axes permutations
    x = x.permute(2, 0, 1)

    #x = torch.squeeze(x, dim=2)

    if keep_dim is False:
        x = x.reshape(x.size(0)*x.size(1), -1)

    return x



def get_distances(points_A, points_B):

    # squared difference beetwen all points
    dd = (points_A[:, None, :] - points_B[None, :, :]) ** 2  # (N x k x 3)

    # somma lungo le componeneti
    dd = dd.sum(-1)  # N x K

    # matrice delle distanze
    dd = torch.sqrt(dd)  # N x K

    return dd



def get_potential_sitesOLD(pdb_df, site_size):
    import sys
    aas = ['CYS', 'HIS', 'GLU', 'ASP']

    # select all C-alpha rows
    pdb_CA = pdb_df[pdb_df['atom_name']=='CA']

    # filter aminoacid 'CYS', 'HIS', 'GLU', 'ASP'
    pdb_CA = pdb_CA[pdb_CA['residue_name'].isin(aas)]

    sequence = (pdb_CA['residue_name']+"_"
                +pdb_CA['chain_id']+"_"
                +pdb_CA['residue_seq_num']
                ).tolist()

    seq = pdb_CA['residue_name'].tolist()
    print(sequence)

    # get CA coordinates
    residues = torch.from_numpy(pdb_CA[['x', 'y', 'z']].to_numpy()).detach().to(device)

    # distances
    distances = get_distances(residues, residues)

    # distance mask over some conditions
    mask = (distances <= 11) & (distances > 6)

    # for each residue, count of resultant connections satisfying conditions
    site_sizes = mask.sum(dim=1)

    SITE_SIZE = site_size

    # idxs of aminoacids who have exactly 6 aminoacids at distance > 6 and < 11
    idxs = ((site_sizes == SITE_SIZE)).nonzero().view(-1)
    print("idx delle righe che hanno 6 connessioni ===> ", idxs)

    # GET GROUPS
    candidate_sites_idx_residues = mask[idxs].nonzero()[:, 1].view(-1)
    candidate_sites_idx_residues = candidate_sites_idx_residues.reshape(-1, SITE_SIZE)

    new_candidate_sites_idx_residues = combinations(candidate_sites_idx_residues, group_size=4)

    #print(new_candidate_sites_idx_residues)

    # number of rows of all possible combinations grouped by 3
    new_size = new_candidate_sites_idx_residues.size(0)

    dd = new_size/idxs.size(0) # ???
    assert new_size % idxs.size(0) == 0 # ???
    #print(idxs)
    #print(new_size)
    dd = int(dd)
    # add a column of the source idx position
    new_candidate_sites_idx_residues = torch.cat((
        idxs.repeat_interleave(dd).view(-1, 1),
        new_candidate_sites_idx_residues
    ), dim=1)

    new_candidate_sites_idx_residues = new_candidate_sites_idx_residues.unique(dim=0)

    #print(new_candidate_sites_idx_residues)

    return residues[new_candidate_sites_idx_residues], \
           new_candidate_sites_idx_residues, \
           sequence, \
           candidate_sites_idx_residues



def get_potential_sites(pdb_df, site_size):
    import sys
    aas = ['CYS', 'HIS', 'GLU', 'ASP']

    # select all C-alpha rows
    pdb_CA = pdb_df[pdb_df['atom_name']=='CA']

    # filter aminoacid 'CYS', 'HIS', 'GLU', 'ASP'
    pdb_CA = pdb_CA[pdb_CA['residue_name'].isin(aas)]

    sequence = (pdb_CA['residue_name']+"_"
                +pdb_CA['chain_id']+"_"
                +pdb_CA['residue_seq_num']
                ).tolist()

    seq = pdb_CA['residue_name'].tolist()
    #print(sequence)

    # get CA coordinates
    residues = torch.from_numpy(pdb_CA[['x', 'y', 'z']].to_numpy()).detach().to(device)

    # distances
    distances = get_distances(residues, residues)

    # distance mask over some conditions
    mask = (distances > 3) & (distances <= 11)
    #mask = (distances > 3) & (distances <= 20)

    # for each residue, count of resultant connections satisfying conditions
    site_sizes = mask.sum(dim=1)

    SITE_SIZE = site_size

    # idxs of aminoacids who have exactly 6 aminoacids at distance > 6 and < 11
    idxs = ((site_sizes == SITE_SIZE)).nonzero().view(-1)
    print(f"idx delle righe che hanno {site_size} connessioni ===> ", idxs)


    # GET GROUPS
    candidate_sites_idx_residues = mask[idxs].nonzero()[:, 1].view(-1)
    candidate_sites_idx_residues = candidate_sites_idx_residues.reshape(-1, SITE_SIZE)

    #print(candidate_sites_idx_residues)

    full_candidate_sites_idx_residues = torch.cat((
        idxs.view(-1, 1),
        candidate_sites_idx_residues
    ), dim=1)

    #print(full_candidate_sites_idx_residues)

    """
    r = []
    for j, _ in enumerate(full_candidate_sites_idx_residues):

        submatrix = mask[full_candidate_sites_idx_residues[j]][:, full_candidate_sites_idx_residues[j]].int()
        #print(submatrix)
        #print(submatrix.triu().sum().item())
        #print(sum(list(range(site_size + 1))))

        if submatrix.triu().sum().item() == sum(list(range(site_size+1))):
            r.append(True)
        else:
            r.append(False)

    print(r)
    """

    #print(residues[full_candidate_sites_idx_residues])

    return residues[full_candidate_sites_idx_residues], \
           full_candidate_sites_idx_residues, \
           residues, \
           seq, \
           distances

    """
    new_candidate_sites_idx_residues = combinations(candidate_sites_idx_residues, group_size=4)

    #print(new_candidate_sites_idx_residues)

    # number of rows of all possible combinations grouped by 3
    new_size = new_candidate_sites_idx_residues.size(0)

    dd = new_size/idxs.size(0) # ???
    assert new_size % idxs.size(0) == 0 # ???
    #print(idxs)
    #print(new_size)
    dd = int(dd)
    # add a column of the source idx position
    new_candidate_sites_idx_residues = torch.cat((
        idxs.repeat_interleave(dd).view(-1, 1),
        new_candidate_sites_idx_residues
    ), dim=1)


    new_candidate_sites_idx_residues = new_candidate_sites_idx_residues.unique(dim=0)

    #print(new_candidate_sites_idx_residues)

    return residues[new_candidate_sites_idx_residues], \
           new_candidate_sites_idx_residues, \
           sequence, \
           candidate_sites_idx_residues
    """



def get_potential_sitesV2(distances, site_size):
    SITE_SIZE = site_size

    # distance mask over some conditions
    mask = (distances > 3) & (distances <= 12)
    #mask = (distances > 3) & (distances <= 20)

    # for each residue, count of resultant connections satisfying conditions
    site_sizes = mask.sum(dim=1)

    # idxs of aminoacids who have exactly 6 aminoacids at distance > 6 and < 11
    idxs = ((site_sizes == SITE_SIZE)).nonzero().view(-1)

    #if idxs.size(0) > 0:
        #print(f"idx delle righe (residui) che hanno {site_size} connessioni ===> ", idxs)

    # GET GROUPS
    candidate_sites_idx_residues = mask[idxs].nonzero()[:, 1].view(-1)
    candidate_sites_idx_residues = candidate_sites_idx_residues.reshape(-1, SITE_SIZE)

    #print(candidate_sites_idx_residues)

    full_candidate_sites_idx_residues = torch.cat((
        idxs.view(-1, 1),
        candidate_sites_idx_residues
    ), dim=1)

    if idxs.size(0) > 0:
        #print(full_candidate_sites_idx_residues)
        pass

    """
    r = []
    for j, _ in enumerate(full_candidate_sites_idx_residues):

        submatrix = mask[full_candidate_sites_idx_residues[j]][:, full_candidate_sites_idx_residues[j]].int()
        #print(submatrix)
        #print(submatrix.triu().sum().item())
        #print(sum(list(range(site_size + 1))))

        if submatrix.triu().sum().item() == sum(list(range(site_size+1))):
            r.append(True)
        else:
            r.append(False)

    print(r)
    """

    #print(residues[full_candidate_sites_idx_residues])

    return full_candidate_sites_idx_residues



def get_all_subgroups(pot_sites_idxs, group_size):

    idxs  = pot_sites_idxs[:, 0]

    pot_sites_idxs = pot_sites_idxs[:,1:]

    new_pot_sites_idxs = combinations(pot_sites_idxs, group_size)

    n_subgroups = new_pot_sites_idxs.size(0)

    dd = n_subgroups/idxs.size(0) # ???
    assert n_subgroups % idxs.size(0) == 0 # ???

    new_pot_sites_idxs = torch.cat((
        idxs.repeat_interleave(int(dd)).view(-1, 1),
        new_pot_sites_idxs
    ), dim=1)

    return new_pot_sites_idxs



def load_potential_sites(pdb_file):

    #print(os.getcwd())
    #pdb_file = '../data/mbsFinderData/1xm5/1xm5.pdb'
    pdb_df = load_pdb(pdb_file)
    #print(pdb_df)


    aas = ['CYS', 'HIS', 'GLU', 'ASP']

    # select all C-alpha rows
    pdb_CA = pdb_df[pdb_df['atom_name']=='CA']

    # select all C-beta rows
    pdb_CB = pdb_df[pdb_df['atom_name'] == 'CB']

    # filter aminoacid 'CYS', 'HIS', 'GLU', 'ASP'
    pdb_CA = pdb_CA[pdb_CA['residue_name'].isin(aas)]
    pdb_CB = pdb_CB[pdb_CB['residue_name'].isin(aas)]

    sequence = (pdb_CA['residue_name']+"_"
                +pdb_CA['chain_id']+"_"
                +pdb_CA['residue_seq_num']
                ).tolist()

    #print(sequence)
    #print(sequence.index('HIS_A_728'), sequence.index('GLU_A_754'), sequence.index('GLU_A_758'))

    sequence2 = np.array(sequence)

    seq = pdb_CA['residue_name'].tolist()

    # get CA coordinates
    residues_ca_xyz = torch.from_numpy(pdb_CA[['x', 'y', 'z']].to_numpy()).detach().to(device)

    # get CB coordinates
    residues_CB_xyz = torch.from_numpy(pdb_CB[['x', 'y', 'z']].to_numpy()).detach().to(device)

    # distances
    distances = get_distances(residues_ca_xyz, residues_ca_xyz)

    # distance mask over some conditions
    #mask = (distances > 3) & (distances <= 11)

    ####################################################################################################################

    ALL = {}

    for group_size in range(3, 30): # poi 2

        #pot_sites_xyz, pot_sites_idxs, residues_ca_xyz, seq, distances \
        #    = get_potential_sites(pdb_df, site_size=group_size)

        pot_sites_idxs = get_potential_sitesV2(distances, site_size=group_size)
        #print(pot_sites_idxs)

        k = f'size_{group_size}'

        SUBGROUP_SIZE = list(range(2, min(group_size, 6)))
        #print("---- Groups of size", group_size, "- ranges: ", SUBGROUP_SIZE, "----")

        if pot_sites_idxs.size(0) > 0:
            ALL2 = {}

            #for subgroup_size in range(3, group_size-1):
            #for subgroup_size in range(2, min(group_size-1, 6)):
            for subgroup_size in SUBGROUP_SIZE:
                #print("subgroup_of_size ", subgroup_size)
                pot_sub_sites_idxs = get_all_subgroups(pot_sites_idxs, group_size=subgroup_size)
                #print(pot_sub_sites_idxs)
                ALL2[f'subgroup_of_size_{subgroup_size}'] = pot_sub_sites_idxs

            ALL[k] = ALL2
        #print("------------------------------------")

    pot_MBS = {}
    for k, v in ALL.items():
        for k2, v2 in ALL[k].items():
            #print(v2)
            key = f"{v2.size(1)}ResSite"

            if v2.size(1) in pot_MBS.keys():  # riferisce al numero di residui (in termini di indici)
                # concatena
                x = torch.cat((pot_MBS[v2.size(1)] , v2))
            else:
                x = v2
            pot_MBS[v2.size(1)] = x
            #pot_MBS[key] = x

    # se il filtri lo mettessimo prima di generare le sotto combinazioni ???
    ####################################################################################################################
    # CLEANING
    empty_keys = []
    for k,v in pot_MBS.items():
        ll = []

        v = v.sort(dim=1)[0].unique(dim=0)

        for jdx in v:
            idx = jdx.numpy()
            ll.append(distances[jdx][:, jdx])

        ll = torch.stack(ll, dim=0)

        ll2 = ll.view(ll.size(0), -1).max(dim=1)[0]

        # seleziona le sottomatrici di adiacenza
        # la cui distanza massima non super una certa soglia
        m = ll.view(ll.size(0), -1).max(dim=1)[0] <= 11 #11

        v = v[m]

        pot_MBS[k] = v
        print(k, v.size())
        if v.size(0)==0:
            empty_keys.append(k)

    for ek in empty_keys:
        pot_MBS.pop(ek, None)

    print(empty_keys)
    for k,v in pot_MBS.items():
        print(k, v.size())
    #sys.exit()

    #poi fare il controllo su chi è il più vicino ...
    ####################################################################################################################

    ### in pot_MBS ci sono gli indici dei residui (le loro coordinate xyz in realtà) che compongono i siti

    # FEATURES

    seq = np.array(seq)

    pot_MBS_final = {}

    # iterate over group idxs
    for k,v in pot_MBS.items():

        features = np.zeros((v.size(0), v.size(1), 5))
        print("===>", k)
        print(v.size())

        #print(features.shape)

        CaCb = residues_CB_xyz - residues_ca_xyz

        CaCb_norm = CaCb[v] / CaCb[v].norm(dim=-1, keepdim=True)

        print("CaCb_norm.size: ", CaCb_norm.size())

        seq_tensor = []

        for j, _ in enumerate(v):

            aa_list = seq[v[j].numpy()].tolist()

            aa_list2 = sequence2[v[j].numpy()].tolist()

            seq_tensor.append(aa_list2)

            for h, aa in enumerate(aa_list):

                #CA_CB_feat =

                if aa == 'ASP':
                    features[j, h, 0] = 1.
                elif aa == 'HIS':
                    features[j, h, 1] = 1.
                elif aa == 'CYS':
                    features[j, h, 2] = 1.
                elif aa == 'GLU':
                    features[j, h, 3] = 1.
                else:
                    features[j, h, 4] = 1.

        features = torch.from_numpy(features).float()
        n_examples, n_ligands = features.size(0), features.size(1)

        ff = torch.zeros((n_ligands*2, 2))

        ff[:n_ligands, 0] = 1

        ff[n_ligands:, 1] = 1

        features = torch.cat((features, features), dim=1)

        features = torch.cat((ff[None,:,:].expand(n_examples,-1,-1), features), dim=2)

        #features = torch.cat((features, CaCb_norm), dim=-1)

        d = {}

        d[f"size{k}_feat"] = features

        d[f"size{k}_geom"] = residues_CB_xyz[v]

        d[f"size{k}_geom_CA"] = residues_ca_xyz[v]

        d[f"size{k}_seq"] = seq_tensor

        t = []

        geom_CA = residues_ca_xyz[v]
        geoms = residues_CB_xyz[v]

        for jj, _ in enumerate(geoms):

            new_geom = torch.cat((geom_CA[jj], geoms[jj]))
            a = get_distances(new_geom, new_geom)
            #a = get_distances(geoms[jj], geoms[jj])
            a = torch.exp(-a / 10)
            t.append(a)
        t = torch.stack(t, dim=0)

        d[f"size{k}_adj"] = t

        pot_MBS_final[k] = d

        #print(v.size(), features.size(), residues_ca_xyz[v].size())
        print("ok")

    return pot_MBS_final


def demo():

    #pdb_file = '../data/mbsFinderData/1xm5/1xm5.pdb'
    pdb_file = "../data/MBSFinder_dataset/3lig/1a0b/1a0b.pdb"


    pot_MBS = load_potential_sites(pdb_file)

    #print("---"*30)
    #for k,v in pot_MBS.items():
    #    print(k)
    #    for k2, v2 in v.items():
    #        print(k2)
    #        print(v2)
    #print("---" * 30)

    for k,v in pot_MBS.items():
        print(k)
        for k2, v2 in v.items():
            if 'seq' in k2:
                print(k2)
                print(np.array(v2))


    """
    # cleaning ...
    ll = []
    for k,v in pot_MBS.items():
        #print(k)
        #print(v)
        print(v.size())
        for jdx in v:
            idx = jdx.numpy()
            ll.append(adj[jdx][:, jdx])

        ll = torch.stack(ll, dim=0)

        m = ll.view(ll.size(0), -1).max(dim=1)[0] <= 11

        v = v[m]
        print(v.size())
        pot_MBS[k] = v
    """

    # setting features
    """
    for k,v in pot_MBS.items():
        seq = np.array(seq)
        features = np.zeros((v.size(0), v.size(1), 5))

        for j, _ in enumerate(v):

            aa_list = seq[v[j].numpy()].tolist()
            #print(v[j])
            #print(adj[v[j]][:, v[j]])

            for h, aa in enumerate(aa_list):
                if aa == 'ASP':
                    features[j, h, 0] = 1.
                elif aa == 'HIS':
                    features[j, h, 1] = 1.
                elif aa == 'CYS':
                    features[j, h, 2] = 1.
                elif aa == 'GLU':
                    features[j, h, 3] = 1.
                else:
                    features[j, h, 4] = 1.

        features = torch.from_numpy(features).float()

        print(features)

        #print(residues_xyz[v])

        #print(residues_xyz[v].size())
    """

#demo()