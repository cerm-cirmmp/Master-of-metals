
import torch
from base.pdb_loader import load_pdb
import numpy as np
from base.features import process_features

class MetalBindingSite:

    def __init__(self, site_path=None, features = None, coord_CA=None, coord_CB=None, adj=None,
                 sequence = None, site_name = None, protein = None, type='holo', debug=False):
        self.protein = protein
        self.site_name = site_name
        self.sequence = sequence

        #self.aa_sequence = self._get_aa_sequence()
        #self.chains = self._get_chains()

        if type=='holo':
            soglia = 40
        elif type == 'apo':
            soglia = 30
        else:
            soglia = -1

        if site_path == None:
            #print('not calling load_site')
            self.x = features
            self.adj = adj
            self.coord_CA = coord_CA
            self.coord_CB = coord_CB
        else:
            #self.adj, self.x = self._load_site(site_path)
            self.adj, self.x, self.sequence, self.coord_CA, self.coord_CB, self.metals = self._load_site2(site_path, soglia, debug)

        self.real_adj = 15*torch.log(self.adj)


    def get_aa_sequence(self):
        mbs_s = [x.split("_")[0] for x in self.sequence]
        return mbs_s


    def get_chains(self):
        chains = []
        for residue in self.sequence:
            chains.append(residue.split("_")[1])
        chains = list(set(chains))
        return chains

    def _load_site(self, site_path):
        #print(site_path)
        site_df = load_pdb(site_path)
        positions = site_df[site_df['temp_factor'].astype(float) >= 40]['residue_seq_num'].tolist()
        positions = list(set(positions))

        CB = site_df[
            site_df['residue_seq_num'].isin(positions)
            & (site_df['atom_name'] == 'CB')]

        CA = site_df[
            site_df['residue_seq_num'].isin(positions)
            & (site_df['atom_name'] == 'CA')]

        if len(CA) != len(CB):
            print("Problems with: ", site_path)
            #print(CA)
            #print(CB)
            #raise Exception


        coordinates_CB = CB[['x', 'y', 'z']].to_numpy()
        coordinates_CA = CA[['x', 'y', 'z']].to_numpy()

        #print(coordinates_CB)
        #print(coordinates_CA)
        CaCb = coordinates_CB-coordinates_CA
        CaCb = torch.from_numpy(CaCb).float()
        CaCb_norm = CaCb/CaCb.norm(dim=-1, keepdim=True)

        aminoacids = CB['residue_name'].tolist()

        real_distances_CB = np.sqrt(((coordinates_CB[:, None, :] - coordinates_CB[None, :, :]) ** 2).sum(-1))
        real_distances_CA = np.sqrt(((coordinates_CA[:, None, :] - coordinates_CA[None, :, :]) ** 2).sum(-1))
        # print(distances)
        distances_CB = np.exp(-real_distances_CB / 10)
        distances_CA = np.exp(-real_distances_CA / 10)

        distances_CB = torch.from_numpy(distances_CB).float()
        distances_CA = torch.from_numpy(distances_CA).float()


        #coordinates_all = np.concatenate((coordinates_CA, coordinates_CB), axis=0)

        features = np.zeros((len(aminoacids), 5))
        for j, aa in enumerate(aminoacids):
            # idx = configs.AMINOACIDS[aa]
            # features[j, idx] = 1.
            if aa == 'ASP':
                features[j, 0] = 1.
            elif aa == 'HIS':
                features[j, 1] = 1.
            elif aa == 'CYS':
                features[j, 2] = 1.
            elif aa == 'GLU':
                features[j, 3] = 1.
            else:
                features[j, 4] = 1.

        features = torch.from_numpy(features).float()
        #features = torch.cat((features, CaCb_norm), dim=-1)

        return distances_CB, features


    def _load_site2(self, site_path, SOGLIA,debug):



        # print(site_path)
        site_df, metals = load_pdb(site_path, is_site=True, return_metals=True)

        site_df['chain_id_seq_num'] = site_df['chain_id']+"_"+site_df['residue_seq_num']

        #positions = site_df[site_df['temp_factor'].astype(float) >= SOGLIA]['residue_seq_num'].tolist()

        positions_and_chain = site_df[site_df['temp_factor'].astype(float) >= SOGLIA]['chain_id_seq_num'].tolist()

        #positions = list(set(positions))
        positions_and_chain = list(set(positions_and_chain))

        if debug:
            print(site_path)
            print(positions_and_chain)

        CB = site_df[
            #site_df['residue_seq_num'].isin(positions)
            site_df['chain_id_seq_num'].isin(positions_and_chain)
            & (site_df['atom_name'] == 'CB')]

        CA = site_df[
            #site_df['residue_seq_num'].isin(positions)
            site_df['chain_id_seq_num'].isin(positions_and_chain)
            & (site_df['atom_name'] == 'CA')]

        if len(CA) != len(CB):
            print("Problems with: ", site_path)
            # print(CA)
            # print(CB)
            raise Exception

        sequence = ( CA['residue_name'] + "_"
                    + CA['chain_id'] + "_"
                    + CA['residue_seq_num']
                    ).tolist()

        aminoacids = CB['residue_name'].tolist()

        coordinates_CB = CB[['x', 'y', 'z']].to_numpy()
        coordinates_CA = CA[['x', 'y', 'z']].to_numpy()

        distances_CB, features = process_features(coordinates_CA,
                                                  coordinates_CB,
                                                  aminoacids=aminoacids)

        return distances_CB, features, sequence, coordinates_CA, coordinates_CB, metals


    def __str__(self):
        return f"MBS => Prot: {self.protein} | Site_name: {self.site_name} | Sequence: {self.sequence}"