"""
- Modulo per la trasformazione delle features -
Different from the dataloader!!!
"""

import sys
import numpy as np
import torch
import pandas as pd


def get_aminoacid_features(aminoacids):

    #print("<<<<", aminoacids)

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

    return features


def angolo_tra_due_vettori(v1, v2):
    v1_norm = v1.norm(dim=-1)
    v2_norm = v2.norm(dim=-1)

    angle = torch.arccos(
        ( (v1/v1_norm)*(v2/v2_norm) ).sum(-1)
    )

    return angle


def get_angle_feature(coordinates_CB):
    """

    - Per ogni nodo/punto si identificano quelli cui è connesso. Come?

    - Nel caso dei triangoli non ce n'è bisogno: quindi ad ogni punto è associato un angolo

    """
    if coordinates_CB.dim() == 3:
        results = []
        for j in range(coordinates_CB.size(0)):
            X = coordinates_CB[j]
            pairs = X[None, :, :] - X[:, None, :]
            a1 = angolo_tra_due_vettori(pairs[0, 1], pairs[0, 2])
            a2 = angolo_tra_due_vettori(pairs[1, 0], pairs[1, 2])
            a3 = angolo_tra_due_vettori(pairs[2, 0], pairs[2, 1])
            res = torch.tensor([a1, a2, a3])
            results.append(res.view(1,-1))
        results = torch.cat(results, dim=0)

    else:
        pairs = coordinates_CB[None, :, :] - coordinates_CB[:, None, :]
        a1 = angolo_tra_due_vettori(pairs[0, 1], pairs[0, 2])
        a2 = angolo_tra_due_vettori(pairs[1, 0], pairs[1, 2])
        a3 = angolo_tra_due_vettori(pairs[2, 0], pairs[2, 1])
        results = torch.tensor([a1, a2, a3]).view(1,-1)

    return results


def get_angle_feature_BCK(coordinates_CB):
    """
    - Per ogni nodo/punto si identificano quelli cui è connesso, nel caso dei triangoli non ce n'è bisogno
    - Quindi ad ogni punto è associato un angolo
    """
    print(">>")
    print(coordinates_CB)
    pairs = coordinates_CB[None, :, :] - coordinates_CB[:, None, :]

    a1 = angolo_tra_due_vettori(pairs[0, 1], pairs[0, 2])
    a2 = angolo_tra_due_vettori(pairs[1, 0], pairs[1, 2])
    a3 = angolo_tra_due_vettori(pairs[2, 0], pairs[2, 1])

    return torch.tensor([a1, a2, a3])



###############################################################################################



def CB_slope_distances():
    pass


def CB_slope_angles():
    pass


def CB_and_CA():
    pass



def process_features_BCK(coordinates_CA,
                     coordinates_CB,
                     real_distances_CA = None,
                     real_distances_CB = None,
                     aminoacids=None,
                     features = None):



    if isinstance(coordinates_CA, np.ndarray):
        coordinates_CA = torch.from_numpy(coordinates_CA).float()
        coordinates_CB = torch.from_numpy(coordinates_CB).float()



    if ((real_distances_CA is None) and (real_distances_CB is None)):

        real_distances_CA = torch.sqrt(
            ((coordinates_CA[:, None, :] - coordinates_CA[None, :, :]) ** 2).sum(-1))

        real_distances_CB = torch.sqrt(
            ((coordinates_CB[:, None, :] - coordinates_CB[None, :, :]) ** 2).sum(-1))


    # ADJ VALUES
    distances_CB = torch.exp(-real_distances_CB / 10).float()
    distances_CA = torch.exp(-real_distances_CA / 10).float()
    #distances_CB = distances_CB.float()
    #distances_CA = distances_CA.float()

    # AMINOACID FEATURES
    if features == None:
        features = get_aminoacid_features(aminoacids)

    # calculate angle features
    angles = get_angle_feature(coordinates_CB)

    # ADD ANGLE FEATURES
    if features.dim()==2:
        features = torch.cat((features, angles.t()), dim=-1)
    else:
        angles  = angles.unsqueeze(2)
        features = torch.cat((features, angles), dim=-1)

    # calculate slope features
    inclinazione = torch.exp(real_distances_CA - real_distances_CB)

    # INSERT PAIR SLOPE FEATURES
    distances_CB = distances_CB * inclinazione

    return distances_CB.float(), features.float()



def process_features(coordinates_CA,
                     coordinates_CB,
                     real_distances_CA = None,
                     real_distances_CB = None,
                     aminoacids=None,
                     features = None):


    if isinstance(coordinates_CA, np.ndarray):
        coordinates_CA = torch.from_numpy(coordinates_CA).float()
        coordinates_CB = torch.from_numpy(coordinates_CB).float()


    # se non sono state calcolate le distanze (==> stiamo processando un singolo sito)
    if ((real_distances_CA is None) and (real_distances_CB is None)):

        real_distances_CA = torch.sqrt(
            ((coordinates_CA[:, None, :] - coordinates_CA[None, :, :]) ** 2).sum(-1))

        real_distances_CB = torch.sqrt(
            ((coordinates_CB[:, None, :] - coordinates_CB[None, :, :]) ** 2).sum(-1))

        coordinates_CAandCB = torch.cat((coordinates_CA, coordinates_CB), dim=0)

        real_distances_CAandCB = torch.sqrt(
            ((coordinates_CAandCB[:, None, :] - coordinates_CAandCB[None, :, :]) ** 2).sum(-1))





    # stiamo processando un batch di potential site
    else:
        # check please
        coordinates_CAandCB = torch.cat((coordinates_CA, coordinates_CB), dim=1)

        real_distances_CAandCB = torch.sqrt(
            ((coordinates_CAandCB[:,:, None, :] - coordinates_CAandCB[:,None, :, :]) ** 2).sum(-1))


    # ADJ VALUES
    distances_CB = torch.exp(-real_distances_CB / 10).float()
    distances_CA = torch.exp(-real_distances_CA / 10).float()
    #distances_CAandCB = torch.exp(-real_distances_CAandCB / 10).float()
    distances_CAandCB = torch.exp(-real_distances_CAandCB/15).float()
    #distances_CAandCB = real_distances_CAandCB.float()

    # tutti 1
    #distances_CAandCB = torch.ones_like(distances_CAandCB)



    #distances_CB = distances_CB.float()
    #distances_CA = distances_CA.float()

    # AMINOACID FEATURES
    if features == None:
        features = get_aminoacid_features(aminoacids)
        AB = torch.zeros(2*len(features), 2)

        AB[0:len(features), 0] = 1.
        AB[len(features):, 1] = 1.

        #vanno concatenate a se stesse e aggiunti due vettori indicanti il tipo (se CA o CB)
        features = torch.cat((features, features), dim=0)

        features = torch.cat((features, AB), dim=1)


    else:

        features = torch.cat((features, features), dim=1)

        AB = torch.zeros(features.size(0), features.size(1), 2)

        f_size = int(features.size(1)/2)
        AB[:, 0:f_size, 0] = 1.
        AB[:, f_size:, 1] = 1.

        features = torch.cat( (features, AB), dim=2)


    # calculate angle features
    #angles = get_angle_feature(coordinates_CB)

    # ADD ANGLE FEATURES
    #if features.dim()==2:
    #    features = torch.cat((features, angles.t()), dim=-1)
    #else:
    #    angles  = angles.unsqueeze(2)
    #    features = torch.cat((features, angles), dim=-1)

    # calculate slope features
    #inclinazione = torch.exp(real_distances_CA - real_distances_CB)

    # INSERT PAIR SLOPE FEATURES
    #distances_CB = distances_CB * inclinazione

    #return distances_CB.float(), features.float()
    return distances_CAandCB.float(), features.float()


def get_features_from_aminoacid(aminoacids):
    features = get_aminoacid_features(aminoacids)
    AB = torch.zeros(2 * len(features), 2)

    AB[0:len(features), 0] = 1.
    AB[len(features):, 1] = 1.

    # vanno concatenate a se stesse e aggiunti due vettori indicanti il tipo (se CA o CB)
    features = torch.cat((features, features), dim=0)

    features = torch.cat((features, AB), dim=1)
    return features