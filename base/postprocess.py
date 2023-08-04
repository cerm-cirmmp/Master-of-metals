"""
Implementation of postprocessing methods
"""
from Bio.SVDSuperimposer import SVDSuperimposer
import numpy as np
from numpy import array, dot, set_printoptions

def group_sites(p_sites):
    size4_sites = [x for x in p_sites if len(x) == 4]
    out_list = []

    for p_site in p_sites:
        contained = False

        for size4_site in size4_sites:
            if all([x in size4_site for x in p_site]):
                contained = True

        if contained == False:
            # non è contenuta da nessuna parte
            out_list.append(p_site)

    return size4_sites+out_list


def superimpose():

    sup = SVDSuperimposer()

    source = np.array([[51.65, -1.90, 50.07],
               [50.40, -1.23, 50.65],
               [50.68, -0.04, 51.54],
               [50.22, -0.02, 52.85]], 'f')

    target = np.array([[51.30, -2.99, 46.54],
               [51.09, -1.88, 47.58],
               [52.36, -1.20, 48.03],
               [52.71, -1.18, 49.38]], 'f')

    sup.set(target, source)
    sup.run()

    rot, tran = sup.get_rotran()
    print(rot)
    print(tran)

    source_on_target = sup.get_transformed()
    print(source_on_target)
    print(dot(source, rot) + tran)
    print("->", dot(np.array([50.22, -0.02, 52.85]), rot) + tran)


def get_metal_coord(p_site_coord, known_site_coord, known_site_metal_coord):

    sup = SVDSuperimposer()

    sup.set(p_site_coord, known_site_coord)

    sup.run()

    rot, tran = sup.get_rotran()

    aligned_source_site_coord = dot(known_site_coord, rot) + tran

    p_metal_coord = dot(known_site_metal_coord, rot) + tran

    del sup

    return p_metal_coord, aligned_source_site_coord, rot, tran

