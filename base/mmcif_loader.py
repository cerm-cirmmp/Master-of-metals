
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import pandas as pd
import base.configs as configs

AA3C = list(configs.AMINOACIDS.keys())


def load_mmcif(file_path, aa3c = AA3C):

    chain_, residue_, position_, atomName_, x_, y_, z_ = [], [], [], [], [], [], []
    occupancy_ , bfactor_ , atom_number_, character_, element_  = [], [], [], [], []

    # Create an MMCIFParser instance
    parser = MMCIFParser()

    # Parse the MMCIF file
    structure = parser.get_structure('my_structure', file_path)

    # Access the structure's data
    #for model in structure:
    model = structure[0]
    for chain in model:
        for residue in chain:
            #print(residue)
            residue_name = residue.get_resname()
            atom_seq_position = str(residue.get_id()[1])
            for atom in residue:
                atom_id = str(atom.get_id())
                atom_name = str(atom.get_name())
                atom_coord = atom.get_coord()

                atom_occupancy = str(atom.get_occupancy())
                atom_number = str(atom.get_serial_number())
                atom_bfactor = atom.get_bfactor()
                alt_loc = str(atom.get_altloc())
                chain_id = str(chain.get_id())
                model_id = str(model.get_id())

                #atom_seq_position = residue.get_full_id()[3][1]
                atom_number_.append(atom_number)
                atomName_.append(atom_name)
                character_.append(alt_loc)
                chain_.append(chain_id)
                residue_.append(residue_name)
                position_.append(atom_seq_position)

                x_.append(atom_coord[0])
                y_.append(atom_coord[1])
                z_.append(atom_coord[2])

                occupancy_.append(atom_occupancy)
                bfactor_.append(atom_bfactor)
                element_.append(atom.element)

                #df = df.append(row, ignore_index=TabError)

                # Do something with the atom information
                #print(f"Model: {model_id}, Chain: {chain_id}, Pos: {atom_seq_position}, Residue: {residue_name}, Atom: {atom_name}, ID: {atom_id}, Coordinates: {atom_coord}")

    df = pd.DataFrame({
        'atom_number':atom_number_,
        'atom_name': atomName_,
        'character': character_,
        'residue_name':residue_,
        'chain_id': chain_,
        'residue_seq_num':position_,
        'x':x_,
        'y':y_,
        'z':z_,
        'occupancy':occupancy_,
        'temp_factor':bfactor_,
        'element_symbol':element_
    })

    del structure, parser

    df = df[df['residue_name'].isin(aa3c)]

    return df


#df = load_mmcif('12ca.cif')

#print(df)


#pdb_info = MMCIF2Dict('12ca.cif')


#for key, value in pdb_info.items():
#    if "_atom_site." in key:
#        print(key)


#print(pdb_info['_atom_site.group_PDB'])