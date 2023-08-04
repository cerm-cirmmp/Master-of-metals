import sys
import functools
import pandas as pd



class Logger:

    def __init__(self, path):
        self.path = path
        with open(self.path, 'w+') as f:
            pass


    def log(self, text):
        with open(self.path, 'a') as f:
            f.writelines(text+'\n')



AA_LIST = [
    'GLY', 'ALA', 'SER', 'VAL',
    'THR', 'ASN', 'GLN', 'PRO',
    'GLU', 'ASP', 'LEU', 'ILE',
    'HIS', 'MET', 'CYS', 'PHE',
    'TYR', 'TRP', 'LYS', 'ARG'
]


metapdb = {
    'record_name': ([0,6]), # 1,6 --> 0,5 => (0,6)
    'atom_number':[6,11],
    'atom_name':[12,16], # 13, 16 --> 12-15 => (12, 16)
    'character': [16, 17],
    'residue_name':[17,20], # 18-20 --> 17-19, ma per prendere il 19 devo mettere 20 => (17,20)
    'chain_id': [21, 22],
    'residue_seq_num': [22,26],
    'x':[30,38],
    'y':[38,46],
    'z':[46,54],
    'occupancy':[54,60],
    'temp_factor':[60,66],
    #'segment_id':[72,76],
    'element_symbol':[76,78],
    #'atom_charge':[78,80]
}

def processPDBrow(x):
    if x.startswith("ATOM"):
        return x


def load_pdb(file_path, is_site=False, return_metals=False, remove_metals = True):

    #print(file_path)

    with open(file_path, 'r') as f:
        res = f.readlines()

    model_info_rows = list(filter(lambda x:
                                  x if (x[0].startswith("MODEL") or x[0].startswith("ENDMDL"))
                                  else None, zip( res, range(len(res)) )))

    #################################
    if len(model_info_rows) > 0:
        #print("ENSAMBLE OF STRUCTURES")
        m_start, m_finish = model_info_rows[0][1], model_info_rows[1][1]
        #print(m_start,m_finish)

        model1_pdb = filter(lambda x:
                            x if ((x[1] > m_start) and (x[1] < m_finish))
                            else None,  zip(  res, range(len(res))  ))

        pdb_rows = list(map(list, zip(*model1_pdb)))[0]

        res = pdb_rows
    #################################

    if is_site == False:
        pdbrow = filter(lambda x: x if x.startswith("ATOM") else None , res)

    #sss = functools.reduce(lambda x,y: [].append(x) if x.startswith("ATOM") else None, res)
    #sss = map(lambda x: x[metapdb['record_name'][0]: metapdb['record_name'][1]], res)
    #sss = map(lambda x: x[metapdb['residue_name'][0]: metapdb['residue_name'][1]], res)
    df_dict = {}
    for k,v in metapdb.items():
        sss = map(lambda x: x[metapdb[k][0]: metapdb[k][1]].strip(), res)
        #print(k, "||", list(sss))
        df_dict[k] = list(sss)

    df = pd.DataFrame(df_dict)

    if is_site == False:
        df = df[df['record_name'] == 'ATOM']

    df = df[df['character'].isin(['','A'])]


    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    df['temp_factor'] = df['temp_factor'].astype(float)


    if is_site:
        metal = df[(df['temp_factor'] >= 50) & (df['element_symbol'].str.upper() == 'ZN')]
        #print(metal)

    if remove_metals == True:
        df = df[df['record_name'] == 'ATOM']
    else:
        # togli solo le acque
        df = df[ df['residue_name'] != 'HOH' ]

    #aas = map(lambda x: x[17:20] if x.startswith("ATOM") else None, res)
    #aas = map(processPDBrow, res)
    #print(list(aas))

    if return_metals == True:
        return df, metal
    else:
        return df


def load_pdb_BCK(file_path):

    with open(file_path, 'r') as f:
        res = f.readlines()

    pdbrow = filter(lambda x: x if x.startswith("ATOM") else None , res)


    #sss = functools.reduce(lambda x,y: [].append(x) if x.startswith("ATOM") else None, res)
    #sss = map(lambda x: x[metapdb['record_name'][0]: metapdb['record_name'][1]], res)
    #sss = map(lambda x: x[metapdb['residue_name'][0]: metapdb['residue_name'][1]], res)
    df_dict = {}
    for k,v in metapdb.items():
        sss = map(lambda x: x[metapdb[k][0]: metapdb[k][1]].strip(), res)
        #print(k, "||", list(sss))
        df_dict[k] = list(sss)

    df = pd.DataFrame(df_dict)
    df = df[df['record_name'] == 'ATOM']

    df = df[df['character'].isin(['','A'])]

    #print(df)
    #print(df[df['record_name']=='ATOM'])
    #sys.exit()

    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    df['temp_factor'] = df['temp_factor'].astype(float)

    #aas = map(lambda x: x[17:20] if x.startswith("ATOM") else None, res)
    #aas = map(processPDBrow, res)
    #print(list(aas))

    return df

#prot = load_pdb('3kls.pdb')
#print(prot)
#print(prot['temp_factor'])

