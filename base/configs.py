# path vari
import pathlib


#AMINOACIDS = ['ALA', 'ARG', 'ANS', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
#              'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRY', 'TYR', 'VAL']

AMINOACIDS = {
    'ALA': 0,
    'ARG': 1,
    #'ANS': 2,
    'ASN': 2,
    'ASP': 3,
    'CYS': 4,
    'GLU': 5,
    'GLN': 6,
    'GLY': 7,
    'HIS': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'MET': 12,
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'THR': 16,
    #'TRY': 17,
    'TRP': 17,
    'TYR': 18,
    'VAL': 19
}

#RND_FILE_LIST = pathlib.Path('~/CERM').joinpath('MetalSitePredictionV2').joinpath('data').joinpath('zinco_negative_sitesV2.csv')
#PHYSIO_FILE_LIST = pathlib.Path('~/CERM').joinpath('MetalSitePredictionV2').joinpath('data').joinpath('zinco_positive_sites.csv')
#ZINCO_PATH = pathlib.Path('~/CERM').expanduser().joinpath('MetalSitePredictionV2').joinpath('data').joinpath('zinco')


DATA_PATH = pathlib.Path(__file__)\
    .parent\
    .joinpath('data')\
    .joinpath('MBSFinder_dataset')


PHYSIO_FILE = pathlib.Path(__file__)\
    .parent\
    .joinpath('data').joinpath('zinco_positive_sites.csv')


RANDOM_FILE = pathlib.Path(__file__)\
    .parent\
    .joinpath('data').joinpath('zinco_negative_sitesV2.csv')


DATA_PATH2 = pathlib.Path(__file__)\
    .parent\
    .joinpath('data')\
    .joinpath('MBSFinderData')

DATA_PATH_ZN = pathlib.Path(__file__)\
    .parent\
    .joinpath('data_Zn')


PROJECT_ROOT = pathlib.Path(__file__).parent

NUM_FEATURES = 7

CU_RESIDUES = ['HIS', 'CYS', 'MET', 'ASP', 'GLU']

SITES_PATH = pathlib.Path("/mnt/disk4Tb/Vincenzo/data/site_files")