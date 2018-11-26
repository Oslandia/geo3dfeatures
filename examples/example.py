from pathlib import Path

import numpy as np

from sklearn.neighbors import KDTree

from geo3dfeatures.io import xyz as read_xyz, write_features
from geo3dfeatures.extract import generate_features


SEED = 42
np.random.seed(SEED)

neighbors_num = 50

_here = Path(__file__).absolute().parent
datapath = _here / '..' / 'data'
# fname = "vegetation-segment.xyz"
# fname = "floor-segment.xyz"
fname = "scene.xyz"
fpath = str(datapath / fname)

print(f"read the file {fpath}")
data = read_xyz(str(fpath))

print(f"generate 3D features")
g = generate_features(data,
                      nb_neighbors=neighbors_num,
                      nb_points=5000,
                      kdtree_leaf_size=1000)

columns = ['a', 'b', 'radius', 'max_difference', 'std_deviation', 'density',
           'verticality', 'curvature_change', 'linearity', 'planarity',
           'scattering', 'omnivariance', 'anisotropy',
           'eigenentropy', 'eigenvalue_sum',
           'r', 'g', 'b']

out_fpath = str(datapath / "features.csv")
print(f"compute and write some geo features in {out_fpath}")
write_features(out_fpath, g, columns)
