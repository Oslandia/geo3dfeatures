from pathlib import Path

import numpy as np

from sklearn.neighbors import KDTree

from geo3dfeatures.io import xyz as read_xyz
from geo3dfeatures.extract import featuregen


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

# all rows, and the third columns
just_points = data[:, [0, 1, 2]]
assert just_points.shape[1] == 3

print("compute the KDTree")
tree = KDTree(just_points, leaf_size=1_000)

# sample the row
size = 5_000
choice = np.random.choice(np.arange(just_points.shape[0]), size=size, replace=False)
sample = (data[idx] for idx in choice)

g = featuregen(sample, data, tree, neighbors_num)
# g = featuregen(data)


columns = ['lambda1', 'lambda2', 'lambda3', 'a', 'b', 'C', 'L', 'P', 'S', 'mudist', 'r', 'g', 'b']

print("compute and write some geo features")
with open('features.csv', 'w') as fobj:
    fobj.write(','.join(columns))
    fobj.write("\n")
    for row in g:
        fobj.write(",".join(str(x) for x in row))
        fobj.write("\n")
