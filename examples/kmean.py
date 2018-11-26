from pathlib import Path

import pandas as pd

from sklearn.cluster import KMeans

from geo3dfeatures.io import xyz as read_xyz


SEED = 1337


_here = Path(__file__).absolute().parent
datapath = _here / '..' / 'data'
fname = "scene.xyz"
fpath = str(datapath / fname)


# print(f"read the file {fpath}")
raw = read_xyz(str(fpath))


print("reading data")
data = pd.read_csv('features.csv')
nvalues = data.shape[0]

data["r"] /= 255
data["g"] /= 255
data["b"] /= 255

# del data["r"]
# del data["g"]
# del data["b"]

print("k-means")
model = KMeans(2, random_state=SEED)
model.fit(data)

# lighter
floor_label = model.cluster_centers_.sum(axis=1).argmax()
labels = pd.Series(model.labels_)


print("writing kmeans.xyz")
result = pd.DataFrame(raw[:nvalues, [0, 1, 2]], columns=['x', 'y', 'z'])
green = [20, 200, 20]
grey = [100, 100, 100]
result["r"] = 20
result["g"] = 200
result["b"] = 20
mask_floor = labels == floor_label
result.loc[mask_floor, "r"] = 100
result.loc[mask_floor, "g"] = 100
result.loc[mask_floor, "b"] = 100

result.to_csv('kmean.xyz', sep=' ', index=False, header=False)
