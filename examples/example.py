from pathlib import Path

import argparse
import numpy as np

from sklearn.neighbors import KDTree

from geo3dfeatures.io import xyz as read_xyz, write_features
from geo3dfeatures.extract import generate_features

SEED = 42
np.random.seed(SEED)

if __name__=='__main__':

    parser = argparse.ArgumentParser(description=("3D point cloud geometric"
                                                  " feature extraction"))
    parser.add_argument('-i', '--input-file', required=True,
                        help="Input 3D point cloud file")
    parser.add_argument('-n', '--neighbors',
                        type=int, default=50,
                        help="Number of neighbors to consider")
    parser.add_argument('-o', '--output-file',
                        default="data/features.csv",
                        help="Output csv file name")
    parser.add_argument('-p', '--sample-points',
                        type=int, default=1000,
                        help="Number of sample points to evaluate")
    parser.add_argument('-t', '--kdtree-leafs',
                        type=int, default=1000,
                        help="Number of leafs in KD-tree")
    args = parser.parse_args()

    print(f"read the file {args.input_file}")
    data = read_xyz(args.input_file)

    print(f"generate 3D features")
    g = generate_features(data,
                          nb_neighbors=args.neighbors,
                          nb_points=args.sample_points,
                          kdtree_leaf_size=args.kdtree_leafs)

    columns = ['alpha', 'beta',
               'z', 'radius', 'z_range', 'std_deviation', 'density', 'verticality',
               'curvature_change', 'linearity', 'planarity',
               'scattering', 'omnivariance', 'anisotropy',
               'eigenentropy', 'eigenvalue_sum',
               'radius_2D', 'density_2D',
               'eigenvalue_sum_2D', 'eigenvalue_ratio_2D',
               'bin_density', 'bin_z_range', 'bin_z_std',
               'r', 'g', 'b']

    out_fpath = args.output_file
    print(f"compute and write some geo features in {out_fpath}")
    write_features(out_fpath, g, columns)
