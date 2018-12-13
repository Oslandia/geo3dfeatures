import sys
import argparse
import numpy as np

from geo3dfeatures.io import xyz as read_xyz, las as read_las, write_features
from geo3dfeatures.extract import generate_features


def _parse_args(args):
    parser = argparse.ArgumentParser(description=("3D point cloud geometric"
                                                  " feature extraction"))
    parser.add_argument('-i', '--input-file', required=True,
                        help="Input 3D point cloud file")
    parser.add_argument('-n', '--neighbors',
                        type=int, default=50,
                        help="Number of neighbors to consider")
    parser.add_argument('-o', '--output-file',
                        default="output_features.csv",
                        help="Output csv file name")
    parser.add_argument('-p', '--sample-points',
                        type=int, help="Number of sample points to evaluate")
    parser.add_argument('-t', '--kdtree-leafs',
                        type=int, default=1000,
                        help="Number of leafs in KD-tree")
    return parser.parse_args(args)


def main(argv=sys.argv[1:]):
    opts = _parse_args(argv)
    print(f"read the file {opts.input_file}")
    extension = str(opts.input_file).split(".")[-1]
    if extension == "xyz":
        data = read_xyz(opts.input_file)
    elif extension == "las":
        data = read_las(opts.input_file)
    else:
        raise ValueError("Wrong file extension, please send xyz or las file.")
    if opts.sample_points is not None:
        print("Work with a sample of points")
        sample_mask = np.random.choice(np.arange(data.shape[0]),
                                       size=opts.sample_points,
                                       replace=False)
        data = data[sample_mask]

    print(f"generate 3D features")
    gen = generate_features(data,
                            nb_neighbors=opts.neighbors,
                            kdtree_leaf_size=opts.kdtree_leafs)

    print(f"compute and write some geo features in {opts.output_file}")
    write_features(opts.output_file, gen)


if __name__ == '__main__':
    main()
