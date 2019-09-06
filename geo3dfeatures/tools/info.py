"""Describe an input .las file by providing the following informations:
  * filename
  * relative path
  * number of 3D points
  * feature list
  * x, y and z definition domains
  * already computed kd-trees
  * already extracted features
  * already computed clusters
"""

from pathlib import Path
import sys

import daiquiri
import numpy as np
import laspy

from geo3dfeatures.io import ply, xyz


logger = daiquiri.getLogger(__name__)


def main(opts):
    input_path = Path(opts.datapath, "input", opts.input_file)
    if not input_path.is_file():
        logger.error("No such file '%s'.", input_path)
        sys.exit(1)
    output_path = Path(opts.datapath, "output", opts.input_file.split(".")[0])
    kdtrees = [f.name for f in output_path.glob("*.pkl")]
    features = [f.name for f in Path(output_path, "features").glob("*.h5")]
    kmeans = [f.name for f in Path(output_path, "clustering").glob("*.xyz")]
    kmeans = kmeans + [f.name for f in Path(output_path, "clustering").glob("*.las")]
    if input_path.suffix == ".las":
        header = laspy.header.Header()
        reader = laspy.base.FileManager(input_path, mode="r")
        hm = laspy.header.HeaderManager(header, reader)
        nb_points = hm.point_records_count
        feature_list = [s.name for s in reader.point_format.specs]
        xmin, ymin, zmin = hm.min
        xmax, ymax, zmax = hm.max
    elif input_path.suffix == ".xyz":
        logger.warning(
            "The file is read; it may take time regarding the file size"
        )
        data = xyz(input_path)
        nb_points = len(data)
        feature_list = "x, y, z, r, g, b"
        xmin, ymin, zmin = np.min(data[:, :3], axis=0)
        xmax, ymax, zmax = np.max(data[:, :3], axis=0)
    elif input_path.suffix == ".ply":
        logger.warning(
            "The file is read; it may take time regarding the file size"
        )
        data = ply(input_path)
        nb_points = len(data)
        feature_list = "x, y, z"
        xmin, ymin, zmin = np.min(data, axis=0)
        xmax, ymax, zmax = np.max(data, axis=0)
    else:
        logger.error("Unknown file extension, '%s' not supported", input_path.suffix)
        sys.exit(1)
    info_string = (
        f"File : {opts.input_file}"
        f"\nFolder : {input_path.parent}"
        f"\nNb points : {nb_points}"
        f"\nFeature list : {feature_list}"
        f"\nx range : {xmax-xmin:.2f}\t[{xmin:.2f}; {xmax:.2f}]"
        f"\ny range : {ymax-ymin:.2f}\t[{ymin:.2f}; {ymax:.2f}]"
        f"\nz range : {zmax-zmin:.2f}\t[{zmin:.2f}; {zmax:.2f}]"
        f"\nkd-tree : {kdtrees}"
        f"\nfeatures : {features}"
        f"\nk-mean : {kmeans}"
        )
    logger.info("Generate info on an input file...\n%s", info_string)
