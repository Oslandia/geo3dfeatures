from pathlib import Path

from geo3dfeatures import io


_here = Path(__file__).absolute().parent
DATADIR = _here / "data"
PLYFILE = DATADIR / "tet.ply"


def test_read_xyz_from_ply_file():
    fpath = str(PLYFILE)
    data = io.read_ply(fpath)
    assert data.shape == (4, 3)
