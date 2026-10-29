"""Built-in dataset formats. Importing this package registers them all.

The registered formats are the folder and HDF5 backends.
"""

from hybrid_aif.data.formats import folder, hdf5  # noqa: F401
from hybrid_aif.data.formats.folder import Folder, FolderReader, FolderWriter
from hybrid_aif.data.formats.hdf5 import HDF5, HDF5Reader, HDF5Writer

__all__ = [
    "HDF5",
    "Folder",
    "FolderReader",
    "FolderWriter",
    "HDF5Reader",
    "HDF5Writer",
]
