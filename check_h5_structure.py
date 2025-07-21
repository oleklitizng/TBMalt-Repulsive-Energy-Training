import h5py
import os

H5_FILE = '../processed_molecules_max200_with_species.h5'

def print_hdf5_structure(group, prefix=''):
    """
    Gibt die Struktur einer HDF5-Gruppe rekursiv aus.
    """
    for key in group.keys():
        item = group[key]
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}  - [Dataset] {key} (Shape: {item.shape}, Dtype: {item.dtype})")
        elif isinstance(item, h5py.Group):
            print(f"{prefix}  - [Gruppe] {key}")
            print_hdf5_structure(item, prefix=prefix + '    ')

if not os.path.exists(H5_FILE):
    print(f"Fehler: Die Datei '{H5_FILE}' wurde nicht gefunden.")
else:
    with h5py.File(H5_FILE, 'r') as f:
        print(f"Struktur der Datei: {H5_FILE}\n" + "="*30)
        print_hdf5_structure(f)
        print("="*30)