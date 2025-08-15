import torch
from tbmalt import Geometry, OrbitalInfo
from tbmalt.tools.downloaders import download_dftb_parameter_set
from tbmalt.physics.dftb import Dftb2
from tbmalt.physics.dftb.feeds import SkFeed, SkfOccupationFeed, HubbardFeed, RepulsiveSplineFeed
from tbmalt.data.units import length_units


torch.set_default_dtype(torch.float64)

def H2O_scc(device):
    # Ensure device is a torch device
    if isinstance(device, str):
        device = torch.device(device)
    
    cutoff = torch.tensor([9.98], device=device)

    geometry = Geometry(
        torch.tensor([8,8,1,1], device=device),
        torch.tensor([
            [0.733608,     0.079290,    -0.028442],
            [-0.732411,    -0.087262,    -0.021032],
            [0.690160,    -0.528189,    -0.014445],
            [-0.696622,     0.666672,    -0.019376]],
            device=device),units='a')

    orbs = OrbitalInfo(geometry.atomic_numbers, {1: [0], 8: [0, 1]})

    return geometry, orbs 

def feeds_scc(device, skf_file):
    species = [1, 8]
    h_feed = SkFeed.from_database(skf_file, species, 'hamiltonian', device=device)
    s_feed = SkFeed.from_database(skf_file, species, 'overlap', device=device)
    o_feed = SkfOccupationFeed.from_database(skf_file, species, device=device)
    u_feed = HubbardFeed.from_database(skf_file, species, device=device)
    r_feed = RepulsiveSplineFeed.from_database(skf_file, species, device=device)

    return h_feed, s_feed, o_feed, u_feed,r_feed

file_path = "mio.h5"  # Save as HDF5 file
device = torch.device('cpu')

h_feed, s_feed, o_feed, u_feed, r_feed= feeds_scc(device,file_path)

calculator = Dftb2(h_feed, s_feed, o_feed, u_feed,r_feed, filling_scheme='fermi', filling_temp=0.001)

geometry, orbs = H2O_scc(device)

energy = calculator(geometry, orbs)
print('Energy:', energy)

total_energy = calculator.total_energy
print('Total energy:', total_energy)

q_final_atomic = calculator.q_final_atomic
print('Q Final Atomic:', q_final_atomic)

hamiltonian = calculator.hamiltonian
print('Hamiltonian:', hamiltonian)

forces = calculator.forces
print('forces:', forces)

dipole = calculator.dipole
print('dipole_moment:', dipole)

repulsive_energy = calculator.repulsive_energy
print('repulsive_energy:', repulsive_energy)