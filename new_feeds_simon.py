from __future__ import annotations
import warnings
import re
import numpy as np
import os
from numpy import ndarray as Array
from itertools import combinations_with_replacement
from typing import List, Literal, Optional, Dict, Tuple, Union, Type
from scipy.interpolate import CubicSpline as ScipyCubicSpline
import torch
from torch import Tensor
from torch.nn import Parameter, ParameterDict, ModuleDict, Module

from tbmalt import Geometry, OrbitalInfo, Periodicity
from tbmalt.structures.geometry import atomic_pair_distances
from tbmalt.ml.integralfeeds import IntegralFeed
from tbmalt.io.skf import Skf, VCRSkf
from tbmalt.physics.dftb.slaterkoster import sub_block_rot
from tbmalt.data.elements import chemical_symbols
from tbmalt.ml import Feed
from tbmalt.common.batch import pack, prepeat_interleave, bT, bT2
from tbmalt.common.maths.interpolation import PolyInterpU, BicubInterpSpl
from tbmalt.common.maths.interpolation import CubicSpline
from tbmalt.common import unique
from tbmalt.physics.dftb.feeds import PairwiseRepulsiveEnergyFeed, DftbpRepulsiveSpline
import matplotlib.pyplot as plt


torch.set_default_dtype(torch.float64)

# Deine Klassen xTBRepulsive, PTBPRepulsive, DFTBGammaRepulsive bleiben hier unverändert.
# ... (Ich lasse sie hier zur Übersichtlichkeit weg, sie sind aber im finalen Code enthalten)
class xTBRepulsive(Feed):
     
    """Repulsive in form of the xTB-Repulsive.

    Computes the repulsive energy term (E_rep) between atoms A and B.

    This expression is commonly used in semiempirical quantum chemical methods to model 
    the short-range repulsive interaction between atoms. The energy is calculated as:

        E_rep = (Z_A^eff * Z_B^eff / R_AB) * exp(-sqrt(α_A * α_B) * (R_AB)^k_f)

    Where:
    - Z_A^eff, Z_B^eff: Effective nuclear charges of atoms A and B
    - R_AB: Distance between atoms A and B
    - α_A, α_B: Element-specific repulsion parameters for atoms A and B
    - k_f: Empirical exponent controlling the distance dependence of the repulsion

    Arguments:
        coefficients: List containing import parameter
            c[0] := Z_A^eff
            c[1] := Z_B^eff
            c[2] := α_A
            c[3] := α_B
            c[4] := k_f
        cutoff: Cutoff radius for the repulsive beyond which interactions
            are assumed to be zero.
    """

    def __init__(
            self, coefficients: List[Tensor], cutoff: Tensor): # Korrigiert: erwarte eine Liste von Tensoren

        super().__init__()
        self.coefficients = coefficients
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        """Evaluate the repulsive interaction at the specified distance(s).

        Arguments:
            distances: Distance(s) at which the repulsive term is to be
                evaluated.

        Returns:
            repulsive: Repulsive interaction energy as evaluated at the
                specified distances.
        """
        results = torch.zeros_like(distances)
        c = self.coefficients
        z1 = c[0]
        z2 = c[1]
        a1 = c[2]
        a2 = c[3]
        kf = c[4]
        mask = distances < self.cutoff

        results[mask] = z1 * z2 / distances[mask] * torch.exp(-torch.sqrt(a1 * a2) * distances[mask]**kf)

        return results
    
class PTBPRepulsive(Feed):
     
    """Repulsive in form of the PTBP-Repulsive.

    The repulsive is calculated as the follwing form:

    E_rep = (Z_A^eff * Z_B^eff / R_AB) * (1 - erf(R_AB / sqrt(α_A^2 + α_B^2)))

    Arguments:
        coefficients: List containing import parameter
            c[0] := Z_A^eff
            c[1] := Z_B^eff
            c[2] := α_A
            c[3] := α_B
        cutoff: Cutoff radius for the repulsive beyond which interactions
            are assumed to be zero.
    """

    def __init__(
            self, coefficients: List[Tensor], cutoff: Tensor): # Korrigiert

        super().__init__()
        self.coefficients = coefficients
        self.cutoff = cutoff

    def forward(self, distances: Tensor) -> Tensor:
        """Evaluate the repulsive interaction at the specified distance(s).

        Arguments:
            distances: Distance(s) at which the repulsive term is to be
                evaluated.

        Returns:
            repulsive: Repulsive interaction energy as evaluated at the
                specified distances.
        """
        results = torch.zeros_like(distances)
        c = self.coefficients
        z1 = c[0]
        z2 = c[1]
        a1 = c[2]
        a2 = c[3]
        gamma = 1 / torch.sqrt(a1**2 + a2**2)
        mask = distances < self.cutoff

        results[mask] = z1 * z2 / distances[mask] * (1 - torch.erf(gamma * distances[mask]))

        return results
    
class DFTBGammaRepulsive(Feed):
    """Repulsive in form of the DFTB-Gamma.

    The repulsive energy is derived from the overlap integral. This implementation
    has been modified to exclude the leading '1/R' term from the standard
    DFTB gamma function, as per user specification.

    Arguments:
        coefficients: A list containing the essential parameters:
            - c[0]: Z_A_eff (Effective nuclear charge of atom A)
            - c[1]: Z_B_eff (Effective nuclear charge of atom B)
            - c[2]: α_A (Element-specific repulsion parameter for atom A)
            - c[3]: α_B (Element-specific repulsion parameter for atom B)
        cutoff: The cutoff radius beyond which the repulsive interaction
            is considered to be zero.
    """

    def __init__(self, coefficients: List[Tensor], cutoff: Tensor): # Korrigiert
        super().__init__()
        self.coefficients = coefficients
        self.cutoff = cutoff

    def _Gamma(self, a, b, R):
        """Helper function to compute a component of the repulsive energy."""
        term1_numerator = b**4 * a
        term1_denominator = 2 * (a**2 - b**2)**2
        term2_numerator = b**6 - 3 * b**4 * a**2
        term2_denominator = (a**2 - b**2)**3 * R
        return (term1_numerator / term1_denominator) - (term2_numerator / term2_denominator)

    def _equal_gamma(self, distances, a1):
        """
        Calculates the repulsive term for equal parameters (a1 == a2),
        omitting the leading '1/R' term as requested.
        
        This now implements: -e^(-τR) * (1/R + 11τ/16 + 3τ²R/16 + τ³R²/48)
        """
        # This is the part inside the parentheses in the original formula
        inner_term = (1 / distances) + (11 * a1 / 16) + (3 * a1**2 * distances / 16) + (a1**3 * distances**2 / 48)
        
        # Calculate the exponential part and multiply by the inner term
        exp_term = torch.exp(-a1 * distances) * inner_term
        
        # Return only the negative exponential term
        return exp_term

    def _unequal_gamma(self, distances, a1, a2):
        """
        Calculates the repulsive term for unequal parameters (a1 != a2),
        omitting the leading '1/R' term as requested.
        
        This now implements: -e^(-τₐR)Γ(τₐ,τᵦ,R) - e^(-τᵦR)Γ(τᵦ,τₐ,R)
        """
        exp1 = torch.exp(-a1 * distances)
        exp2 = torch.exp(-a2 * distances)

        # Calculate the two Gamma terms
        term2 = exp1 * self._Gamma(a1, a2, distances)
        term3 = exp2 * self._Gamma(a2, a1, distances)

        # The initial "1 / distances" term is removed.
        return term2 - term3

    def forward(self, distances: Tensor) -> Tensor:
        """Evaluate the repulsive interaction at the specified distance(s).

        Arguments:
            distances: A tensor of distances at which the repulsive term
                is to be evaluated.

        Returns:
            A tensor containing the repulsive interaction energy evaluated at the
            specified distances.
        """
        results = torch.zeros_like(distances)
        c = self.coefficients
        z1, z2, a1, a2 = c[0], c[1], c[2], c[3]
        
        # Apply a mask to compute interactions only within the cutoff radius
        mask = distances < self.cutoff

        # A small tolerance for floating-point comparison
        if torch.abs(a1 - a2) < 1e-6:
            results[mask] = self._equal_gamma(distances[mask], a1)
        else:
            results[mask] = self._unequal_gamma(distances[mask], a1, a2)

        # Scale the result by the product of the effective nuclear charges
        return results * z1 * z2

# ####################################################################
# --- NEUE HILFSFUNKTION ---
# Diese Funktion bereitet das Input-Dictionary für PairwiseRepulsiveEnergyFeed vor.
# ####################################################################

def pairwise_repulsive(geometry: Geometry, alpha: dict, z: dict, repulsive_class: Type[Feed], cutoff: dict) -> ModuleDict:
    """
    Erstellt ein ModuleDict mit paarweisen abstoßenden Feeds für eine bestimmte Geometrie.
    Diese Version kann sowohl mit einfachen Zahlen (floats) als auch mit
    bestehenden Tensoren (torch.nn.Parameter) umgehen.
    """
    feeds_dict = ModuleDict({})
    for species_pair, _, _ in atomic_pair_distances(geometry, True, True):
        z1_num, z2_num = species_pair[0].item(), species_pair[1].item()
        pair_key = str(tuple(sorted((z1_num, z2_num))))

        # =======================================================================
        # KORRIGIERTER ABSCHNITT
        # =======================================================================
        # Wandle die Parameter NUR DANN in einen Tensor um, wenn sie es nicht schon sind.
        # Das macht die Funktion sowohl für das Training als auch für die Analyse kompatibel.
        
        # Sicherstellen, dass die Werte als Tensoren vorliegen
        z1_val = z[z1_num]
        z2_val = z[z2_num]
        a1_val = alpha[z1_num]
        a2_val = alpha[z2_num]

        coefficients = [
            z1_val if torch.is_tensor(z1_val) else torch.tensor(z1_val, dtype=torch.float64),
            z2_val if torch.is_tensor(z2_val) else torch.tensor(z2_val, dtype=torch.float64),
            a1_val if torch.is_tensor(a1_val) else torch.tensor(a1_val, dtype=torch.float64),
            a2_val if torch.is_tensor(a2_val) else torch.tensor(a2_val, dtype=torch.float64)
        ]
        # =======================================================================

        if repulsive_class is xTBRepulsive:
            cond1, cond2 = z1_num <= 2, z2_num <= 2
            kb = 1.0 if cond1 and cond2 else 1.5
            coefficients.append(torch.tensor(kb, dtype=torch.float64))

        feeds_dict[pair_key] = repulsive_class(
            coefficients=coefficients,
            cutoff=cutoff[pair_key]
        )
    return feeds_dict

def parse_parameters(file_path):
    """
    Parses the extracted_parameters.txt file, correctly handling multiple
    parameter sets for each model.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Split the entire content into model blocks.
    model_texts = re.split(r'\n(?=PTBP|GAMMA|XTB)', content)

    data = {}
    for model_text in model_texts:
        if not model_text.strip():
            continue

        model_name_match = re.match(r'^(XTB|PTBP|GAMMA)', model_text.strip())
        if not model_name_match:
            continue
        model_name = model_name_match.group(1)
        
        if model_name not in data:
            data[model_name] = {'Alpha': [], 'Z': []}

        parameter_sets = model_text.split('--')

        for param_set_text in parameter_sets:
            alpha_match = re.search(r'Alpha:\s*({[^}]+})', param_set_text)
            z_match = re.search(r'Z:\s*({[^}]+})', param_set_text)

            if alpha_match and z_match:
                try:
                    alpha_dict_str = alpha_match.group(1).replace('\n', '').replace(' ', '')
                    z_dict_str = z_match.group(1).replace('\n', '').replace(' ', '')
                    
                    alpha_dict = eval(alpha_dict_str)
                    z_dict = eval(z_dict_str)

                    data[model_name]['Alpha'].append(alpha_dict)
                    data[model_name]['Z'].append(z_dict)
                except (SyntaxError, NameError) as e:
                    print(f"Could not parse a dictionary in {model_name}: {e}")
    
    return data


def plot_repulsive_feeds(parameters, spline_feed_instance, atom_pair=(1, 8), cutoff=5.0, param_index=0):
    """
    Stellt die abstoßende Energie gegen die Entfernung dar.
    """
    plt.figure(figsize=(10, 6))
    
    # Standard-Startdistanz ist 1.0 Å
    start_distance = 1.0
    # Sonderfall für H-O (1-8) Paare
    if tuple(sorted(atom_pair)) == (1, 8) or tuple(sorted(atom_pair)) == (1,6) or tuple(sorted(atom_pair)) == (1,7):
        start_distance = 0.5

    distances_angstrom = torch.linspace(start_distance, cutoff, 500)

    ANGSTROM_TO_BOHR = 1.8897259886
    distances_bohr = distances_angstrom * ANGSTROM_TO_BOHR

    model_map = {'XTB': xTBRepulsive, 'PTBP': PTBPRepulsive, 'GAMMA': DFTBGammaRepulsive}
    param_set_label = f"(Set {param_index})"
    
    # Plot für analytische Modelle
    for model_name, repulsive_class in model_map.items():
        if parameters and model_name in parameters:
            if param_index >= len(parameters[model_name]['Alpha']):
                continue

            alpha_params = parameters[model_name]['Alpha'][param_index]
            z_params = parameters[model_name]['Z'][param_index]
            z1_num, z2_num = atom_pair

            if z1_num not in alpha_params or z2_num not in alpha_params:
                continue
            
            # Koeffizienten-Setup
            coefficients = [
                torch.tensor(z_params[z1_num]),
                torch.tensor(z_params[z2_num]),
                torch.tensor(alpha_params[z1_num]),
                torch.tensor(alpha_params[z2_num])
            ]
            if model_name == 'XTB':
                cond1 = z1_num <= 2; cond2 = z2_num <= 2
                kb = 1.0 if cond1 and cond2 else 1.5
                coefficients.append(torch.tensor(kb))

            model_instance = repulsive_class(coefficients=coefficients, cutoff=torch.tensor(20.0))
            energies = model_instance.forward(distances_bohr)
            plt.plot(distances_angstrom.numpy(), energies.detach().numpy(), label=f'{model_name} {param_set_label}')

    # Plot für das Spline-Modell
    if spline_feed_instance:
        pair_key = str(tuple(sorted(atom_pair)))
        if pair_key in spline_feed_instance.repulsive_feeds:
            spline_model = spline_feed_instance.repulsive_feeds[pair_key]
            spline_energies = spline_model(distances_bohr)
            plt.plot(distances_angstrom.numpy(), spline_energies.detach().numpy(), label='Spline (from mio.h5)', linestyle='--', color='black')
        else:
            print(f"Spline-Daten für Atompaar {atom_pair} nicht gefunden.")

    plt.xlabel('Distance (Angstrom)')
    plt.ylabel('Repulsive Energy (Hartree)')
    plt.title(f'Repulsive Feeds for Atom Pair {atom_pair} - Parameter Set {param_index}')
    plt.legend()
    plt.grid(True)
    
    filename = f'repulsive_feeds_plot_set_{param_index}_{atom_pair[0]}-{atom_pair[1]}.png'
    plt.savefig(filename)
    plt.close()
    # print(f"Plot saved as {filename}") # Weniger Output in der Konsole

if __name__ == '__main__':
    
    parameters = parse_parameters('extracted_parameters.txt')

    print("\n" + "="*70)
    print("--- Erstellung der Plots für Atompaare ---")
    print("="*70)
    
    spline_feed_instance = None
    spline_file = 'mio.h5'
    if os.path.exists(spline_file):
        try:
            print(f"Lade Spline-Daten aus {spline_file}...")
            # Korrekte Initialisierung des Spline-Feeds
            spline_feed_instance = PairwiseRepulsiveEnergyFeed.from_database(
                path=spline_file, species=[1, 6, 7, 8])
        except Exception as e:
            print(f"Fehler beim Laden der Spline-Daten: {e}")
    else:
        print(f"Warnung: '{spline_file}' nicht gefunden. Spline-Modell wird übersprungen.")

    atomic_numbers_for_plots = [1, 6, 7, 8]
    all_atom_pairs = list(combinations_with_replacement(atomic_numbers_for_plots, 2))
    
    num_param_sets_plot = 0
    if parameters:
        first_model = next(iter(parameters))
        num_param_sets_plot = len(parameters[first_model]['Alpha'])

    if num_param_sets_plot > 0:
        print(f"\nBeginne mit der Erstellung von Plots für {len(all_atom_pairs)} Atompaare...")
        
        for pair in all_atom_pairs:
            for index in range(num_param_sets_plot):
                plot_repulsive_feeds(parameters, spline_feed_instance, atom_pair=pair, param_index=index)
        
        print("\nAlle Plots wurden erfolgreich erstellt.")
    
    print("\nAlle Operationen abgeschlossen.")