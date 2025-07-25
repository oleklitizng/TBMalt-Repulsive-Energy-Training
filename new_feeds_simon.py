from __future__ import annotations
import warnings
import re
import numpy as np
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
from tbmalt.physics.dftb.feeds import PairwiseRepulsiveEnergyFeed, RepulsiveSplineFeed
import matplotlib.pyplot as plt


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
            self, coefficients: Parameter, cutoff: Tensor):

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
            self, coefficients: Parameter, cutoff: Tensor):

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

    def __init__(self, coefficients: Parameter, cutoff: Tensor):
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

def pairwise_repulsive(Geometry, alpha, Z, Repulsive, cutoff):
    """
    Provides the input for PairwiseRepulsiveEnergyFeed.

    - For xTBRepulsive, 5 coefficients are passed: [Z1, Z2, a1, a2, kb]
    - For PTBPRepulsive & DFTBGammaRepulsive, 4 coefficients are passed: [Z1, Z2, a1, a2]

    Arguments:
        Geometry: Geometry of a system in tbmalt notation.
        alpha: Dictionary containing element-specific repulsion parameters.
        Z: Dictionary containing element-specific effective nuclear charges.
        Repulsive: The repulsion class to be used (e.g., xTBRepulsive).
        cutoff: Dictionary with the cutoff radii for each atom pair.

    Returns:
        A torch `ModuleDict` of pairwise, distance-dependent repulsive feeds.
    """
    Dict = ModuleDict({})
    for species_pair, _, _ in atomic_pair_distances(
        Geometry, True, True):

        z1_num = species_pair[0].item()
        z2_num = species_pair[1].item()
        pair_key = str((z1_num, z2_num))

        # Base coefficients used by all classes
        coefficients = [
            Z[z1_num],
            Z[z2_num],
            alpha[z1_num],
            alpha[z2_num]
        ]

        # Add the specific parameter 'kb' only for the xTBRepulsive class
        if Repulsive is xTBRepulsive:
            cond1 = z1_num <= 2
            cond2 = z2_num <= 2
            kb = 1.0 if cond1 and cond2 else 1.5
            coefficients.append(kb)

        # The class is now instantiated with the correct number of parameters
        Dict[pair_key] = Repulsive(
            coefficients=coefficients,
            cutoff=cutoff[pair_key]
        )
    return Dict

def parse_parameters(file_path):
    """
    Parses the extracted_parameters.txt file, correctly handling multiple
    parameter sets for each model.
    """
    with open(file_path, 'r') as f:
        # Clean up the content by removing source tags and newlines within dictionaries
        content = f.read()
        content = re.sub(r'\\s*', '', content)
        content = re.sub(r'(:\s*{\s*[\d\s:.,\w-]+\n)\s*', r'\1', content)


    # Split the entire content into model blocks. The model names act as delimiters.
    # The (?=...) is a lookahead that keeps the delimiter in the next split.
    model_texts = re.split(r'\n(?=PTBP|GAMMA)', content)

    data = {}
    for model_text in model_texts:
        if not model_text.strip():
            continue

        # The model name is the first word of the block
        model_name = model_text.strip().split()[0]
        data[model_name] = {'Alpha': [], 'Z': []}

        # Split the model's text into individual parameter sets using '--'
        parameter_sets = model_text.split('--')

        for param_set_text in parameter_sets:
            # Now, find the single Alpha and Z dict in this smaller text block
            alpha_match = re.search(r'Alpha:\s*({[^}]+})', param_set_text)
            z_match = re.search(r'Z:\s*({[^}]+})', param_set_text)

            if alpha_match and z_match:
                try:
                    # Use eval to convert the string representation of the dict to a real dict
                    alpha_dict_str = alpha_match.group(1).replace('\n', '')
                    z_dict_str = z_match.group(1).replace('\n', '')
                    
                    alpha_dict = eval(alpha_dict_str)
                    z_dict = eval(z_dict_str)

                    data[model_name]['Alpha'].append(alpha_dict)
                    data[model_name]['Z'].append(z_dict)
                except (SyntaxError, NameError) as e:
                    print(f"Could not parse a dictionary in {model_name}: {e}")
    
    return data


def plot_repulsive_feeds(parameters, spline_feed_instance, atom_pair=(1, 8), cutoff=5.0, param_index=0):
    """
    Stellt die abstoßende Energie gegen die Entfernung dar und verwendet
    dabei die exakte RepulsiveSplineFeed-Klasse.
    """
    plt.figure(figsize=(10, 6))
    distances = torch.linspace(0.1, cutoff, 500)

    # Die Modelle, die wir vergleichen wollen
    model_names = ['XTB', 'PTBP', 'GAMMA', 'Spline']
    
    # Label für die Legende für die analytischen Modelle
    param_set_label = f"(Set {param_index})"

    for model_name in model_names:
        # --- Spezielle Behandlung für das Spline-Modell ---
        if model_name == 'Spline':
            if spline_feed_instance is not None:
                # Überprüfen, ob Daten für dieses Paar im Spline-Feed vorhanden sind
                pair_frozenset = frozenset(atom_pair)
                if pair_frozenset not in spline_feed_instance.spline_data:
                    print(f"Spline-Daten für Atompaar {atom_pair} nicht gefunden. Überspringe.")
                    continue
                
                # Berechne die Energien, indem die _repulsive_calc-Methode für jeden Abstand aufgerufen wird
                spline_energies = [spline_feed_instance._repulsive_calc(r, atom_pair[0], atom_pair[1]) for r in distances]
                spline_energies_tensor = torch.tensor(spline_energies)
                
                plt.plot(distances.numpy(), spline_energies_tensor.detach().numpy(), label='Spline (from mio.h5)')
            continue # Weiter mit dem nächsten Modell

        # --- Bestehende Logik für die anderen Modelle ---
        if parameters and model_name in parameters:
            if param_index >= len(parameters[model_name]['Alpha']):
                print(f"Index {param_index} liegt außerhalb des Bereichs für {model_name}. Überspringe.")
                continue

            alpha_params = parameters[model_name]['Alpha'][param_index]
            z_params = parameters[model_name]['Z'][param_index]
            z1_num, z2_num = atom_pair

            if z1_num not in alpha_params or z2_num not in alpha_params:
                print(f"Atompaar {atom_pair} in Parametern für {model_name} nicht gefunden. Überspringe.")
                continue
            
            # Die Instanziierung der analytischen Modelle bleibt gleich
            repulsive_class = {'XTB': xTBRepulsive, 'PTBP': PTBPRepulsive, 'GAMMA': DFTBGammaRepulsive}[model_name]
            a1 = torch.tensor([alpha_params[z1_num]])
            a2 = torch.tensor([alpha_params[z2_num]])
            z1 = torch.tensor([z_params[z1_num]])
            z2 = torch.tensor([z_params[z2_num]])
            
            coefficients = [z1, z2, a1, a2]
            if model_name == 'XTB':
                cond1 = z1_num <= 2
                cond2 = z2_num <= 2
                kb = 1.0 if cond1 and cond2 else 1.5
                coefficients.append(torch.tensor([kb]))
            
            model_instance = repulsive_class(coefficients=coefficients, cutoff=cutoff)
            energies = model_instance.forward(distances)
            plt.plot(distances.numpy(), energies.detach().numpy(), label=f'{model_name} {param_set_label}')

    plt.xlabel('Distance (Angstrom)')
    plt.ylabel('Repulsive Energy (Hartree)')
    plt.title(f'Repulsive Feeds for Atom Pair {atom_pair} - Parameter Set {param_index}')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.1, 1)
    
    filename = f'repulsive_feeds_plot_set_{param_index}_{atom_pair[0]}-{atom_pair[1]}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as {filename}")

if __name__ == '__main__':
    from itertools import combinations_with_replacement
    import os

    # Laden der Parameter für die analytischen Modelle
    parameters = parse_parameters('results/extracted_parameters.txt')

    # Definieren der Atome, die uns interessieren
    atomic_numbers = [1, 6, 7, 8]

    # Erstellen EINER Instanz des Spline-Feeds für alle relevanten Atomsorten
    spline_feed_instance = None
    spline_file = 'mio.h5'
    if os.path.exists(spline_file):
        try:
            # Annahme: RepulsiveSplineFeed und seine Abhängigkeiten (Skf, Feed, etc.)
            # sind aus Ihrer tbmalt-Bibliothek importierbar.
            print(f"Lade Spline-Daten aus {spline_file}...")
            spline_feed_instance = RepulsiveSplineFeed.from_database(
                path=spline_file, species=atomic_numbers)
        except Exception as e:
            print(f"Fehler beim Laden der Spline-Daten aus {spline_file}: {e}")
    else:
        print(f"Warnung: Spline-Datei '{spline_file}' nicht gefunden. Spline-Modell wird übersprungen.")


    # Erstellen aller einzigartigen Atompaare
    all_atom_pairs = list(combinations_with_replacement(atomic_numbers, 2))
    
    num_param_sets = 0
    if parameters:
        first_model = next(iter(parameters))
        num_param_sets = len(parameters[first_model]['Alpha'])

    if num_param_sets > 0:
        print(f"\nBeginne mit der Erstellung von Plots für {len(all_atom_pairs)} Atompaare...")
        
        for pair in all_atom_pairs:
            # Für die analytischen Modelle durch alle Parametersätze loopen
            for index in range(num_param_sets):
                print(f"--> Erstelle Plot für Paar {pair} mit Parametersatz {index + 1}...")
                # Die Instanz des Spline-Feeds bei jedem Aufruf übergeben
                plot_repulsive_feeds(parameters, spline_feed_instance, atom_pair=pair, param_index=index)
    
    print("\nAlle Plots wurden erfolgreich erstellt.")