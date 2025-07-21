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
from tbmalt.physics.dftb.feeds import PairwiseRepulsiveEnergyFeed



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
    """Repulsive in form of the DFTB-Gamma."""

    def __init__(self, coefficients: Parameter, cutoff: Tensor):
        super().__init__()
        self.coefficients = coefficients
        self.cutoff = cutoff

    def _Gamma(self, a, b, R):
        # Diese Funktion bleibt unverändert
        zaehler1 = b**4 * a
        nenner1 = 2 * (a**2 - b**2)**2
        zaehler2 = b**6 - 3 * b**4 * a**2
        nenner2 = (a**2 - b**2)**3 * R
        result = zaehler1 / nenner1 - zaehler2 / nenner2
        return result

    def _equal_gamma(self, distances, a1):
        """
        Numerisch STABILE Implementierung für den Fall gleicher Parameter.
        Vermeidet den katastrophalen Genauigkeitsverlust.
        """
        # Der Polynom-Teil OHNE den problematischen 1/R-Term
        poly_part = (11 * a1 / 16) + (3 * a1**2 * distances / 16) + (a1**3 * distances**2 / 48)

        # Der (1/R - exp(-aR)/R) Teil wird jetzt stabil berechnet.
        # Die Funktion torch.expm1(x) berechnet präzise (exp(x) - 1) für kleines x.
        # (1 - exp(-aR))/R  ==>  -expm1(-aR)/R
        stable_term_1_over_R = -torch.expm1(-a1 * distances) / distances
        
        # Der zweite Teil der Formel
        exp_term = torch.exp(-a1 * distances) * poly_part
        
        # Das stabile Endergebnis
        return stable_term_1_over_R - exp_term

    def _unequal_gamma(self, distances, a1, a2):
        """
        Implementierung für ungleiche Parameter.
        Diese Methode profitiert vom Epsilon-Trick in der forward-Methode.
        """
        exp1 = torch.exp(-a1 * distances)
        exp2 = torch.exp(-a2 * distances)
        term1 = 1 / distances
        term2 = exp1 * self._Gamma(a1, a2, distances)
        term3 = exp2 * self._Gamma(a2, a1, distances)
        return term1 - term2 - term3

    def forward(self, distances: Tensor) -> Tensor:
        """Haupt-Berechnungsfunktion mit Stabilitäts-Fixes."""
        results = torch.zeros_like(distances)
        c = self.coefficients
        z1, z2, a1, a2 = c[0], c[1], c[2], c[3]
        mask = distances < self.cutoff

        # Epsilon, um die _unequal_gamma-Methode bei kleinen Abständen abzusichern
        epsilon = 1e-7
        stable_distances = distances[mask] + epsilon

        if torch.abs(a1 - a2) < 1e-6:
            # Hier verwenden wir die neue, stabile _equal_gamma-Methode.
            # Wir übergeben die originalen, nicht-stabilisierten distances,
            # da die Methode nun von sich aus stabil ist.
            results[mask] = self._equal_gamma(distances[mask], a1)
        else:
            # Hier verwenden wir die stabilisierten Abstände für den
            # numerisch heikleren Fall.
            results[mask] = self._unequal_gamma(stable_distances, a1, a2)

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


if __name__ == '__main__':

    alpha = {
        1: Parameter(Tensor([2.0]),requires_grad = True),
        8: Parameter(Tensor([2.0]),requires_grad = True)
    }

    Z = {
        1: Parameter(Tensor([1.0]),requires_grad = True),
        8: Parameter(Tensor([8.0]),requires_grad = True)
    }

    cutoff = {}
    
    H2O_geo = Geometry(torch.tensor([8, 1, 1]), 
               torch.tensor([[0.0, -1.0, 0.0],
                             [0.0, 0.0, 0.78306400000],
                             [0.0, 0.0, -0.78306400000]], requires_grad=False),
               units='angstrom'
               )
    
    for species_pair, _, _ in atomic_pair_distances(
        H2O_geo, True, True):
        cutoff[str((species_pair[0].item(), species_pair[1].item()))
               ]= Tensor([5.0])

    xTB_pair_repulsive = pairwise_repulsive(H2O_geo, alpha, Z, xTBRepulsive, cutoff)

    xTB_total_repulsive = PairwiseRepulsiveEnergyFeed(xTB_pair_repulsive)

    print(xTB_total_repulsive.forward(H2O_geo))

    #PTBP_pair_repulsive = pairwise_repulsive(H2O_geo, alpha, Z, PTBPRepulsive)

    #PTBP_total_repulsive = PairwiseRepulsiveEnergyFeed(PTBP_pair_repulsive)
    
    #print(PTBP_total_repulsive.forward(H2O_geo))

    """
    Gamma = DFTBGammaRepulsive([Parameter(Tensor([0.2236])),
                                Parameter(Tensor([0.2236])),
                                Parameter(Tensor([0.2580])),
                                Parameter(Tensor([0.2580]))], 
                                60.0)
    xTB = xTBRepulsive([Parameter(Tensor([0.2722])),
                                Parameter(Tensor([0.2722])),
                                Parameter(Tensor([1.5193])),
                                Parameter(Tensor([1.5193])), 
                                1.0], 
                                60.0)
    PTBP = PTBPRepulsive([Parameter(Tensor([0.6246])),
                                Parameter(Tensor([0.6246])),
                                Parameter(Tensor([0.5649])),
                                Parameter(Tensor([0.5649]))], 
                                60.0)
    
    r = torch.arange(1, 10, 0.1)
    
    a = Gamma.forward(Tensor(r))
    b = xTB.forward(Tensor(r))
    c = PTBP.forward(Tensor(r))

    import matplotlib.pyplot as plt
    import numpy
    fig, ax = plt.subplots()
    ax.plot(r.numpy(), a.detach().numpy(), 'r', label = 'Gamma')
    ax.plot(r.numpy(), b.detach().numpy(), 'b', label = 'xTB')
    ax.plot(r.numpy(), c.detach().numpy(), 'g', label = 'PTBP')
    ax.set_xlabel('distance [bohr]')
    ax.set_ylabel('repulsive energy [Ha]')
    ax.set_title('H-H repulsive optimized for a small batch of molecules')

    ax.legend()

    plt.show()
    """
    