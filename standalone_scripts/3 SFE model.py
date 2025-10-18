#!/usr/bin/env python3
"""
Stacking Fault Energy (SFE) Calculator

A GUI application for calculating the stacking fault energy of austenitic steels
based on their chemical composition. The application supports single composition
calculations, batch processing from Excel files, and parametric studies.


"""

import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import pandas as pd
import os
import sys


# ------------------------------------------------------------------
# Physical Constants Class
# ------------------------------------------------------------------
class PhysicalConstants:
    """Physical constants used in SFE calculations."""

    # Universal gas constant in J/(mol*K)
    R = 8.3144621

    # Avogadro's number in particles/mol
    NA = 6.0221413e23

    # Magnetic contribution constants
    p = 0.28
    d = 2.342456517


# ------------------------------------------------------------------
# Element Properties Class
# ------------------------------------------------------------------
class ElementProperties:
    """Properties of chemical elements used in SFE calculations."""

    # Atomic weights of elements
    ATOMIC_WEIGHTS = {
        'C': 12.011,
        'Fe': 55.847,
        'Mn': 54.93805,
        'Al': 26.981539,
        'Cr': 51.9961,
        'Ni': 58.6934,
        'Cu': 63.546,
        'Si': 28.0855,
        'Mo': 95.95,
        'N': 14.00674
    }


# ------------------------------------------------------------------
# Default Parameters Class
# ------------------------------------------------------------------
class DefaultParameters:
    """Default parameters for SFE calculations."""

    # Interfacial energy in mJ/m²
    σ = 10

    # Grain size in μm
    GRAIN_SIZE = 2

    # Lattice parameter in meters
    LATTICE_PARAM = 3.6e-10

    # Default temperature in K
    TEMPERATURE = 298

    # Version information
    VERSION = "1.0"
    DEVELOPER = "Developed by Mahmoud Elaraby"


# ------------------------------------------------------------------
# Composition Class
# ------------------------------------------------------------------
class Composition:
    """
    Class for handling steel composition and related calculations.

    Handles conversion between weight percentages and mole fractions.
    """

    def __init__(self, composition_dict):
        """
        Initialize with composition in weight percentages.

        Parameters:
        -----------
        composition_dict : dict
            Dictionary containing element compositions in weight percent:
            {'C': float, 'Mn': float, 'Si': float, 'Al': float,
             'Mo': float, 'N': float, 'Cr': float, 'Ni': float, 'Cu': float}
        """
        self.wt_pct = composition_dict.copy()

        # Calculate Fe as remainder
        total_alloying = sum(self.wt_pct.values())
        self.wt_pct['Fe'] = 100.0 - total_alloying

        # Calculate mole fractions
        self._calculate_mole_fractions()

    def _calculate_mole_fractions(self):
        """Calculate mole fractions from weight percentages."""
        # Convert weight percentages to moles
        moles = {}
        for element, wt_pct in self.wt_pct.items():
            atomic_weight = ElementProperties.ATOMIC_WEIGHTS.get(element, 0)
            if atomic_weight == 0:
                moles[element] = 0
            else:
                moles[element] = wt_pct / atomic_weight

        # Calculate total moles
        total_moles = sum(moles.values())
        if total_moles == 0:
            raise ValueError("Total moles cannot be zero. Check your composition input.")

        # Calculate mole fractions
        self.mole_fraction = {element: moles[element] / total_moles for element in moles}

    def get_mole_fraction(self, element):
        """Get mole fraction for a specific element."""
        return self.mole_fraction.get(element, 0.0)


# ------------------------------------------------------------------
# Thermodynamic Parameters Class
# ------------------------------------------------------------------
class ThermodynamicParameters:
    """
    Class for calculating thermodynamic parameters used in SFE calculations.

    Includes Gibbs free energies, interaction parameters, and other
    thermodynamic quantities.
    """

    def __init__(self, composition, temperature):
        """
        Initialize with composition and temperature.

        Parameters:
        -----------
        composition : Composition
            Composition object containing mole fractions

        temperature : float
            Temperature in Kelvin
        """
        self.composition = composition
        self.T = temperature

        # Calculate all thermodynamic parameters
        self._calculate_gibbs_energies()
        self._calculate_interaction_parameters()
        self._calculate_nitrogen_interaction_energies()

    def _calculate_gibbs_energies(self):
        """Calculate Gibbs free energies of pure elements in austenite."""
        self.ΔG = {
            'Fe': -2243.38 + 4.309 * self.T,
            'C': -24595.12,
            'Mn': -1000 + 1.123 * self.T,
            'Al': 5481.04 - 1.799 * self.T,
            'Si': -560 - 8 * self.T,
            'Ni': 1046 + 1.2552 * self.T,
            'Cr': 1370 - 0.163 * self.T,
            'Mo': -3650 + 0.63 * self.T,
            'Cu': 600 + 0.2 * self.T
        }

    def _calculate_interaction_parameters(self):
        """Calculate interaction parameters between elements."""
        # Shorthand for mole fractions
        Fe_f = self.composition.get_mole_fraction('Fe')
        Mn_f = self.composition.get_mole_fraction('Mn')
        Si_f = self.composition.get_mole_fraction('Si')

        # Interaction parameters (J/mol)
        self.ω = {
            'FeC': 42500,
            'FeMn': 2873 - 717 * (Fe_f - Mn_f),
            'FeAl': 3323,
            'FeSi': 2850 + 3520 * (Fe_f - Si_f),
            'FeNi': 0,
            'FeCr': 2095,
            'CrNi': 4190,
            'MnC': 26910,
            'FeMo': 0
        }

    def _calculate_nitrogen_interaction_energies(self):
        """Calculate interaction energies for nitrogen."""
        # Interaction energy terms for nitrogen (J/mol)
        self.U_g = {
            'FeCr': 18800,
            'FeNi': -17000,
            'CrNi': -35800,
            'FeMn': 8800,
            'CrMn': -10000,
            'NiMn': 25800
        }

        # Calculate excess interaction energies
        self.U_e = {
            'FeCr': self.U_g['FeCr'] - (self.ΔG['Fe'] - self.ΔG['Cr']),
            'FeNi': self.U_g['FeNi'] - (self.ΔG['Fe'] - self.ΔG['Ni']),
            'CrNi': self.U_g['CrNi'] - (self.ΔG['Cr'] - self.ΔG['Ni']),
            'FeMn': self.U_g['FeMn'] - (self.ΔG['Fe'] - self.ΔG['Mn']),
            'CrMn': self.U_g['CrMn'] - (self.ΔG['Cr'] - self.ΔG['Mn']),
            'NiMn': self.U_g['NiMn'] - (self.ΔG['Ni'] - self.ΔG['Mn'])
        }


# ------------------------------------------------------------------
# SFE Calculator Class
# ------------------------------------------------------------------
class SFECalculator:
    """
    Core class for calculating Stacking Fault Energy.

    Implements the thermodynamic model for SFE calculation based on
    chemical composition and physical parameters.
    """

    def __init__(self, temperature, composition_dict, advanced_params=None):
        """
        Initialize the SFE calculator.

        Parameters:
        -----------
        temperature : float
            Temperature in Kelvin

        composition_dict : dict
            Dictionary containing element compositions in weight percent:
            {'C': float, 'Mn': float, 'Si': float, 'Al': float,
             'Mo': float, 'N': float, 'Cr': float, 'Ni': float, 'Cu': float}

        advanced_params : dict, optional
            Dictionary containing advanced parameters:
            {'sigma': float,       # Interfacial energy in mJ/m²
             'grain_size': float,  # Grain size in μm
             'lattice_param': float} # Lattice parameter in meters
        """
        # Set advanced parameters or use defaults
        if advanced_params is None:
            advanced_params = {}

        self.σ = advanced_params.get('sigma', DefaultParameters.σ)
        self.grain_size = advanced_params.get('grain_size', DefaultParameters.GRAIN_SIZE)
        self.a = advanced_params.get('lattice_param', DefaultParameters.LATTICE_PARAM)
        self.T = temperature

        # Initialize composition
        self.composition = Composition(composition_dict)

        # Initialize thermodynamic parameters
        self.thermo = ThermodynamicParameters(self.composition, self.T)

    def calculate(self):
        """
        Calculate the Stacking Fault Energy.

        Returns:
        --------
        float
            Calculated SFE value in J/m²
        """
        # Shorthand for mole fractions
        C_f = self.composition.get_mole_fraction('C')
        Fe_f = self.composition.get_mole_fraction('Fe')
        Mn_f = self.composition.get_mole_fraction('Mn')
        Al_f = self.composition.get_mole_fraction('Al')
        Cr_f = self.composition.get_mole_fraction('Cr')
        Ni_f = self.composition.get_mole_fraction('Ni')
        Cu_f = self.composition.get_mole_fraction('Cu')
        Si_f = self.composition.get_mole_fraction('Si')
        Mo_f = self.composition.get_mole_fraction('Mo')
        N_f = self.composition.get_mole_fraction('N')

        # Calculate nitrogen bulk energy
        ΔG_Nbulk = self._calculate_nitrogen_bulk_energy(Fe_f, Mn_f, Cr_f, Ni_f, N_f)

        # Calculate chemical and interaction free energies
        ΔG_chem = self._calculate_chemical_free_energy(
            Fe_f, C_f, Mn_f, Al_f, Cr_f, Ni_f, Mo_f, Cu_f, Si_f
        )

        ΔG_int = self._calculate_interaction_energy(
            Fe_f, Mn_f, Al_f, Cr_f, Ni_f, Si_f, C_f
        )

        # Calculate magnetic free energy difference
        ΔG_mag = self._calculate_magnetic_energy(Fe_f, Mn_f, Cr_f, Ni_f, C_f, Si_f, Al_f)

        # Calculate density from lattice parameter
        ρ = self._calculate_density()

        # Calculate nitrogen surface energy terms
        λ_N, X_Ns = self._calculate_nitrogen_surface_parameters(N_f)
        ΔG_chem_N = self._calculate_nitrogen_chemical_energy(N_f, X_Ns)
        ΔG_sur = self._calculate_nitrogen_surface_energy(λ_N, N_f, X_Ns)

        # Calculate total Gibbs free energy
        ΔG_total = 1000 * (ΔG_chem + ΔG_int + ΔG_mag)

        # Calculate grain size effect (strain energy)
        ΔG_exgs = self._calculate_grain_size_effect()

        # Calculate partial energy terms
        γ_n = 2000 * ρ * (ΔG_chem_N + ΔG_sur) if ρ != 0 else 0
        γ_nb = 2000 * ρ * ΔG_Nbulk if ρ != 0 else 0

        # Calculate basic stacking fault energy
        γ_basic = 2 * ρ * ΔG_total + 2 * self.σ + 2 * ρ * ΔG_exgs if ρ != 0 else 0

        # Calculate total SFE
        γ_total = γ_basic + γ_nb + γ_n

        return γ_total

    def _calculate_nitrogen_bulk_energy(self, Fe_f, Mn_f, Cr_f, Ni_f, N_f):
        """Calculate nitrogen bulk energy contribution."""
        # Calculate exponential terms for nitrogen bulk energy
        exp_terms = self._calculate_nitrogen_exp_terms(Fe_f, Mn_f, Cr_f, Ni_f)

        # Calculate nitrogen bulk energy
        ΔG_Nbulk = 6 * N_f * (
                self.thermo.ΔG['Mn'] +
                (self.thermo.U_e['FeMn'] * Fe_f) / exp_terms['term1'] +
                (self.thermo.U_e['CrMn'] * Cr_f) / exp_terms['term2'] +
                (self.thermo.U_e['NiMn'] * Ni_f) / exp_terms['term3']
        )

        return ΔG_Nbulk

    def _calculate_nitrogen_exp_terms(self, Fe_f, Mn_f, Cr_f, Ni_f):
        """Calculate exponential terms for nitrogen energy calculations."""
        exp_terms = {
            'term1': (
                    Fe_f +
                    Mn_f * np.exp(-self.thermo.U_e['FeMn'] / (PhysicalConstants.R * self.T)) +
                    Cr_f * np.exp(-self.thermo.U_e['FeCr'] / (PhysicalConstants.R * self.T)) +
                    Ni_f * np.exp(-self.thermo.U_e['FeNi'] / (PhysicalConstants.R * self.T))
            ),
            'term2': (
                    Fe_f * np.exp(self.thermo.U_e['FeCr'] / (PhysicalConstants.R * self.T)) +
                    Mn_f * np.exp(-self.thermo.U_e['CrMn'] / (PhysicalConstants.R * self.T)) +
                    Cr_f +
                    Ni_f * np.exp(-self.thermo.U_e['CrNi'] / (PhysicalConstants.R * self.T))
            ),
            'term3': (
                    Fe_f * np.exp(self.thermo.U_e['FeNi'] / (PhysicalConstants.R * self.T)) +
                    Mn_f * np.exp(-self.thermo.U_e['NiMn'] / (PhysicalConstants.R * self.T)) +
                    Cr_f * np.exp(self.thermo.U_e['CrNi'] / (PhysicalConstants.R * self.T)) +
                    Ni_f
            )
        }

        return exp_terms

    def _calculate_chemical_free_energy(self, Fe_f, C_f, Mn_f, Al_f, Cr_f, Ni_f, Mo_f, Cu_f, Si_f):
        """Calculate chemical free energy contribution."""
        ΔG_chem = (
                Fe_f * self.thermo.ΔG['Fe'] +
                C_f * self.thermo.ΔG['C'] +
                Mn_f * self.thermo.ΔG['Mn'] +
                Al_f * self.thermo.ΔG['Al'] +
                Cr_f * self.thermo.ΔG['Cr'] +
                Ni_f * self.thermo.ΔG['Ni'] +
                Mo_f * self.thermo.ΔG['Mo'] +
                Cu_f * self.thermo.ΔG['Cu'] +
                Si_f * self.thermo.ΔG['Si']
        )

        return ΔG_chem

    def _calculate_interaction_energy(self, Fe_f, Mn_f, Al_f, Cr_f, Ni_f, Si_f, C_f):
        """Calculate interaction energy contribution."""
        ΔG_int = (
                Fe_f * Mn_f * self.thermo.ω['FeMn'] +
                Fe_f * Al_f * self.thermo.ω['FeAl'] +
                Fe_f * Cr_f * self.thermo.ω['FeCr'] +
                Fe_f * Ni_f * self.thermo.ω['FeNi'] +
                Fe_f * Si_f * self.thermo.ω['FeSi'] +
                Cr_f * Ni_f * self.thermo.ω['CrNi'] +
                Mn_f * C_f * self.thermo.ω['MnC'] +
                Fe_f * C_f * self.thermo.ω['FeC']
        )

        return ΔG_int

    def _calculate_magnetic_energy(self, Fe_f, Mn_f, Cr_f, Ni_f, C_f, Si_f, Al_f):
        """Calculate magnetic free energy difference."""
        # Calculate magnetic contribution for gamma phase (austenite)
        denom_γ = (
                10 * Mn_f ** 3 +
                898.4 * Mn_f ** 2 +
                1176 * Mn_f -
                1992 * C_f -
                1272 * Si_f -
                661 * Al_f -
                170 * Cr_f +
                152.4
        )

        # Avoid division by zero
        τ_γ = self.T / denom_γ if denom_γ != 0 else 0

        # Calculate f(τ) for gamma phase
        f_γ = self._calculate_f_tau(τ_γ)

        # Calculate magnetic term for gamma phase
        term_γ = (
                0.7 * Fe_f +
                0.62 * Mn_f +
                0.62 * Ni_f -
                0.8 * Cr_f -
                0.64 * Fe_f * Mn_f -
                4 * C_f
        )

        G_mag_γ = PhysicalConstants.R * self.T * np.log(1 + term_γ) * f_γ if τ_γ != 0 else 0

        # Calculate magnetic contribution for epsilon phase (martensite)
        denom_ε = 580 * Mn_f
        τ_ε = self.T / denom_ε if denom_ε != 0 else 0

        # Calculate f(τ) for epsilon phase
        f_ε = self._calculate_f_tau(τ_ε)

        # Calculate magnetic term for epsilon phase
        term_ε = 0.62 * Mn_f - 4 * C_f
        G_mag_ε = PhysicalConstants.R * self.T * np.log(1 + term_ε) * f_ε if τ_ε != 0 else 0

        # Magnetic free energy difference
        ΔG_mag = G_mag_ε - G_mag_γ

        return ΔG_mag

    def _calculate_f_tau(self, τ):
        """Calculate f(τ) function for magnetic contribution."""
        p = PhysicalConstants.p
        d = PhysicalConstants.d

        if τ <= 1 and τ != 0:
            f_τ = 1 - (1 / d) * (
                    (79 / (140 * p)) * τ ** -1 +
                    (474 / 497) * (1 / p - 1) * (
                            τ ** 3 / 6 +
                            τ ** 9 / 135 +
                            τ ** 15 / 600
                    )
            )
        elif τ != 0:
            f_τ = -(
                    τ ** -5 / 10 +
                    τ ** -15 / 315 +
                    τ ** -25 / 1500
            ) / d
        else:
            f_τ = 0

        return f_τ

    def _calculate_density(self):
        """Calculate planar atomic density from lattice parameter."""
        if self.a == 0:
            return 0
        else:
            return 4 / (np.sqrt(3) * self.a ** 2 * PhysicalConstants.NA)

    def _calculate_nitrogen_surface_parameters(self, N_f):
        """Calculate nitrogen surface parameters."""
        # Lambda parameter for nitrogen
        λ_N = -20705 * N_f ** 2 + 23097 * N_f

        # Calculate surface nitrogen mole fraction
        X_Ns = 0
        if N_f > 0:
            denom_XNs = (1 - N_f) / N_f * np.exp(-λ_N / (PhysicalConstants.R * self.T))
            X_Ns = 1 / (1 + denom_XNs) if denom_XNs != 0 else 1.0

        return λ_N, X_Ns

    def _calculate_nitrogen_chemical_energy(self, N_f, X_Ns):
        """Calculate chemical driving force for nitrogen partitioning."""
        if N_f > 0 and (0 < X_Ns < 1):
            return PhysicalConstants.R * self.T * (
                    N_f * np.log(X_Ns / N_f) +
                    (1 - N_f) * np.log((1 - X_Ns) / (1 - N_f))
            )
        else:
            return 0

    def _calculate_nitrogen_surface_energy(self, λ_N, N_f, X_Ns):
        """Calculate nitrogen surface energy."""
        return 0.25 * λ_N * (X_Ns - N_f) ** 2

    def _calculate_grain_size_effect(self):
        """Calculate grain size effect (strain energy)."""
        if self.grain_size == 0:
            return 0
        else:
            return 1000 * 170.06 * np.exp(-self.grain_size / 18.55)


# ------------------------------------------------------------------
# Advanced Parameters Dialog Class
# ------------------------------------------------------------------
class AdvancedParametersDialog:
    """Dialog for setting advanced calculation parameters."""

    def __init__(self, parent, current_params):
        """
        Initialize the advanced parameters dialog.

        Parameters:
        -----------
        parent : tk.Tk or tk.Toplevel
            Parent window

        current_params : dict
            Current advanced parameters
        """
        self.window = tk.Toplevel(parent)
        self.window.title("Advanced Options")
        self.params = current_params.copy()

        self._build_gui()

    def _build_gui(self):
        """Build the dialog GUI."""
        # Create entry fields for advanced parameters
        params_info = [
            ("Interfacial energy σ (mJ/m²)", 'sigma', DefaultParameters.σ),
            ("Grain Size (μm)", 'grain_size', DefaultParameters.GRAIN_SIZE),
            ("Lattice parameter a (m)", 'lattice_param', DefaultParameters.LATTICE_PARAM),
            ("Temperature (K)", 'temperature', DefaultParameters.TEMPERATURE)
        ]

        self.entries = {}

        for i, (label_text, param_name, default_value) in enumerate(params_info):
            tk.Label(self.window, text=label_text).grid(row=i, column=0, padx=10, pady=5)
            entry = tk.Entry(self.window)
            entry.grid(row=i, column=1, padx=10, pady=5)
            entry.insert(0, str(self.params.get(param_name, default_value)))
            self.entries[param_name] = entry

        # Apply button
        tk.Button(
            self.window,
            text="Apply",
            command=self._apply_parameters
        ).grid(row=len(params_info), column=0, columnspan=2, pady=10)

    def _apply_parameters(self):
        """Apply the advanced parameters and close the window."""
        try:
            for param_name, entry in self.entries.items():
                self.params[param_name] = float(entry.get())
            self.window.destroy()
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values.")

    def get_parameters(self):
        """
        Get the parameters after dialog is closed.

        Returns:
        --------
        dict
            Updated parameters
        """
        return self.params


# ------------------------------------------------------------------
# Composition Range Dialog Class
# ------------------------------------------------------------------
class CompositionRangeDialog:
    """Dialog for parametric study with composition ranges."""

    def __init__(self, parent, advanced_params, callback):
        """
        Initialize the composition range dialog.

        Parameters:
        -----------
        parent : tk.Tk or tk.Toplevel
            Parent window

        advanced_params : dict
            Advanced parameters for calculations

        callback : function
            Function to call with results
        """
        self.window = tk.Toplevel(parent)
        self.window.title("Chemical Composition Range")
        # Create a deep copy of advanced_params to ensure no shared references
        self.advanced_params = advanced_params.copy()
        self.callback = callback

        self._build_gui()

    def _build_gui(self):
        """Build the dialog GUI."""
        # Create entry fields for element ranges
        elements = ['C', 'Mn', 'Si', 'Al', 'Mo', 'N', 'Cr', 'Ni', 'Cu']
        self.entries_range = {}

        for i, element in enumerate(elements):
            tk.Label(self.window, text=f"{element} Start:").grid(row=i, column=0, padx=10, pady=5)
            entry_start = tk.Entry(self.window)
            entry_start.grid(row=i, column=1, padx=10, pady=5)
            entry_start.insert(0, "0.0")

            tk.Label(self.window, text=f"{element} End:").grid(row=i, column=2, padx=10, pady=5)
            entry_end = tk.Entry(self.window)
            entry_end.grid(row=i, column=3, padx=10, pady=5)
            entry_end.insert(0, "0.0")

            tk.Label(self.window, text=f"{element} Step:").grid(row=i, column=4, padx=10, pady=5)
            entry_step = tk.Entry(self.window)
            entry_step.grid(row=i, column=5, padx=10, pady=5)
            entry_step.insert(0, "0.0")

            self.entries_range[element] = (entry_start, entry_end, entry_step)

        # Add advanced parameters display for verification
        row = len(elements)
        tk.Label(self.window, text="Current Advanced Parameters:").grid(row=row, column=0, columnspan=6, pady=5)
        row += 1

        # Display current advanced parameters
        params_text = f"Grain Size: {self.advanced_params.get('grain_size', DefaultParameters.GRAIN_SIZE)} μm, " + \
                      f"Interfacial Energy: {self.advanced_params.get('sigma', DefaultParameters.σ)} mJ/m², " + \
                      f"Temperature: {self.advanced_params.get('temperature', DefaultParameters.TEMPERATURE)} K"
        tk.Label(self.window, text=params_text).grid(row=row, column=0, columnspan=6, pady=5)
        row += 1

        # Calculate button
        tk.Button(
            self.window,
            text="Calculate",
            command=self._calculate_for_range
        ).grid(row=row, column=0, columnspan=6, pady=10)

    def _calculate_for_range(self):
        """Perform parametric study calculations."""
        try:
            # Get ranges for each element
            ranges = {}
            for element, (start_entry, end_entry, step_entry) in self.entries_range.items():
                start = float(start_entry.get())
                end = float(end_entry.get())
                step = float(step_entry.get())

                # If step is zero, fix element at start value
                if step == 0:
                    ranges[element] = [start]
                else:
                    # Create range including end if it's a multiple of step
                    ranges[element] = np.arange(start, end + step / 2, step)

            # Prepare for results
            results = []

            # Iterate over all combinations
            for C in ranges['C']:
                for Mn in ranges['Mn']:
                    for Si in ranges['Si']:
                        for Al in ranges['Al']:
                            for Mo in ranges['Mo']:
                                for N in ranges['N']:
                                    for Cr in ranges['Cr']:
                                        for Ni in ranges['Ni']:
                                            for Cu in ranges['Cu']:
                                                # Create composition dictionary
                                                composition = {
                                                    'C': C, 'Mn': Mn, 'Si': Si, 'Al': Al,
                                                    'Mo': Mo, 'N': N, 'Cr': Cr, 'Ni': Ni, 'Cu': Cu
                                                }

                                                # Calculate SFE
                                                calculator = SFECalculator(
                                                    self.advanced_params['temperature'],
                                                    composition,
                                                    self.advanced_params
                                                )
                                                sfe = calculator.calculate()

                                                # Add to results
                                                result_row = composition.copy()
                                                result_row['SFE (J/m²)'] = sfe
                                                # Add advanced parameters to results for verification
                                                result_row['Grain Size (μm)'] = self.advanced_params.get('grain_size',
                                                                                                         DefaultParameters.GRAIN_SIZE)
                                                results.append(result_row)

            # Call callback with results
            self.callback(results)
            self.window.destroy()

        except ValueError as e:
            messagebox.showerror("Input Error", f"Please enter valid numerical values: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


# ------------------------------------------------------------------
# GUI Application Class
# ------------------------------------------------------------------
class SFECalculatorApp:
    """
    GUI application for Stacking Fault Energy calculations.

    Provides interfaces for:
    - Single composition calculations
    - Batch processing from Excel files
    - Parametric studies with composition ranges
    """

    def __init__(self, root):
        """Initialize the application with the root window."""
        self.root = root
        self.root.title("Stacking Fault Energy Calculator")

        # Initialize advanced parameters with defaults
        self.advanced_params = {
            'sigma': DefaultParameters.σ,
            'grain_size': DefaultParameters.GRAIN_SIZE,
            'lattice_param': DefaultParameters.LATTICE_PARAM,
            'temperature': DefaultParameters.TEMPERATURE
        }

        self._build_gui()

    def _build_gui(self):
        """Build the graphical user interface."""
        # Version and developer info
        version_label = tk.Label(
            self.root,
            text=f"Version {DefaultParameters.VERSION}, {DefaultParameters.DEVELOPER}",
            font=("Arial", 10, "italic")
        )
        version_label.grid(row=0, column=0, columnspan=2, padx=10, pady=5)

        # Labels for composition inputs
        labels = [
            "Carbon (wt%)", "Manganese (wt%)", "Silicon (wt%)", "Aluminum (wt%)",
            "Molybdenum (wt%)", "Nitrogen (wt%)", "Chromium (wt%)", "Nickel (wt%)", "Copper (wt%)"
        ]

        # Create entry fields
        self.entries = {}
        element_symbols = ['C', 'Mn', 'Si', 'Al', 'Mo', 'N', 'Cr', 'Ni', 'Cu']

        for i, (label_text, symbol) in enumerate(zip(labels, element_symbols)):
            tk.Label(self.root, text=label_text).grid(row=i + 1, column=0, padx=10, pady=5)
            entry = tk.Entry(self.root)
            entry.grid(row=i + 1, column=1, padx=10, pady=5)
            entry.insert(0, "0.0")  # Default value
            self.entries[symbol] = entry

        # Buttons
        row = len(labels) + 1

        # Calculate button
        calculate_button = tk.Button(
            self.root,
            text="Calculate SFE",
            command=self.calculate_sfe
        )
        calculate_button.grid(row=row, column=0, columnspan=2, pady=10)

        # Advanced options button
        row += 1
        advanced_button = tk.Button(
            self.root,
            text="Advanced Options",
            command=self.open_advanced_options
        )
        advanced_button.grid(row=row, column=0, columnspan=2, pady=10)

        # Range calculation button
        row += 1
        range_button = tk.Button(
            self.root,
            text="Chemical Composition Range",
            command=self.open_chemical_composition_range
        )
        range_button.grid(row=row, column=0, columnspan=2, pady=10)

        # Excel import button
        row += 1
        excel_button = tk.Button(
            self.root,
            text="Import from Excel",
            command=self.import_from_excel
        )
        excel_button.grid(row=row, column=0, columnspan=2, pady=10)

        # Result label
        row += 1
        self.result_label = tk.Label(self.root, text="")
        self.result_label.grid(row=row, column=0, columnspan=2)

        # Advanced parameters display
        row += 1
        self.params_label = tk.Label(
            self.root,
            text=f"Grain Size: {self.advanced_params['grain_size']} μm"
        )
        self.params_label.grid(row=row, column=0, columnspan=2, pady=5)
        self._update_params_display()

    def _update_params_display(self):
        """Update the display of current advanced parameters."""
        params_text = f"Grain Size: {self.advanced_params['grain_size']} μm, " + \
                      f"Interfacial Energy: {self.advanced_params['sigma']} mJ/m², " + \
                      f"Temperature: {self.advanced_params['temperature']} K"
        self.params_label.config(text=params_text)

    def get_composition_from_entries(self):
        """Get composition values from entry fields."""
        composition = {}

        try:
            for element, entry in self.entries.items():
                composition[element] = float(entry.get())
            return composition
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values.")
            return None

    def calculate_sfe(self):
        """Calculate SFE for a single composition."""
        composition = self.get_composition_from_entries()
        if composition is None:
            return

        try:
            # Calculate SFE using the SFECalculator class
            calculator = SFECalculator(
                self.advanced_params['temperature'],
                composition,
                self.advanced_params
            )
            result = calculator.calculate()

            # Display result
            if result is not None:
                self.result_label.config(text=f"Stacking Fault Energy: {result:.2f} J/m²")
            else:
                messagebox.showerror("Calculation Error", "The SFE calculation returned an invalid result.")

        except Exception as e:
            messagebox.showerror("Calculation Error", f"An error occurred: {str(e)}")

    def open_advanced_options(self):
        """Open dialog for advanced parameter settings."""
        dialog = AdvancedParametersDialog(self.root, self.advanced_params)
        self.root.wait_window(dialog.window)
        self.advanced_params = dialog.get_parameters()
        self._update_params_display()

    def open_chemical_composition_range(self):
        """Open dialog for parametric study with composition ranges."""
        # Ensure we're using the most up-to-date advanced parameters
        dialog = CompositionRangeDialog(
            self.root,
            self.advanced_params.copy(),  # Pass a copy to prevent shared references
            self.handle_range_results
        )

    def handle_range_results(self, results):
        """Handle results from parametric study."""
        if results:
            filename = self._generate_unique_filename('sfe_results.xlsx')
            df = pd.DataFrame(results)
            df.to_excel(filename, index=False)
            messagebox.showinfo("Success", f"Results saved to {filename}")
        else:
            messagebox.showinfo("No Results", "No valid combinations were found.")

    def import_from_excel(self):
        """Import compositions from Excel and calculate SFE values."""
        try:
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select Excel File",
                filetypes=[("Excel files", "*.xlsx *.xls")]
            )

            if not file_path:
                return  # User cancelled

            # Read Excel file
            df = pd.read_excel(file_path)

            # Accept long Thermo-Calc-style column names by renaming to internal symbols

            column_map = {
                'Mass_fraction_C_in_FCC_A1': 'C',
                'Mass_fraction_Mn_in_FCC_A1': 'Mn',
                'Mass_fraction_Si_in_FCC_A1': 'Si',
                'Mass_fraction_Al_in_FCC_A1': 'Al',
                'Mass_fraction_Mo_in_FCC_A1': 'Mo',
                'Mass_fraction_N_in_FCC_A1': 'N',
                'Mass_fraction_Cr_in_FCC_A1': 'Cr',
                'Mass_fraction_Ni_in_FCC_A1': 'Ni',
                'Mass_fraction_Cu_in_FCC_A1': 'Cu',
            }
            df = df.rename(columns=column_map)

            # Check required columns
            required_columns = ['C', 'Mn', 'Si', 'Al', 'Mo', 'N', 'Cr', 'Ni', 'Cu']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                messagebox.showerror(
                    "Missing Columns",
                    f"The following required columns are missing: {', '.join(missing_columns)}"
                )
                return

            # Calculate SFE for each row
            df['SFE (J/m²)'] = df.apply(
                lambda row: SFECalculator(
                    self.advanced_params['temperature'],
                    {
                        'C': row['C'], 'Mn': row['Mn'], 'Si': row['Si'],
                        'Al': row['Al'], 'Mo': row['Mo'], 'N': row['N'],
                        'Cr': row['Cr'], 'Ni': row['Ni'], 'Cu': row['Cu']
                    },
                    self.advanced_params.copy()  # Pass a copy to prevent shared references
                ).calculate(),
                axis=1
            )

            # Save results
            output_file = self._generate_unique_filename('sfe_results_from_import.xlsx')
            df.to_excel(output_file, index=False)
            messagebox.showinfo("Success", f"SFE calculations complete. Results saved to {output_file}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def _generate_unique_filename(self, base_name):
        """Generate a unique filename to avoid overwriting existing files."""
        counter = 1
        name, ext = os.path.splitext(base_name)
        filename = base_name

        while os.path.exists(filename):
            filename = f"{name}_{counter}{ext}"
            counter += 1

        return filename


# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------
def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = SFECalculatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
