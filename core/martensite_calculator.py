"""
Martensite and retained austenite calculations
"""

import pandas as pd
import numpy as np


class MartensiteCalculator:
    """Calculates martensite and retained austenite using thermodynamic models."""

    def __init__(self, app):
        self.app = app

    def calculate(self, phase_data):
        """Calculate martensite and retained austenite fractions."""
        try:
            self.app.update_log("Calculating Ms temperature...")
            self.app.update_log("Calculating martensite fraction...")
            self.app.update_log("Calculating retained austenite...")

            # Check for required columns
            required = [
                "Mass_fraction_C_in_FCC_A1",
                "Mass_fraction_Mn_in_FCC_A1",
                "Mass_fraction_Si_in_FCC_A1",
                "Mass_fraction_Al_in_FCC_A1",
                "FCC_A1_Fraction"
            ]

            # Handle optional columns
            has_N = "Mass_fraction_N_in_FCC_A1" in phase_data.columns
            has_Cr = "Mass_fraction_Cr_in_FCC_A1" in phase_data.columns

            missing = [c for c in required if c not in phase_data.columns]
            if missing:
                self.app.update_log(f"WARNING: Missing required columns: {missing}")
                self.app.update_log("Using default values for missing elements...")

                for col in missing:
                    phase_data[col] = 0.0

            # Extract composition data
            wC = phase_data["Mass_fraction_C_in_FCC_A1"].fillna(0.0).values
            wN = phase_data["Mass_fraction_N_in_FCC_A1"].fillna(0.0).values if has_N else np.zeros_like(wC)
            wMn = phase_data["Mass_fraction_Mn_in_FCC_A1"].fillna(0.0).values
            wSi = phase_data["Mass_fraction_Si_in_FCC_A1"].fillna(0.0).values
            wAl = phase_data["Mass_fraction_Al_in_FCC_A1"].fillna(0.0).values
            wCr = phase_data["Mass_fraction_Cr_in_FCC_A1"].fillna(0.0).values if has_Cr else np.zeros_like(wC)
            f_gamma = phase_data["FCC_A1_Fraction"].fillna(0.0).values

            # Convert to wt% for formulas
            wC_pct = wC * 100.0
            wN_pct = wN * 100.0
            wMn_pct = wMn * 100.0
            wSi_pct = wSi * 100.0
            wAl_pct = wAl * 100.0
            wCr_pct = wCr * 100.0

            self.app.update_log(
                f"Sample compositions (wt%): C={wC_pct[0]:.3f}, Mn={wMn_pct[0]:.2f}, Si={wSi_pct[0]:.2f}")

            # Calculate Ms (Martensite Start Temperature) in °C
            Ms = (692 - 502 * np.sqrt(wC_pct + 0.86 * wN_pct) - 37 * wMn_pct -
                  14 * wSi_pct + 20 * wAl_pct - 11 * wCr_pct)

            # Calculate martensite fraction parameters
            alpha = 0.0231 - 0.0105 * wC_pct
            beta = 1.4304 - 1.1836 * wC_pct + 0.7527 * wC_pct ** 2

            # Quenching temperature
            Tq = getattr(self.app, 'quenching_temp', 25.0)
            self.app.update_log(f"Quenching temperature: {Tq}°C")

            # Calculate martensite fraction (Fm)
            delta_T = Ms - Tq
            exponent = alpha * np.power(np.maximum(delta_T, 0), beta)
            exponent = np.clip(exponent, 0, 100)  # Prevent overflow

            Fm = np.where(Ms > Tq,
                          (1 - np.exp(-exponent)) * f_gamma,
                          0.0)

            # Calculate retained austenite (RA)
            RA = f_gamma - Fm
            RA = np.maximum(RA, 0.0)  # Ensure non-negative

            # Add calculated columns
            phase_data['Ms'] = Ms
            phase_data['Fm'] = Fm
            phase_data['RA'] = RA

            # Calculate statistics
            Ms_valid = Ms[~np.isnan(Ms) & (f_gamma > 0)]
            Fm_valid = Fm[~np.isnan(Fm)]
            RA_valid = RA[~np.isnan(RA)]

            if len(Ms_valid) > 0:
                self.app.update_log(f"Ms range: {Ms_valid.min():.2f} to {Ms_valid.max():.2f} °C")
            if len(Fm_valid) > 0:
                self.app.update_log(f"Fm range: {Fm_valid.min():.4f} to {Fm_valid.max():.4f}")
            if len(RA_valid) > 0:
                self.app.update_log(f"RA range: {RA_valid.min():.4f} to {RA_valid.max():.4f}")

            self.app.update_log("M/A calculations complete")
            return phase_data

        except Exception as e:
            self.app.update_log(f"ERROR in M/A calculations: {str(e)}")
            import traceback
            self.app.update_log(traceback.format_exc())
            return None