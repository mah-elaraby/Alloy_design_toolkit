"""
Stacking Fault Energy calculations
"""

import pandas as pd
import numpy as np
import importlib.util
import os

class SFECalculator:
    """Calculates Stacking Fault Energy using thermodynamic models."""

    def __init__(self, app):
        self.app = app

    def calculate(self, ma_data):
        """Calculate SFE for each composition."""
        try:
            self.app.update_log(f"Using advanced parameters: σ={self.app.sfe_params['sigma']} mJ/m²")
            self.app.update_log(f"Grain size: {self.app.sfe_params['grain_size']} μm")
            self.app.update_log(f"Temperature: {self.app.sfe_params['temperature']} K")
            self.app.update_log("Calculating SFE for each composition...")

            # Import SFE calculator dynamically
            try:
                spec = importlib.util.spec_from_file_location(
                    "sfe_model",
                    os.path.join('standalone_scripts', '3 SFE model.py')
                )
                sfe_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sfe_module)
                SFECalculator = sfe_module.SFECalculator
                self.app.update_log("SFE calculator module loaded successfully")
            except Exception as e:
                self.app.update_log(f"ERROR: Could not load SFE calculator: {e}")
                return None

            # Check for required columns
            required_cols = [
                'Mass_fraction_C_in_FCC_A1',
                'Mass_fraction_Mn_in_FCC_A1',
                'Mass_fraction_Si_in_FCC_A1',
                'Mass_fraction_Al_in_FCC_A1',
                'Mass_fraction_Mo_in_FCC_A1'
            ]

            # Optional columns
            optional_cols = [
                'Mass_fraction_N_in_FCC_A1',
                'Mass_fraction_Cr_in_FCC_A1',
                'Mass_fraction_Ni_in_FCC_A1',
                'Mass_fraction_Cu_in_FCC_A1'
            ]

            missing = [c for c in required_cols if c not in ma_data.columns]
            if missing:
                self.app.update_log(f"WARNING: Missing required columns: {missing}")
                return None

            # Add optional columns if missing
            for col in optional_cols:
                if col not in ma_data.columns:
                    element = col.split('_')[2]
                    ma_data[col] = 0.0
                    self.app.update_log(f"Note: {element} not present, using 0.0")

            # Prepare advanced parameters
            advanced_params = {
                'sigma': self.app.sfe_params['sigma'],
                'grain_size': self.app.sfe_params['grain_size'],
                'lattice_param': self.app.sfe_params['lattice_param'],
                'temperature': self.app.sfe_params['temperature']
            }

            # Calculate SFE for each row
            sfe_values = []
            total_rows = len(ma_data)

            for idx, row in ma_data.iterrows():
                try:
                    # Extract composition (convert to wt%)
                    composition = {
                        'C': row.get('Mass_fraction_C_in_FCC_A1', 0.0) * 100.0,
                        'Mn': row.get('Mass_fraction_Mn_in_FCC_A1', 0.0) * 100.0,
                        'Si': row.get('Mass_fraction_Si_in_FCC_A1', 0.0) * 100.0,
                        'Al': row.get('Mass_fraction_Al_in_FCC_A1', 0.0) * 100.0,
                        'Mo': row.get('Mass_fraction_Mo_in_FCC_A1', 0.0) * 100.0,
                        'N': row.get('Mass_fraction_N_in_FCC_A1', 0.0) * 100.0,
                        'Cr': row.get('Mass_fraction_Cr_in_FCC_A1', 0.0) * 100.0,
                        'Ni': row.get('Mass_fraction_Ni_in_FCC_A1', 0.0) * 100.0,
                        'Cu': row.get('Mass_fraction_Cu_in_FCC_A1', 0.0) * 100.0
                    }

                    # Only calculate if austenite present
                    if row.get('FCC_A1_Fraction', 0.0) > 0:
                        calculator = SFECalculator(
                            advanced_params['temperature'],
                            composition,
                            advanced_params
                        )
                        sfe = calculator.calculate()
                    else:
                        sfe = np.nan

                    sfe_values.append(sfe)

                    if (idx + 1) % 100 == 0 or (idx + 1) == total_rows:
                        self.app.update_log(f"Progress: {idx + 1}/{total_rows} rows")

                except Exception as e:
                    self.app.update_log(f"Warning: SFE calculation failed for row {idx}: {e}")
                    sfe_values.append(np.nan)

            # Add SFE column
            ma_data['SFE'] = sfe_values

            # Calculate statistics
            valid_sfe = [s for s in sfe_values if not np.isnan(s)]
            if valid_sfe:
                self.app.update_log(f"SFE range: {min(valid_sfe):.2f} to {max(valid_sfe):.2f} mJ/m²")
                self.app.update_log(f"SFE mean: {np.mean(valid_sfe):.2f} mJ/m²")
                self.app.update_log(f"Valid calculations: {len(valid_sfe)}/{total_rows}")
            else:
                self.app.update_log("WARNING: No valid SFE values calculated")

            self.app.update_log("SFE calculations complete")
            return ma_data

        except Exception as e:
            self.app.update_log(f"ERROR in SFE calculations: {str(e)}")
            import traceback
            self.app.update_log(traceback.format_exc())
            return None