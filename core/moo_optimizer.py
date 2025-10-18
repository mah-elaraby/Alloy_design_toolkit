"""
Multi-Objective Optimization module
"""

import pandas as pd
import numpy as np
import importlib.util
import os
import sys

class MOOOptimizer:
    """Handles Multi-Objective Optimization for alloy design."""

    def __init__(self, app):
        self.app = app

    def optimize(self, sfe_data):
        """Run multi-objective optimization using MOO algorithm."""
        try:
            self.app.update_log("Loading MOO algorithm module...")

            # Import the MOO module
            moo_path = os.path.join('standalone_scripts', '4 MOO algorithm.py')

            if not os.path.exists(moo_path):
                self.app.update_log(f"ERROR: MOO algorithm file not found at {moo_path}")
                return None

            spec = importlib.util.spec_from_file_location("moo_algorithm", moo_path)
            moo_module = importlib.util.module_from_spec(spec)
            sys.modules['moo_algorithm'] = moo_module
            spec.loader.exec_module(moo_module)

            # Get the MOO class
            MediumMnSteelMOO = moo_module.MediumMnSteelMOO
            self.app.update_log("MOO algorithm module loaded successfully")

            # Create MOO instance
            moo = MediumMnSteelMOO()

            # Prepare data
            self.app.update_log("Preparing data for MOO algorithm...")
            moo_input_data = self.prepare_moo_input_data(sfe_data)

            if moo_input_data is None or len(moo_input_data) == 0:
                self.app.update_log("ERROR: No valid data to process")
                return None

            # Set the data in MOO instance
            moo.data = moo_input_data

            # Update constraints
            self.app.update_log("Applying design constraints...")
            moo.constraints.update(self.app.moo_constraints)

            self.app.update_log(f"Ms range: [{moo.constraints['Ms_min']}, {moo.constraints['Ms_max']}]°C")
            self.app.update_log(f"RA range: [{moo.constraints['Ra_min']}, {moo.constraints['Ra_max']}]")
            self.app.update_log(f"SFE range: [{moo.constraints['SFE_min']}, {moo.constraints['SFE_max']}] mJ/m²")
            self.app.update_log(f"Min ΔT: {moo.constraints['min_deltaT']}°C")
            self.app.update_log(f"Cementite max: {moo.constraints['Cementite_max']}")

            # Run MOO process
            self.app.update_log("Running multi-objective optimization...")
            moo_results = moo.process_moo()

            if moo_results is None or len(moo_results) == 0:
                self.app.update_log("WARNING: No solutions found with current constraints")
                return None

            # Report results
            rank_1_count = len(moo_results[moo_results['Rank'] == 1])
            gm_count = len(moo_results.dropna(subset=['GM_Rank']))
            total_count = len(moo_results)

            self.app.update_log(f"Optimization complete!")
            self.app.update_log(f"Total valid solutions: {total_count}")
            self.app.update_log(f"Pareto front (Rank 1): {rank_1_count}")
            self.app.update_log(f"GM ranked alloys: {gm_count}")

            if gm_count > 0:
                self.app.update_log(f"Top GM ranked alloy: Rank={int(moo_results.iloc[0]['GM_Rank'])}")

            return moo_results

        except Exception as e:
            self.app.update_log(f"ERROR in MOO optimization: {str(e)}")
            import traceback
            self.app.update_log(traceback.format_exc())
            return None

    def prepare_moo_input_data(self, sfe_data):
        """Prepare data from SFE calculations for MOO algorithm."""
        try:
            self.app.update_log("Transforming data structure for MOO algorithm...")

            # Create a copy
            moo_data = sfe_data.copy()

            # Map element content columns - Si is MANDATORY
            element_mapping = {
                'C_content': 'C',
                'Mn_content': 'Mn',
                'Al_content': 'Al',
                'Mo_content': 'Mo',
                'Nb_content': 'Nb',
                'V_content': 'V',
                'Si_content': 'Si'  # CRITICAL: Si must be included
            }

            for old_col, new_col in element_mapping.items():
                if old_col in moo_data.columns:
                    moo_data[new_col] = moo_data[old_col]
                elif f'{new_col}_content' in moo_data.columns:
                    moo_data[new_col] = moo_data[f'{new_col}_content']
                else:
                    self.app.update_log(f"WARNING: {new_col} not found in data, using 0.0")
                    moo_data[new_col] = 0.0

            # Check if Si was successfully mapped
            if 'Si' in moo_data.columns:
                si_values = moo_data['Si'].dropna()
                if len(si_values) > 0:
                    self.app.update_log(f"Si found! Range: {si_values.min():.3f} to {si_values.max():.3f}")
            else:
                self.app.update_log(f"ERROR: Si column not created!")

            # Temperature is already in correct format
            if 'Temperature' not in moo_data.columns:
                self.app.update_log("ERROR: Temperature column not found")
                return None

            # Map phase fractions
            if 'FCC_A1_Fraction' in moo_data.columns:
                moo_data['Gamma'] = moo_data['FCC_A1_Fraction']
            else:
                self.app.update_log("WARNING: FCC_A1_Fraction not found, using 0")
                moo_data['Gamma'] = 0.0

            if 'CEMENTITE_D011_Fraction' in moo_data.columns:
                moo_data['Cementite'] = moo_data['CEMENTITE_D011_Fraction']
            else:
                moo_data['Cementite'] = 0.0

            # Ms, Fm, and Ra should already be calculated
            required_cols = ['Ms', 'Fm', 'RA']
            for col in required_cols:
                if col not in moo_data.columns:
                    self.app.update_log(f"WARNING: {col} column not found")
                    if col == 'RA':
                        if 'Ra' in moo_data.columns:
                            moo_data['RA'] = moo_data['Ra']
                        else:
                            moo_data['RA'] = 0.0
                    else:
                        moo_data[col] = 0.0

            # Rename RA to Ra for MOO algorithm
            moo_data['Ra'] = moo_data['RA']

            # Ensure SFE column exists
            if 'SFE' not in moo_data.columns:
                self.app.update_log("ERROR: SFE column not found")
                return None

            # Required columns for MOO
            required_moo_cols = ['C', 'Mn', 'Si', 'Al', 'Mo', 'Nb', 'V',
                                 'Temperature', 'Ms', 'Ra', 'SFE', 'Fm', 'Cementite', 'Gamma']

            # Check which columns are available
            available_cols = [col for col in required_moo_cols if col in moo_data.columns]
            missing_cols = [col for col in required_moo_cols if col not in moo_data.columns]

            if missing_cols:
                self.app.update_log(f"Adding missing columns with default values: {missing_cols}")
                for col in missing_cols:
                    moo_data[col] = 0.0

            # Select only required columns
            moo_data = moo_data[required_moo_cols]

            # Remove rows with NaN in critical columns
            critical_cols = ['C', 'Mn', 'Si', 'Temperature', 'Ms', 'Ra', 'SFE']
            before_count = len(moo_data)
            moo_data = moo_data.dropna(subset=critical_cols)
            after_count = len(moo_data)

            if before_count > after_count:
                self.app.update_log(f"Removed {before_count - after_count} rows with missing critical data")

            self.app.update_log(f"Data prepared: {len(moo_data)} rows ready for MOO")

            # Show sample
            if len(moo_data) > 0:
                sample = moo_data.iloc[0]
                self.app.update_log(f"Sample composition: C={sample['C']:.3f}, Mn={sample['Mn']:.2f}, "
                                    f"Si={sample['Si']:.3f}, Al={sample['Al']:.2f}, "
                                    f"Ms={sample['Ms']:.1f}°C, Ra={sample['Ra']:.3f}, SFE={sample['SFE']:.1f}")

            return moo_data

        except Exception as e:
            self.app.update_log(f"ERROR preparing MOO input data: {str(e)}")
            import traceback
            self.app.update_log(traceback.format_exc())
            return None