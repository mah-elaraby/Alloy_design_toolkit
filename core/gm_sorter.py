"""
Geometric Mean sorting for optimal annealing time - UPDATED VERSION
"""

import pandas as pd
import numpy as np
from scipy.stats import gmean

class GMSorter:
    """Handles GM sorting for optimal annealing time determination."""

    def __init__(self, app):
        self.app = app

    def optimize_annealing(self, precip_data):
        """Optimize annealing time using geometric mean ranking and return complete ranked data."""
        try:
            self.app.update_log("Starting PRISMA GM sorting for optimal annealing time...")

            # Check for precipitation time-series data
            if 'precipitation_timeseries' not in self.app.workflow_runner.workflow_data:
                self.app.update_log("WARNING: No time-series precipitation data available")
                self.app.update_log("Skipping GM sorting step...")
                return precip_data

            # Get the time-series data
            timeseries_df = self.app.workflow_runner.workflow_data['precipitation_timeseries'].copy()
            self.app.update_log(f"Processing {len(timeseries_df)} time-series data points")

            # Check identifier columns
            has_gm_rank = 'GM_Rank' in timeseries_df.columns
            has_alloy_index = 'Alloy_Index' in timeseries_df.columns

            if not has_gm_rank and not has_alloy_index:
                self.app.update_log("ERROR: No alloy identifier found in time-series data")
                return precip_data

            # Use the best available identifier
            if has_gm_rank:
                alloy_identifier = 'GM_Rank'
                timeseries_df['Alloy_ID'] = 'Alloy_GM' + timeseries_df['GM_Rank'].astype(str)
            elif has_alloy_index:
                alloy_identifier = 'Alloy_Index'
                timeseries_df['Alloy_ID'] = 'Alloy_' + timeseries_df['Alloy_Index'].astype(str)

            # Create composition label
            element_order = ['C', 'Mn', 'Si', 'Al', 'Mo', 'Nb', 'V']

            def create_composition_label(row):
                parts = []
                for elem in element_order:
                    col = f'{elem}_wt_percent'
                    if col in row.index and pd.notna(row[col]):
                        value = row[col]
                        if value == 0:
                            formatted = "0"
                        elif value >= 1:
                            formatted = f"{value:.0f}" if value == int(value) else f"{value:.1f}"
                        else:
                            formatted = f"{value:.3f}".rstrip('0').rstrip('.')
                        parts.append(f"{formatted}{elem}")
                return "-".join(parts) if parts else "Unknown"

            timeseries_df['Composition'] = timeseries_df.apply(create_composition_label, axis=1)

            # Map column names to match GM sorting expectations
            column_mapping = {
                'Time_s': 'Time [s]',
                'Mean_radius_m': 'Mean radius (Nb-V)C',
                'Number_density_m3': 'Number density (Nb-V)C',
                'Precipitate_volume_fraction': 'Volume fraction (Nb-V)C',
                'Matrix_wtfrac_Mo': 'Matrix composition Mo',
                'Matrix_wtfrac_C': 'Matrix composition C'
            }

            for old_col, new_col in column_mapping.items():
                if old_col in timeseries_df.columns:
                    timeseries_df[new_col] = timeseries_df[old_col]
                elif new_col not in timeseries_df.columns:
                    self.app.update_log(f"WARNING: Column '{old_col}' not found, using 0.0 for '{new_col}'")
                    timeseries_df[new_col] = 0.0

            # Get selected objectives
            objectives_to_use = []
            for obj_name, obj_config in self.app.gm_objectives.items():
                if obj_config['selected']:
                    objectives_to_use.append({
                        'name': obj_name,
                        'goal': obj_config['goal']
                    })
                    self.app.update_log(f"[+] {obj_name} ({obj_config['goal']})")

            if not objectives_to_use:
                self.app.update_log("WARNING: No objectives selected for GM ranking!")
                return precip_data

            # Process each alloy separately
            unique_alloys = timeseries_df['Alloy_ID'].unique()
            self.app.update_log(f"Processing {len(unique_alloys)} unique alloys")

            processed_alloy_dfs = []

            for alloy_id in unique_alloys:
                alloy_df = timeseries_df[timeseries_df['Alloy_ID'] == alloy_id].copy()

                # Get display info
                comp_label = alloy_df['Composition'].iloc[0] if 'Composition' in alloy_df.columns else alloy_id
                gm_rank = alloy_df['GM_Rank'].iloc[0] if 'GM_Rank' in alloy_df.columns else 'N/A'

                # Normalize objectives for this alloy
                for obj_config in objectives_to_use:
                    obj_name = obj_config['name']
                    goal = obj_config['goal']

                    if obj_name not in alloy_df.columns:
                        self.app.update_log(f"WARNING: Objective '{obj_name}' not found for {alloy_id}")
                        continue

                    values = pd.to_numeric(alloy_df[obj_name], errors='coerce')
                    min_val = values.min()
                    max_val = values.max()

                    # Sanitized name for normalized column
                    norm_col_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in obj_name) + '_norm'

                    if pd.isna(min_val) or pd.isna(max_val):
                        alloy_df[norm_col_name] = np.nan
                        continue

                    # Handle small variations
                    relative_diff = abs(max_val - min_val) / max(abs(max_val), abs(min_val), 1e-15)

                    if relative_diff < 1e-6:
                        if goal == 'maximize':
                            alloy_df[norm_col_name] = (values - min_val) / (max_val - min_val + 1e-15)
                        elif goal == 'minimize':
                            alloy_df[norm_col_name] = (max_val - values) / (max_val - min_val + 1e-15)
                    elif abs(max_val - min_val) < 1e-15:
                        alloy_df[norm_col_name] = 1.0
                    else:
                        if goal == 'maximize':
                            alloy_df[norm_col_name] = (values - min_val) / (max_val - min_val)
                        elif goal == 'minimize':
                            alloy_df[norm_col_name] = (max_val - values) / (max_val - min_val)

                    alloy_df[norm_col_name] = alloy_df[norm_col_name].clip(0, 1)

                # Calculate Geometric Mean (Overall Score)
                norm_cols = [''.join(c if c.isalnum() or c == '_' else '_' for c in obj['name']) + '_norm'
                             for obj in objectives_to_use]
                norm_cols = [col for col in norm_cols if col in alloy_df.columns]

                gmean_values = []
                for _, row in alloy_df.iterrows():
                    if not norm_cols:
                        current_gmean = 0.0
                    else:
                        obj_values = [row[norm_col] for norm_col in norm_cols if pd.notna(row[norm_col])]
                        if not obj_values:
                            current_gmean = 0.0
                        else:
                            current_gmean = gmean(obj_values)
                    gmean_values.append(current_gmean)

                alloy_df['Overall_Score'] = gmean_values
                alloy_df['Overall_Score'] = alloy_df['Overall_Score'].fillna(0)

                # Sort by Overall Score
                alloy_df = alloy_df.sort_values(by='Overall_Score', ascending=False)

                # Add rank within this alloy
                alloy_df['Rank_within_Alloy'] = range(1, len(alloy_df) + 1)

                # Log optimal time for this alloy
                if len(alloy_df) > 0:
                    best_row = alloy_df.iloc[0]
                    optimal_time = best_row.get('Time [s]', 0)
                    optimal_score = best_row['Overall_Score']
                    self.app.update_log(
                        f"GM_Rank {gm_rank}: Optimal time = {optimal_time:.1f}s, Score = {optimal_score:.4f}")

                processed_alloy_dfs.append(alloy_df)

            # Combine all processed alloys
            if processed_alloy_dfs:
                gm_sorted_results = pd.concat(processed_alloy_dfs, ignore_index=True)

                # Sort by GM_Rank and Rank_within_Alloy
                if 'GM_Rank' in gm_sorted_results.columns:
                    gm_sorted_results = gm_sorted_results.sort_values(
                        by=['GM_Rank', 'Rank_within_Alloy'],
                        ascending=[True, True]
                    )
                else:
                    gm_sorted_results = gm_sorted_results.sort_values(
                        by=['Alloy_ID', 'Rank_within_Alloy'],
                        ascending=[True, True]
                    )

                self.app.update_log(f"GM sorting complete: {len(gm_sorted_results)} rows ranked")
                self.app.update_log(f"Processed {len(unique_alloys)} individual alloys")

                # Store the complete ranked precipitation data
                self.app.workflow_runner.workflow_data['precipitation_timeseries'] = gm_sorted_results.copy()
                self.app.workflow_runner.workflow_data['gm_sorted_results'] = gm_sorted_results.copy()

                # Extract optimal times
                optimal_times = gm_sorted_results[gm_sorted_results['Rank_within_Alloy'] == 1].copy()
                self.app.update_log(f"Found optimal annealing times for {len(optimal_times)} alloys")

                self.app.update_log("GM sorting results stored in memory")

                # Return the complete GM-sorted precipitation data
                return gm_sorted_results
            else:
                self.app.update_log("WARNING: No GM sorting results generated")
                return precip_data

        except Exception as e:
            self.app.update_log(f"ERROR in GM sorting: {str(e)}")
            import traceback
            self.app.update_log(traceback.format_exc())
            return precip_data