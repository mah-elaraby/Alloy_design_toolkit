"""
Excel file handling utilities for saving workflow results - UPDATED VERSION
"""

import pandas as pd
import numpy as np
import os

def save_workflow_results(workflow_data, output_file='Alloy_design_results.xlsx'):
    """Save all workflow results to Excel with cleaned and organized data."""

    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:

            # Sheet 1: Phase Calculations & RA
            if workflow_data.get('phase_results') is not None:
                phase_df = workflow_data['phase_results'].copy()

                # Remove unwanted columns
                cols_to_remove = [
                    'Mass_fraction_N_in_FCC_A1',
                    'Mass_fraction_Cr_in_FCC_A1',
                    'Mass_fraction_Ni_in_FCC_A1',
                    'Mass_fraction_Cu_in_FCC_A1',
                    'SFE'
                ]
                for col in cols_to_remove:
                    if col in phase_df.columns:
                        phase_df = phase_df.drop(columns=[col])

                phase_df.to_excel(writer, sheet_name='1&2_Phase_Calculation & RA', index=False)

            # Sheet 2: SFE Results
            if workflow_data.get('sfe_results') is not None:
                sfe_df = workflow_data['sfe_results'].copy()

                # Keep only specified columns
                cols_to_keep = [
                    'Al_content', 'C_content', 'Mn_content', 'Mo_content',
                    'Nb_content', 'Si_content', 'V_content', 'Temperature',
                    'Mass_fraction_C_in_FCC_A1', 'Mass_fraction_Mn_in_FCC_A1',
                    'Mass_fraction_Si_in_FCC_A1', 'Mass_fraction_Al_in_FCC_A1',
                    'Mass_fraction_Mo_in_FCC_A1', 'Mass_fraction_N_in_FCC_A1',
                    'Mass_fraction_Cr_in_FCC_A1', 'Mass_fraction_Ni_in_FCC_A1',
                    'Mass_fraction_Cu_in_FCC_A1', 'SFE'
                ]

                # Filter to only columns that exist
                existing_cols = [col for col in cols_to_keep if col in sfe_df.columns]
                sfe_df = sfe_df[existing_cols]

                # Rename SFE column
                if 'SFE' in sfe_df.columns:
                    sfe_df = sfe_df.rename(columns={'SFE': 'SFE mJ/m^2'})

                sfe_df.to_excel(writer, sheet_name='3_SFE_calculation', index=False)

            # Sheet 3: MOO Results
            if workflow_data.get('moo_results') is not None:
                moo_df = workflow_data['moo_results'].copy()

                # Remove unwanted columns
                cols_to_remove = [
                    'Time_s', 'Volume_fraction_percent', 'Mean_radius_m',
                    'Mean_radius_nm', 'Number_density_m3',
                    'Matrix_comp_C_wtfrac', 'Matrix_comp_C_wt%',
                    'Matrix_comp_Mn_wtfrac', 'Matrix_comp_Mn_wt%',
                    'Matrix_comp_Si_wtfrac', 'Matrix_comp_Si_wt%',
                    'Matrix_comp_Al_wtfrac', 'Matrix_comp_Al_wt%',
                    'Matrix_comp_Mo_wtfrac', 'Matrix_comp_Mo_wt%',
                    'Matrix_comp_Nb_wtfrac', 'Matrix_comp_Nb_wt%',
                    'Matrix_comp_V_wtfrac', 'Matrix_comp_V_wt%'
                ]
                for col in cols_to_remove:
                    if col in moo_df.columns:
                        moo_df = moo_df.drop(columns=[col])

                # Reorder columns to move GM_Score and GM_Rank after Norm_J4_deltaT
                if 'Norm_J4_deltaT' in moo_df.columns and 'GM_Score' in moo_df.columns:
                    cols = list(moo_df.columns)

                    # Remove GM_Score and GM_Rank from their current positions
                    cols_to_move = ['GM_Score', 'GM_Rank']
                    remaining_cols = [col for col in cols if col not in cols_to_move]

                    # Find position of Norm_J4_deltaT
                    if 'Norm_J4_deltaT' in remaining_cols:
                        insert_pos = remaining_cols.index('Norm_J4_deltaT') + 1

                        # Insert GM columns after Norm_J4_deltaT
                        for col in reversed(cols_to_move):
                            if col in cols:
                                remaining_cols.insert(insert_pos, col)

                        moo_df = moo_df[remaining_cols]

                moo_df.to_excel(writer, sheet_name='4_MOO_elements & Temp', index=False)

            # NEW SHEET: Precipitation Results GM Ranked (Complete time-series data)
            if workflow_data.get('gm_sorted_results') is not None:
                try:
                    gm_df = workflow_data['gm_sorted_results'].copy()

                    # Ensure we have all the precipitation data columns
                    required_precipitation_columns = [
                        'C_wt_percent', 'Mn_wt_percent', 'Si_wt_percent', 'Al_wt_percent',
                        'Mo_wt_percent', 'Nb_wt_percent', 'V_wt_percent',
                        'Temperature_C', 'Time_s', 'Precipitate_volume_fraction',
                        'Mean_radius_m', 'Number_density_m3', 'Nucleation_rate_m3s',
                        'GM_Rank', 'Composition', 'Overall_Score', 'Rank_within_Alloy'
                    ]

                    # Add wtfrac columns if they exist
                    wtfrac_columns = [
                        'Precipitate_wtfrac_C', 'Matrix_wtfrac_C',
                        'Precipitate_wtfrac_Mn', 'Matrix_wtfrac_Mn',
                        'Precipitate_wtfrac_Si', 'Matrix_wtfrac_Si',
                        'Precipitate_wtfrac_Al', 'Matrix_wtfrac_Al',
                        'Precipitate_wtfrac_Mo', 'Matrix_wtfrac_Mo',
                        'Precipitate_wtfrac_Nb', 'Matrix_wtfrac_Nb',
                        'Precipitate_wtfrac_V', 'Matrix_wtfrac_V'
                    ]

                    # Convert wtfrac to wt.% and rename columns
                    rename_dict = {}
                    for col in wtfrac_columns:
                        if col in gm_df.columns:
                            new_name = col.replace('wtfrac', 'wt.%')
                            rename_dict[col] = new_name
                            gm_df[col] = gm_df[col] * 100

                    gm_df = gm_df.rename(columns=rename_dict)

                    # Add Temperature in Kelvin
                    if 'Temperature_C' in gm_df.columns:
                        gm_df['Temperature_K'] = gm_df['Temperature_C'] + 273.15

                    # Select and order columns for better readability
                    display_columns = []

                    # Identifiers first
                    identifier_cols = ['GM_Rank', 'Alloy_Index', 'Composition', 'Rank_within_Alloy', 'Overall_Score']
                    for col in identifier_cols:
                        if col in gm_df.columns:
                            display_columns.append(col)

                    # Composition next
                    composition_cols = ['C_wt_percent', 'Mn_wt_percent', 'Si_wt_percent', 'Al_wt_percent',
                                      'Mo_wt_percent', 'Nb_wt_percent', 'V_wt_percent']
                    for col in composition_cols:
                        if col in gm_df.columns:
                            display_columns.append(col)

                    # Temperature and time
                    temp_time_cols = ['Temperature_C', 'Temperature_K', 'Time_s']
                    for col in temp_time_cols:
                        if col in gm_df.columns:
                            display_columns.append(col)

                    # Precipitation results
                    precip_cols = ['Precipitate_volume_fraction', 'Mean_radius_m', 'Mean_radius_nm',
                                 'Number_density_m3', 'Nucleation_rate_m3s']
                    for col in precip_cols:
                        if col in gm_df.columns:
                            display_columns.append(col)

                    # Matrix and precipitate compositions
                    comp_wt_cols = [col for col in gm_df.columns if 'wt.%' in col]
                    display_columns.extend(comp_wt_cols)

                    # Normalized objective columns (for transparency)
                    norm_cols = [col for col in gm_df.columns if '_norm' in col]
                    display_columns.extend(norm_cols)

                    # Select only existing columns
                    existing_columns = [col for col in display_columns if col in gm_df.columns]
                    gm_df = gm_df[existing_columns]

                    # Sort by GM_Rank and Rank_within_Alloy for better organization
                    if 'GM_Rank' in gm_df.columns and 'Rank_within_Alloy' in gm_df.columns:
                        gm_df = gm_df.sort_values(['GM_Rank', 'Rank_within_Alloy'], ascending=[True, True])

                    gm_df.to_excel(writer, sheet_name='5_precipitate_results_GM_ranked', index=False)

                except Exception as e:
                    print(f"Warning: Could not process GM sorted precipitation results: {e}")
                    import traceback
                    print(traceback.format_exc())

            # Sheet 6: Optimal Annealing Times (Rank 1 for each alloy)
            if workflow_data.get('gm_sorted_results') is not None:
                try:
                    gm_results = workflow_data['gm_sorted_results'].copy()
                    optimal_df = gm_results[gm_results['Rank_within_Alloy'] == 1].copy()

                    if len(optimal_df) > 0:
                        # Convert Temperature from Celsius to Kelvin if needed
                        if 'Temperature_C' in optimal_df.columns:
                            optimal_df['Temperature'] = optimal_df['Temperature_C'] + 273.15
                            optimal_df = optimal_df.drop(columns=['Temperature_C'])

                        # Convert wtfrac to wt.% and multiply by 100
                        wtfrac_columns = [
                            'Precipitate_wtfrac_C', 'Matrix_wtfrac_C',
                            'Precipitate_wtfrac_Mn', 'Matrix_wtfrac_Mn',
                            'Precipitate_wtfrac_Si', 'Matrix_wtfrac_Si',
                            'Precipitate_wtfrac_Al', 'Matrix_wtfrac_Al',
                            'Precipitate_wtfrac_Mo', 'Matrix_wtfrac_Mo',
                            'Precipitate_wtfrac_Nb', 'Matrix_wtfrac_Nb',
                            'Precipitate_wtfrac_V', 'Matrix_wtfrac_V'
                        ]

                        rename_dict = {}
                        for col in wtfrac_columns:
                            if col in optimal_df.columns:
                                new_name = col.replace('wtfrac', 'wt.%')
                                rename_dict[col] = new_name
                                optimal_df[col] = optimal_df[col] * 100

                        optimal_df = optimal_df.rename(columns=rename_dict)

                        # Remove unwanted columns
                        cols_to_remove = [
                            'Alloy_Index', 'Alloy_ID',
                            'Time [s]', 'Mean radius (Nb-V)C', 'Number density (Nb-V)C',
                            'Volume fraction (Nb-V)C', 'Matrix composition Mo',
                            'Matrix composition C'
                        ]
                        for col in cols_to_remove:
                            if col in optimal_df.columns:
                                optimal_df = optimal_df.drop(columns=[col])

                        optimal_df.to_excel(writer, sheet_name='6_Optimal_annealing_time', index=False)

                except Exception as e:
                    print(f"Warning: Could not process optimal annealing times from workflow data: {e}")

        return True

    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def export_to_csv(workflow_data, output_dir='output'):
    """Export workflow results to individual CSV files."""
    try:
        os.makedirs(output_dir, exist_ok=True)

        if workflow_data.get('phase_results') is not None:
            workflow_data['phase_results'].to_csv(f'{output_dir}/phase_results.csv', index=False)

        if workflow_data.get('sfe_results') is not None:
            workflow_data['sfe_results'].to_csv(f'{output_dir}/sfe_results.csv', index=False)

        if workflow_data.get('moo_results') is not None:
            workflow_data['moo_results'].to_csv(f'{output_dir}/moo_results.csv', index=False)

        if workflow_data.get('precipitation_results') is not None:
            workflow_data['precipitation_results'].to_csv(f'{output_dir}/precipitation_results.csv', index=False)

        # Export GM sorted precipitation results
        if workflow_data.get('gm_sorted_results') is not None:
            try:
                workflow_data['gm_sorted_results'].to_csv(f'{output_dir}/precipitation_gm_ranked.csv', index=False)
            except Exception as e:
                print(f"Warning: Could not export GM sorted precipitation results to CSV: {e}")

        return True

    except Exception as e:
        print(f"Error exporting to CSV: {e}")
        return False