"""
Precipitation kinetics calculations using TC-PRISMA
"""

import pandas as pd
import numpy as np
import os


class PrecipitationCalculator:
    """Handles precipitation kinetics calculations for top alloys."""

    def __init__(self, app):
        self.app = app

    def calculate(self, moo_data):
        """Calculate precipitation kinetics for top alloys."""
        try:
            self.app.update_log("Checking TC-Python availability...")

            # Check TC-Python
            try:
                from tc_python import TCPython
                from tc_python.precipitation import (
                    MatrixPhase, PrecipitatePhase, CompositionUnit, GrowthRateModel
                )
                self.app.update_log("TC-Python available for precipitation calculations")
            except ImportError:
                self.app.update_log("ERROR: TC-Python not available. Skipping precipitation step...")
                return moo_data

            # Select top alloys
            top_alloys = moo_data.dropna(subset=['GM_Rank']).copy()
            if len(top_alloys) == 0:
                self.app.update_log("WARNING: No GM-ranked alloys found. Using top 20 by rank.")
                top_alloys = moo_data.head(20).copy()

            n_alloys = len(top_alloys)
            self.app.update_log(f"Processing {n_alloys} top alloys for precipitation kinetics")

            # Log parameters
            self.app.update_log(f"Simulation time: {self.app.precipitation_params['sim_time']} seconds")
            self.app.update_log(f"Matrix phase: {self.app.precipitation_params['matrix_phase']}")
            self.app.update_log(f"Precipitate phase: {self.app.precipitation_params['precipitate_phase']}")

            # Composition columns
            composition_cols = ['C', 'Mn', 'Si', 'Al', 'Mo', 'Nb', 'V']

            # Verify all composition columns are present
            missing_cols = [col for col in composition_cols if col not in top_alloys.columns]
            if missing_cols:
                self.app.update_log(f"WARNING: Missing composition columns: {missing_cols}")
                for col in missing_cols:
                    top_alloys[col] = 0.0

            # Verify Si has actual values
            if 'Si' in top_alloys.columns:
                si_values = top_alloys['Si'].dropna()
                if len(si_values) > 0:
                    self.app.update_log(f"Si values in top alloys: {si_values.min():.3f} to {si_values.max():.3f}")

            # Initialize results storage
            precipitation_results = []
            precipitation_full_rows = []

            # Create cache folder
            cache_folder = getattr(self.app, 'cache_folder', './cache/')
            os.makedirs(cache_folder, exist_ok=True)

            # Selected elements for TC-Python
            selected_elements = ['Fe', 'C', 'Mn', 'Si', 'Al', 'Mo', 'Nb', 'V']
            self.app.update_log(f"Selected elements for TC-Python: {selected_elements}")

            # Start TC-Python session
            self.app.update_log("Initializing TC-Python session...")

            with TCPython() as session:
                # Setup system
                system = (
                    session
                    .set_cache_folder(cache_folder)
                    .select_thermodynamic_and_kinetic_databases_with_elements(
                        self.app.precipitation_params['tdb'],
                        self.app.precipitation_params['kdb'],
                        selected_elements
                    )
                    .get_system()
                )

                self.app.update_log("TC-Python system initialized successfully")

                # Process each alloy
                for idx, (row_idx, alloy) in enumerate(top_alloys.iterrows()):
                    if self.app.workflow_runner.stop_requested:
                        self.app.update_log("Stop requested, aborting precipitation calculations")
                        break

                    try:
                        # Extract composition
                        composition = {}
                        for elem in composition_cols:
                            composition[elem] = float(alloy.get(elem, 0.0))

                        # Get optimal temperature
                        temp_kelvin = float(alloy.get('T_opt', 873.15))
                        temp_celsius = temp_kelvin - 273.15

                        # Get GM rank
                        gm_rank = int(alloy.get('GM_Rank', idx + 1))

                        # Log current calculation
                        comp_str = ", ".join([f"{elem}={composition[elem]:.3f}"
                                              for elem in ['C', 'Mn', 'Si', 'Al']])
                        self.app.update_log(
                            f"[{idx + 1}/{n_alloys}] GM Rank {gm_rank}: {comp_str}, T={temp_celsius:.0f}°C")

                        # Build and run precipitation calculation
                        precip_result = self.calculate_single_precipitation(
                            system,
                            composition,
                            temp_kelvin,
                            self.app.precipitation_params['sim_time'],
                            self.app.precipitation_params['matrix_phase'],
                            self.app.precipitation_params['precipitate_phase']
                        )

                        if precip_result:
                            precip_result['GM_Rank'] = gm_rank
                            precip_result['Alloy_Index'] = idx + 1

                            # Add composition to results
                            for elem in composition_cols:
                                precip_result[f'{elem}_input'] = composition[elem]

                            precipitation_results.append(precip_result)

                        # Export full time-series
                        try:
                            _summary2, _rows = self.calculate_precipitation_full(
                                system,
                                composition,
                                temp_kelvin,
                                self.app.precipitation_params['sim_time'],
                                self.app.precipitation_params['matrix_phase'],
                                self.app.precipitation_params['precipitate_phase']
                            )
                            if _rows:
                                for _r in _rows:
                                    _r['GM_Rank'] = gm_rank
                                    _r['Alloy_Index'] = idx + 1
                                precipitation_full_rows.extend(_rows)
                        except Exception as _e:
                            self.app.update_log(f"WARNING: could not extract full time series: {_e}")

                        # Update progress
                        progress_pct = 80 + (20 * (idx + 1) / n_alloys)
                        self.app.update_progress(progress_pct)

                    except Exception as e:
                        self.app.update_log(f"Error calculating alloy {idx + 1}: {str(e)}")
                        continue

            # Process results
            if precipitation_results:
                self.app.update_log(
                    f"Precipitation calculations complete: {len(precipitation_results)} alloys processed")

                # Add precipitation results to dataframe
                precip_df = pd.DataFrame(precipitation_results)

                # Merge back with top_alloys
                top_alloys_with_precip = top_alloys.merge(
                    precip_df,
                    on='GM_Rank',
                    how='left',
                    suffixes=('', '_precip')
                )

                # Update the full moo_data
                precip_columns = [col for col in precip_df.columns if col not in ['GM_Rank', 'Alloy_Index']]
                for col in precip_columns:
                    if col not in moo_data.columns:
                        moo_data[col] = np.nan

                # Update rows for top alloys
                for col in precip_columns:
                    moo_data.loc[top_alloys.index, col] = top_alloys_with_precip[col].values

                # Store full time-series
                if precipitation_full_rows:
                    try:
                        full_df = pd.DataFrame(precipitation_full_rows)
                        self.app.workflow_runner.workflow_data['precipitation_timeseries'] = full_df.copy()
                        self.app.update_log("Precipitation time-series data stored in memory")
                    except Exception as _e:
                        self.app.update_log(f"WARNING: could not store precipitation time-series: {_e}")

                # Log summary statistics
                if 'Mean_radius_nm' in precip_df.columns:
                    mean_radii = precip_df['Mean_radius_nm'].dropna()
                    if len(mean_radii) > 0:
                        self.app.update_log(f"Mean radius range: {mean_radii.min():.2f} to {mean_radii.max():.2f} nm")

                if 'Volume_fraction_percent' in precip_df.columns:
                    vol_fracs = precip_df['Volume_fraction_percent'].dropna()
                    if len(vol_fracs) > 0:
                        self.app.update_log(f"Volume fraction range: {vol_fracs.min():.3f} to {vol_fracs.max():.3f} %")

            else:
                self.app.update_log("WARNING: No precipitation results generated")

            return moo_data

        except Exception as e:
            self.app.update_log(f"ERROR in precipitation kinetics: {str(e)}")
            import traceback
            self.app.update_log(traceback.format_exc())
            return moo_data

    def calculate_single_precipitation(self, system, composition, temperature_kelvin,
                                       sim_time, matrix_phase, precipitate_phase):
        """Calculate precipitation kinetics for a single alloy."""
        try:
            from tc_python.precipitation import (
                MatrixPhase, PrecipitatePhase, CompositionUnit, GrowthRateModel
            )

            # Create precipitate phase object
            precipitate_phase_obj = PrecipitatePhase(precipitate_phase)

            # Apply growth model
            growth_model_name = self.app.precipitation_params['growth_model']
            growth_model_map = {
                "Simplified": GrowthRateModel.SIMPLIFIED,
                "General": GrowthRateModel.GENERAL,
                "Advanced": GrowthRateModel.ADVANCED,
                "Para_eq": GrowthRateModel.PARA_EQ,
                "NPLE": GrowthRateModel.NPLE,
            }

            if growth_model_name == "PE_AUTOMATIC":
                precipitate_phase_obj = (
                    precipitate_phase_obj
                    .with_growth_rate_model(GrowthRateModel.PE_AUTOMATIC)
                    .enable_driving_force_approximation()
                )
            elif growth_model_name in growth_model_map:
                precipitate_phase_obj = precipitate_phase_obj.with_growth_rate_model(
                    growth_model_map[growth_model_name]
                )

            # Apply nucleation site
            nucleation_site_name = self.app.precipitation_params['nucleation_site']
            nucleation_methods = {
                "Bulk": precipitate_phase_obj.set_nucleation_in_bulk,
                "Grain boundaries": precipitate_phase_obj.set_nucleation_at_grain_boundaries,
                "Grain edges": precipitate_phase_obj.set_nucleation_at_grain_edges,
                "Grain corners": precipitate_phase_obj.set_nucleation_at_grain_corners,
                "Dislocations": precipitate_phase_obj.set_nucleation_at_dislocations
            }

            if nucleation_site_name in nucleation_methods:
                precipitate_phase_obj = nucleation_methods[nucleation_site_name]()

            # Build precipitation calculation
            calc = (
                system
                .with_isothermal_precipitation_calculation()
                .set_composition_unit(CompositionUnit.MASS_PERCENT)
            )

            # Set composition for each element
            for elem, value in composition.items():
                calc = calc.set_composition(elem, value)

            # Complete calculation setup and run
            precip_calc = (
                calc
                .set_temperature(temperature_kelvin)
                .set_simulation_time(sim_time)
                .with_matrix_phase(
                    MatrixPhase(matrix_phase).add_precipitate_phase(precipitate_phase_obj)
                )
                .calculate()
            )

            # Extract results at final time point
            times, vol_fracs = precip_calc.get_volume_fraction_of(precipitate_phase)
            _, mean_radii = precip_calc.get_mean_radius_of(precipitate_phase)
            _, number_densities = precip_calc.get_number_density_of(precipitate_phase)

            # Get final values
            final_vol_frac = vol_fracs[-1] if len(vol_fracs) > 0 else 0.0
            final_mean_radius = mean_radii[-1] if len(mean_radii) > 0 else 0.0
            final_number_density = number_densities[-1] if len(number_densities) > 0 else 0.0

            # Get matrix composition at final time
            matrix_compositions = {}
            for elem in composition.keys():
                try:
                    _, comp_values = precip_calc.get_matrix_composition_in_weight_fraction_of(elem)
                    matrix_compositions[elem] = comp_values[-1] if len(comp_values) > 0 else None
                except:
                    matrix_compositions[elem] = None

            # Build result dictionary
            result = {
                'Time_s': times[-1] if len(times) > 0 else sim_time,
                'Volume_fraction_percent': final_vol_frac * 100,
                'Mean_radius_m': final_mean_radius,
                'Mean_radius_nm': final_mean_radius * 1e9,
                'Number_density_m3': final_number_density,
            }

            # Add matrix compositions
            for elem, value in matrix_compositions.items():
                if value is not None:
                    result[f'Matrix_comp_{elem}_wtfrac'] = value
                    result[f'Matrix_comp_{elem}_wt%'] = value * 100

            return result

        except Exception as e:
            self.app.update_log(f"Error in precipitation calculation: {str(e)}")
            return None

    def calculate_precipitation_full(self, system, composition, temperature_kelvin,
                                     sim_time, matrix_phase, precipitate_phase):
        """Run PRISMA and return full time-series data."""
        try:
            from tc_python.precipitation import (
                MatrixPhase, PrecipitatePhase, CompositionUnit, GrowthRateModel
            )

            # Create precipitate phase object
            precipitate_phase_obj = PrecipitatePhase(precipitate_phase)
            growth_model_name = self.app.precipitation_params['growth_model']
            growth_model_map = {
                "Simplified": GrowthRateModel.SIMPLIFIED,
                "General": GrowthRateModel.GENERAL,
                "Advanced": GrowthRateModel.ADVANCED,
                "Para_eq": GrowthRateModel.PARA_EQ,
                "NPLE": GrowthRateModel.NPLE,
            }

            if growth_model_name == "PE_AUTOMATIC":
                precipitate_phase_obj = (
                    precipitate_phase_obj
                    .with_growth_rate_model(GrowthRateModel.PE_AUTOMATIC)
                    .enable_driving_force_approximation()
                )
            elif growth_model_name in growth_model_map:
                precipitate_phase_obj = precipitate_phase_obj.with_growth_rate_model(
                    growth_model_map[growth_model_name]
                )

            # Apply nucleation site
            nucleation_site_name = self.app.precipitation_params['nucleation_site']
            nucleation_methods = {
                "Bulk": precipitate_phase_obj.set_nucleation_in_bulk,
                "Grain boundaries": precipitate_phase_obj.set_nucleation_at_grain_boundaries,
                "Grain edges": precipitate_phase_obj.set_nucleation_at_grain_edges,
                "Grain corners": precipitate_phase_obj.set_nucleation_at_grain_corners,
                "Dislocations": precipitate_phase_obj.set_nucleation_at_dislocations
            }

            if nucleation_site_name in nucleation_methods:
                precipitate_phase_obj = nucleation_methods[nucleation_site_name]()

            # Build calculation
            calc = (
                system
                .with_isothermal_precipitation_calculation()
                .set_composition_unit(CompositionUnit.MASS_PERCENT)
            )

            for elem, value in composition.items():
                calc = calc.set_composition(elem, float(value))

            precip_calc = (
                calc
                .set_temperature(float(temperature_kelvin))
                .set_simulation_time(float(sim_time))
                .with_matrix_phase(
                    MatrixPhase(matrix_phase).add_precipitate_phase(precipitate_phase_obj)
                )
                .calculate()
            )

            # Extract time series data
            times, vol_fracs = precip_calc.get_volume_fraction_of(precipitate_phase)
            _, mean_radii = precip_calc.get_mean_radius_of(precipitate_phase)
            _, number_densities = precip_calc.get_number_density_of(precipitate_phase)

            try:
                _, nucleation_rates = precip_calc.get_nucleation_rate_of(precipitate_phase)
            except Exception:
                nucleation_rates = [None] * len(times)

            # Per-element compositions
            precip_comp = {}
            matrix_comp = {}
            for elem in composition.keys():
                try:
                    _, pvals = precip_calc.get_precipitate_composition_in_weight_fraction_of(
                        precipitate_phase, elem
                    )
                except Exception:
                    pvals = [None] * len(times)
                precip_comp[elem] = pvals

                try:
                    _, mvals = precip_calc.get_matrix_composition_in_weight_fraction_of(elem)
                except Exception:
                    mvals = [None] * len(times)
                matrix_comp[elem] = mvals

            temp_celsius = float(temperature_kelvin) - 273.15
            full_rows = []

            for i, t in enumerate(times):
                row = {}
                for elem, wt in composition.items():
                    row[f"{elem}_wt_percent"] = float(wt)
                row["Temperature_C"] = temp_celsius
                row["Time_s"] = float(t)
                row["Precipitate_volume_fraction"] = vol_fracs[i]
                row["Mean_radius_m"] = mean_radii[i]
                row["Number_density_m3"] = number_densities[i]

                if nucleation_rates is not None:
                    row["Nucleation_rate_m3s"] = nucleation_rates[i]

                for elem in composition.keys():
                    row[f"Precipitate_wtfrac_{elem}"] = precip_comp[elem][i]
                    row[f"Matrix_wtfrac_{elem}"] = matrix_comp[elem][i]

                full_rows.append(row)

            # Summary (final point)
            final = full_rows[-1]
            final_vol_frac = final.get("Precipitate_volume_fraction")
            final_mean_radius_m = final.get("Mean_radius_m")

            summary = {
                "Time_s": final.get("Time_s"),
                "Volume_fraction_percent": (final_vol_frac * 100.0) if final_vol_frac is not None else None,
                "Mean_radius_m": final_mean_radius_m,
                "Mean_radius_nm": (final_mean_radius_m * 1e9) if final_mean_radius_m is not None else None,
                "Number_density_m3": final.get("Number_density_m3"),
            }

            for elem in composition.keys():
                last_matrix = matrix_comp.get(elem, [None])[-1]
                if last_matrix is not None:
                    summary[f"Matrix_comp_{elem}_wtfrac"] = last_matrix
                    summary[f"Matrix_comp_{elem}_wt%"] = last_matrix * 100.0

            return summary, full_rows

        except Exception as e:
            self.app.update_log(f"Error in full precipitation extraction: {str(e)}")
            return None, []