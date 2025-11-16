"""
Phase calculation module - COMPLETE VERSION
"""

import os
import itertools
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed


class PhaseCalculator:
    " Handles phase fraction and composition calculations using TC-Python. "

    def __init__(self, app):
        self.app = app

    def calculate(self):
        "Execute phase calculations."
        try:
            self.app.update_log("Setting up phase calculations...")

            # Build composition list
            comp_list = self.build_composition_list()
            if not comp_list:
                self.app.update_log("ERROR: No compositions to evaluate")
                return None

            # Run calculations in parallel
            results = self.run_parallel_calculations(comp_list)
            return results

        except Exception as e:
            self.app.update_log(f"ERROR in phase calculations: {str(e)}")
            return None

    def build_composition_list(self):
        """Build list of composition dictionaries."""
        comp_list = []
        names = sorted([e for e in self.app.selected_elements if e != 'Fe'])
        ranges = []

        for e in names:
            r = self.app.element_ranges.get(e, {'start': 0.1, 'end': 0.1, 'step': 0.1})
            if r['start'] == r['end']:
                ranges.append([float(r['start'])])
            else:
                ranges.append(list(np.arange(
                    float(r['start']),
                    float(r['end']) + float(r['step']) / 2.0,
                    float(r['step'])
                )))

        for vals in itertools.product(*ranges):
            comp_list.append(dict(zip(names, vals)))

        return comp_list

    def run_parallel_calculations(self, comp_list):
        """Run calculations in parallel processes."""
        static_args = (
            self.app.phase_database,
            self.app.cache_folder,
            self.app.selected_elements,
            self.app.selected_phases,
            self.app.temp_start,
            self.app.temp_end,
            self.app.temp_step,
            self.app.axis_max_step,
            self.app.calc_fraction,
            self.app.calc_composition
        )

        out_rows = []
        total = len(comp_list)
        max_workers = max(1, int(self.app.workers))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for comp in comp_list:
                if self.app.workflow_runner.stop_requested:
                    break
                futures.append(executor.submit(compute_one_composition, static_args + (comp,)))

            for i, future in enumerate(as_completed(futures)):
                if self.app.workflow_runner.stop_requested:
                    break
                try:
                    rows = future.result()
                    out_rows.extend(rows)
                    if (i + 1) % 5 == 0 or (i + 1) == total:
                        self.app.update_log(f"Progress: {i + 1}/{total} compositions")
                except Exception as e:
                    self.app.update_log(f"Worker error: {e}")

        if out_rows:
            df = pd.DataFrame(out_rows)
            self.app.update_log(f"Phase data processed: {len(df):,} rows")
            return df
        return None


# Worker function (must be at module level for multiprocessing)
def compute_one_composition(args):
    """
    Runs phase calculation for one composition in a separate process.
    This is the ACTUAL calculation code from "1 phase fraction and composition.py"
    """
    (database, cache, selected_elements, selected_phases,
     T0, T1, dT, axis_max_step,
     do_fraction, do_composition,
     composition_dict) = args

    from tc_python import TCPython, ThermodynamicQuantity, CalculationAxis
    from tc_python.step_or_map_diagrams import AxisType
    import numpy as _np

    T_grid = _np.arange(float(T0), float(T1) + 1e-9, float(dT))
    rows = []

    with TCPython() as sess:
        system = (sess
                  .set_cache_folder(cache)
                  .select_database_and_elements(database, selected_elements)
                  .get_system())

        # Property Diagram with axis densification
        calc = system.with_property_diagram_calculation()
        for e, wt in composition_dict.items():
            calc = calc.set_condition(f"W({e})", float(wt) / 100.0)

        calc = (calc
        .set_condition("N", 1)
        .set_condition("P", 1e5)
        .with_axis(
            CalculationAxis(ThermodynamicQuantity.temperature())
            .set_min(float(T0))
            .set_max(float(T1))
            .with_axis_type(
                AxisType.linear().set_max_step_size(axis_max_step or float(dT))
            )
        ))

        res = calc.calculate()

        # Phase fractions (resampled)
        frac_grid = {}
        if do_fraction or do_composition:
            for ph in selected_phases:
                Tx, Fy = res.get_values_of(
                    ThermodynamicQuantity.temperature(),
                    ThermodynamicQuantity.volume_fraction_of_a_phase(ph)
                )
                Tx = _np.asarray(Tx, float)
                Fy = _np.asarray(Fy, float)
                frac_grid[ph] = _np.interp(T_grid, Tx, Fy, left=0.0, right=0.0)

        # Phase compositions (resampled, masked where phase is absent)
        comp_grid = {}
        if do_composition:
            for ph in selected_phases:
                ph_dict = {}
                mask_fraction = frac_grid.get(ph, _np.zeros_like(T_grid))
                for e in selected_elements:
                    Tx, Wy = res.get_values_of(
                        ThermodynamicQuantity.temperature(),
                        ThermodynamicQuantity.composition_of_phase_as_weight_fraction(ph, e)
                    )
                    Tx = _np.asarray(Tx, float)
                    Wy = _np.asarray(Wy, float)
                    arr = _np.interp(T_grid, Tx, Wy, left=_np.nan, right=_np.nan)
                    arr = _np.where(mask_fraction > 0.0, arr, _np.nan)
                    ph_dict[e] = arr
                comp_grid[ph] = ph_dict

        # Emit one row per temperature
        for k, T in enumerate(T_grid):
            row = {f"{e}_content": composition_dict[e] for e in composition_dict}
            row["Temperature"] = float(T)
            if do_fraction:
                for ph in selected_phases:
                    row[f"{ph}_Fraction"] = float(frac_grid[ph][k])
            if do_composition:
                for ph in selected_phases:
                    for e in selected_elements:
                        row[f"Mass_fraction_{e}_in_{ph}"] = comp_grid[ph][e][k]
            rows.append(row)

    return rows
