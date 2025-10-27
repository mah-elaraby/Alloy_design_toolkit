# -*- coding: utf-8 -*-
"""
Thermodynamic Calculator (GUI) - Parallel Property Diagram with Resampling

Features:
- Property Diagram with linear axis densification (set_max_step_size)
- Phase fractions and compositions resampled to exact temperature grid
- Parallel processing across compositions via ProcessPoolExecutor
- Chunked saving to avoid memory issues with large datasets
- Pause/resume functionality during calculations
- Real-time progress tracking with time estimation
- Single output CSV: Results_PhaseData.csv
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import timedelta
import multiprocessing

# TC-Python imports
try:
    from tc_python import TCPython, ThermodynamicQuantity, CalculationAxis
    from tc_python.step_or_map_diagrams import AxisType

    TC_PYTHON_AVAILABLE = True
except Exception:
    TC_PYTHON_AVAILABLE = False
    print("Warning: TC-Python not found. Calculations will not work without it.", flush=True)

# Global pause event for multiprocessing
pause_event = None


def _compute_one_composition(args):
    """
    Worker function that runs in a separate process.
    Builds property diagram with axis densification, resamples fractions
    and compositions onto exact grid, returns list of row dictionaries.
    """
    global pause_event

    # Handle pause requests
    if pause_event is not None:
        while pause_event.is_set():
            time.sleep(0.5)

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

        # Build property diagram calculation
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

        # Extract and resample phase fractions
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

        # Extract and resample phase compositions
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
                    # Mask composition where phase is absent
                    arr = _np.where(mask_fraction > 0.0, arr, _np.nan)
                    ph_dict[e] = arr
                comp_grid[ph] = ph_dict

        # Build output rows
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


class ThermoCalcGUI:
    """Main GUI application for thermodynamic calculations"""

    DEFAULT_ELEMENTS = ["Fe", "C", "Mn", "Si", "Al", "Mo", "Nb", "V"]
    AVAILABLE_ELEMENTS = {
        'B': (1, 12), 'C': (1, 13), 'N': (1, 14), 'O': (1, 15),
        'Mg': (2, 1), 'Al': (2, 12), 'Si': (2, 13), 'P': (2, 14), 'S': (2, 15),
        'Ca': (3, 1), 'Sc': (3, 2), 'Ti': (3, 3), 'V': (3, 4), 'Cr': (3, 5),
        'Mn': (3, 6), 'Fe': (3, 7), 'Co': (3, 8), 'Ni': (3, 9), 'Cu': (3, 10),
        'Zn': (3, 11), 'Nb': (4, 4), 'Mo': (4, 5), 'Cs': (5, 0), 'Ta': (5, 4), 'W': (5, 5)
    }
    AVAILABLE_PHASES = [
        "FCC_A1", "BCC_A2", "CEMENTITE_D011", "FCC_A1#2", "HCP_A3", "HCP_A3#2",
        "KAPPA_E21", "LIQUID", "M23C6_D84", "M3C2_D510", "M5C2", "M6C_E93", "M7C3_D101", "MC_ETA"
    ]

    AXIS_MAX_STEP_SIZE = 1.0
    OUTPUT_FILE = "Results_PhaseData.csv"
    CHUNK_SIZE = 5000
    TEMP_DIR = "./temp_chunks/"

    def __init__(self, root):
        self.root = root
        self.root.title("Thermodynamic Calculator")
        self.root.resizable(True, True)

        self.stop_requested = False
        self.pause_requested = False
        self.calc_thread = None

        # Initialize global pause event for multiprocessing
        global pause_event
        if pause_event is None:
            pause_event = multiprocessing.Event()

        self.selected_elements = self.DEFAULT_ELEMENTS.copy()
        self.selected_phases = []
        self.phase_vars = {ph: tk.BooleanVar(value=False) for ph in self.AVAILABLE_PHASES}

        # Results options
        self.calc_phase_fraction = tk.BooleanVar(value=True)
        self.calc_phase_composition = tk.BooleanVar(value=True)

        self.element_inputs = {}

        self._build_ui()
        self.root.after(100, self._size_to_content)

    def _build_ui(self):
        """Build the main user interface"""
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.scroll = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.container = ttk.Frame(self.canvas, padding="10")
        self.container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.container, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

        ttk.Label(self.container, text="Thermodynamic Calculator", font=('Arial', 14, 'bold')).pack(pady=10)

        self._make_elements_section(self.container)
        self._make_temperature_section(self.container)
        self._make_settings_section(self.container)
        self._make_buttons_section(self.container)
        self._make_log_section(self.container)

    def _size_to_content(self, h=24, v=80):
        """Adjust window size to fit content"""
        self.root.update_idletasks()
        req_w = self.container.winfo_reqwidth()
        req_h = self.container.winfo_reqheight()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        tw = min(req_w, max(800, sw - h))
        th = min(req_h, max(300, sh - v))
        needs_scroll = req_h > th
        sbw = self.scroll.winfo_reqwidth() or 18
        if needs_scroll:
            tw = min(req_w + sbw, max(800, sw - h))
            cw = max(100, tw - sbw)
        else:
            cw = tw
        self.canvas.config(width=cw, height=th)
        self.root.geometry(f"{int(tw)}x{int(th)}")

    def _make_elements_section(self, parent):
        """Create the element composition input section"""
        self.el_frame = ttk.LabelFrame(parent, text="Element Composition (wt%)", padding="10")
        self.el_frame.pack(fill=tk.X, pady=5)
        self._rebuild_element_inputs()

    def _rebuild_element_inputs(self):
        """Rebuild element input fields based on selected elements"""
        for w in self.el_frame.winfo_children():
            w.destroy()
        headers = ["Element", "Start", "End", "Step"]
        for i, h in enumerate(headers):
            ttk.Label(self.el_frame, text=h, font=('Arial', 9, 'bold')).grid(row=0, column=i, padx=5, pady=5)
        defaults = {'C': (0.05, 0.5, 0.05), 'Mn': (3, 12, 1.0), 'Si': (0, 3.0, 1.0),
                    'Al': (0, 3, 1.0), 'Mo': (0, 2, 0.6), 'Nb': (0, 0.2, 0.06), 'V': (0, 0.5, 0.12)}
        old = {e: {'start': f['start'].get(), 'end': f['end'].get(), 'step': f['step'].get()} for e, f in
               self.element_inputs.items()}
        self.element_inputs = {}
        r = 1
        for e in sorted(self.selected_elements):
            if e == "Fe": continue
            s, en, st = (old[e]['start'], old[e]['end'], old[e]['step']) if e in old else defaults.get(e,
                                                                                                       (0.1, 0.1, 0.1))
            ttk.Label(self.el_frame, text=e).grid(row=r, column=0, padx=5, pady=3)
            self.element_inputs[e] = {
                'start': self._num_entry(self.el_frame, r, 1, s),
                'end': self._num_entry(self.el_frame, r, 2, en),
                'step': self._num_entry(self.el_frame, r, 3, st),
            }
            r += 1

    def _num_entry(self, parent, row, col, default):
        """Create a numeric entry widget"""
        var = tk.DoubleVar(value=default)
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=col, padx=5, pady=3)
        return var

    def _make_temperature_section(self, parent):
        """Create the temperature range input section"""
        f = ttk.LabelFrame(parent, text="Temperature Range (K)", padding="10")
        f.pack(fill=tk.X, pady=5)
        ttk.Label(f, text="Exact output grid (applies to ALL outputs):", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, sticky=tk.W, columnspan=4, pady=(0, 6))
        self.t_start = self._temp_in(f, 1, "Start:", 573)
        self.t_end = self._temp_in(f, 1, "End:", 1073, off=2)
        self.t_step = self._temp_in(f, 2, "Step:", 1)
        ttk.Label(f,
                  text="Solver densification: linear axis with max step size ≈ 1 K; results are resampled to the grid above.",
                  foreground="gray").grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))

    def _temp_in(self, parent, row, label, default, off=0):
        """Create a temperature input field"""
        ttk.Label(parent, text=label).grid(row=row, column=off, padx=5, sticky=tk.W)
        var = tk.DoubleVar(value=default)
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=off + 1, padx=5)
        return var

    def _make_settings_section(self, parent):
        """Create the settings section"""
        f = ttk.LabelFrame(parent, text="Settings", padding="10")
        f.pack(fill=tk.X, pady=5)
        ttk.Label(f, text="Database:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.database = tk.StringVar(value="TCFE13")
        ttk.Entry(f, textvariable=self.database, width=15).grid(row=0, column=1, padx=5, sticky=tk.W)

        ttk.Label(f, text="Workers (parallel processes):").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.workers = tk.IntVar(value=8)
        ttk.Spinbox(f, from_=1, to=16, textvariable=self.workers, width=8).grid(row=0, column=3, padx=5, sticky=tk.W)

        ttk.Label(f, text="Cache:").grid(row=1, column=0, padx=5, sticky=tk.W, pady=5)
        self.cache = tk.StringVar(value="./cache/")
        ttk.Entry(f, textvariable=self.cache, width=30).grid(row=1, column=1, columnspan=2, padx=5, sticky=tk.W)
        ttk.Button(f, text="Browse", command=self._browse_cache).grid(row=1, column=3, padx=5)

    def _browse_cache(self):
        """Open folder browser for cache directory"""
        folder = filedialog.askdirectory(title="Select Cache Folder")
        if folder:
            self.cache.set(folder)

    def _make_buttons_section(self, parent):
        """Create the control buttons section"""
        f = ttk.Frame(parent, padding="10")
        f.pack(fill=tk.X, pady=10)

        row1 = ttk.Frame(f)
        row1.pack(fill=tk.X, pady=5)
        ttk.Button(row1, text="Advanced Options", command=self._show_advanced).pack(side=tk.LEFT, padx=3)
        ttk.Button(row1, text="Preview", command=self._preview).pack(side=tk.LEFT, padx=3)

        row2 = ttk.Frame(f)
        row2.pack(fill=tk.X, pady=5)
        self.run_btn = ttk.Button(row2, text="Run Calculations", command=self._run)
        self.run_btn.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        self.pause_btn = ttk.Button(row2, text="Pause", command=self._toggle_pause, state='disabled')
        self.pause_btn.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        self.stop_btn = ttk.Button(row2, text="Stop", command=self._stop, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        self.time_est_label = ttk.Label(f, text="Estimated time: --", font=('Arial', 10))
        self.time_est_label.pack(fill=tk.X, pady=(5, 2))

        self.progress = ttk.Progressbar(f, maximum=100, length=400)
        self.progress.pack(fill=tk.X, pady=5)

    def _make_log_section(self, parent):
        """Create the log output section"""
        f = ttk.LabelFrame(parent, text="Log", padding="5")
        f.pack(fill=tk.BOTH, expand=True, pady=5)
        sb = ttk.Scrollbar(f)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log = tk.Text(f, height=8, yscrollcommand=sb.set, wrap=tk.WORD)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.log.yview)

    def _show_advanced(self):
        """Open advanced options dialog"""
        win = tk.Toplevel(self.root)
        win.title("Advanced Options")
        win.geometry("1000x700")
        win.resizable(True, True)

        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tab_el = ttk.Frame(nb, padding="20")
        nb.add(tab_el, text="Alloying Elements")
        tab_ph = ttk.Frame(nb, padding="20")
        nb.add(tab_ph, text="Phases")
        tab_rs = ttk.Frame(nb, padding="20")
        nb.add(tab_rs, text="Results")

        self._adv_elements_tab(tab_el)
        self._adv_phases_tab(tab_ph)
        self._adv_results_tab(tab_rs)

        btns = ttk.Frame(win, padding="10")
        btns.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Button(btns, text="Cancel", command=lambda: self._adv_cancel(win)).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btns, text="OK", command=lambda: self._adv_ok(win)).pack(side=tk.RIGHT, padx=5)

        self._store_adv_state()

    def _adv_elements_tab(self, parent):
        """Create the alloying elements selection tab"""
        ttk.Label(parent, text="Select Alloying Elements (Fe is balance)", font=('Arial', 14, 'bold')).pack(
            pady=(0, 10))
        pt = ttk.Frame(parent)
        pt.pack(pady=10)
        self.element_buttons = {}
        self.temp_selected = self.selected_elements.copy()
        for e, (r, c) in self.AVAILABLE_ELEMENTS.items():
            b = tk.Button(pt, text=e, width=4, height=2, font=('Arial', 10, 'bold'),
                          relief=tk.RAISED, borderwidth=2, command=lambda x=e: self._toggle_element(x))
            if e in self.temp_selected:
                b.config(bg='#4CAF50', fg='white', relief=tk.SUNKEN)
            else:
                b.config(bg='white', fg='black')
            if e == 'Fe':
                b.config(state='disabled', bg='#E0E0E0')
            b.grid(row=r, column=c, padx=2, pady=2, sticky='nsew')
            self.element_buttons[e] = b
        sf = ttk.LabelFrame(parent, text="Selected", padding="10")
        sf.pack(fill=tk.X, pady=15)
        self.sel_label = ttk.Label(sf, text=", ".join(sorted(self.temp_selected)), font=('Arial', 11), wraplength=800)
        self.sel_label.pack()

    def _adv_phases_tab(self, parent):
        """Create the phases selection tab"""
        ttk.Label(parent, text="Select Phases", font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        ttk.Label(parent, text="Select at least one phase.", font=('Arial', 10), foreground='gray').pack(pady=(0, 15))

        canvas = tk.Canvas(parent, height=400)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        sf = ttk.Frame(canvas)
        sf.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.phase_vars = {ph: tk.BooleanVar(value=(ph in self.selected_phases)) for ph in self.AVAILABLE_PHASES}
        for ph in self.AVAILABLE_PHASES:
            fr = ttk.Frame(sf)
            fr.pack(fill=tk.X, padx=20, pady=3)
            ttk.Checkbutton(fr, text=ph, variable=self.phase_vars[ph]).pack(anchor=tk.W)

    def _adv_results_tab(self, parent):
        """Create the results selection tab"""
        ttk.Label(parent, text="Select Results to Export", font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        fr = ttk.LabelFrame(parent, text="Results Options", padding="20")
        fr.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        ttk.Checkbutton(fr, text="Phase Fraction (volume fraction of phases)", variable=self.calc_phase_fraction).pack(
            anchor=tk.W, pady=5)
        ttk.Checkbutton(fr, text="Phase Composition (mass fraction of elements in phases)",
                        variable=self.calc_phase_composition).pack(anchor=tk.W, pady=5)
        ttk.Label(fr, text="At least one result must be selected.", font=('Arial', 9, 'italic'),
                  foreground='gray').pack(anchor=tk.W, pady=(12, 0))

    def _toggle_element(self, e):
        """Toggle element selection"""
        if e == 'Fe':
            return
        if e in self.temp_selected:
            self.temp_selected.remove(e)
            self.element_buttons[e].config(bg='white', fg='black', relief=tk.RAISED)
        else:
            self.temp_selected.append(e)
            self.element_buttons[e].config(bg='#4CAF50', fg='white', relief=tk.SUNKEN)
        self.sel_label.config(text=", ".join(sorted(self.temp_selected)))

    def _store_adv_state(self):
        """Store current advanced options state for cancel functionality"""
        self._orig_elements = self.selected_elements.copy()
        self._orig_phases = self.selected_phases.copy()
        self._orig_results = (self.calc_phase_fraction.get(), self.calc_phase_composition.get())

    def _adv_ok(self, win):
        """Apply advanced options changes"""
        if len(self.temp_selected) < 2:
            messagebox.showwarning("Insufficient Elements", "Select at least one alloying element besides Fe.")
            return
        sel_ph = [ph for ph, v in self.phase_vars.items() if v.get()]
        if not sel_ph:
            messagebox.showwarning("No Phases", "Select at least one phase.")
            return
        if not (self.calc_phase_fraction.get() or self.calc_phase_composition.get()):
            messagebox.showwarning("No Results", "Select at least one result (Phase Fraction or Phase Composition).")
            return

        self.selected_elements = self.temp_selected.copy()
        self.selected_phases = sel_ph
        self._rebuild_element_inputs()
        win.destroy()
        self.root.after(100, self._size_to_content)
        messagebox.showinfo("Options Updated",
                            f"Elements: {', '.join(sorted(self.selected_elements))}\n"
                            f"Phases: {', '.join(self.selected_phases)}\n"
                            f"Results: "
                            f"{'Fraction ' if self.calc_phase_fraction.get() else ''}"
                            f"{'Composition' if self.calc_phase_composition.get() else ''}")

    def _adv_cancel(self, win):
        """Cancel advanced options changes"""
        self.selected_elements = self._orig_elements.copy()
        self.selected_phases = self._orig_phases.copy()
        self.calc_phase_fraction.set(self._orig_results[0])
        self.calc_phase_composition.set(self._orig_results[1])
        win.destroy()

    def _validate(self):
        """Validate input parameters"""
        try:
            for e, f in self.element_inputs.items():
                s, en, st = f['start'].get(), f['end'].get(), f['step'].get()
                if st <= 0:
                    raise ValueError(f"{e}: Step must be positive")
                if s > en:
                    raise ValueError(f"{e}: Start must be <= End")
            if self.t_step.get() <= 0:
                raise ValueError("Temperature step must be positive")
            if self.t_start.get() > self.t_end.get():
                raise ValueError("Start must be <= End")
            if not self.selected_phases:
                raise ValueError("Select at least one phase in Advanced Options")
            if not (self.calc_phase_fraction.get() or self.calc_phase_composition.get()):
                raise ValueError("Select at least one result (fraction or composition)")
            return True
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return False

    def _get_config(self):
        """Get current configuration"""
        cfg = {
            'database': getattr(self, 'database').get(),
            'cache': getattr(self, 'cache').get(),
            'workers': max(1, int(getattr(self, 'workers').get())),
            'selected_elements': self.selected_elements,
            'selected_phases': self.selected_phases,
            'temperature': {
                'start': float(self.t_start.get()),
                'end': float(self.t_end.get()),
                'step': float(self.t_step.get()),
            },
            'do_fraction': self.calc_phase_fraction.get(),
            'do_composition': self.calc_phase_composition.get(),
            'axis_max_step': float(self.AXIS_MAX_STEP_SIZE),
        }
        cfg['elements'] = {}
        for e, f in self.element_inputs.items():
            cfg['elements'][e] = {'start': f['start'].get(), 'end': f['end'].get(), 'step': f['step'].get()}
        return cfg

    def _preview(self):
        """Show calculation preview"""
        if not self._validate():
            return
        cfg = self._get_config()
        nT = int((cfg['temperature']['end'] - cfg['temperature']['start']) / cfg['temperature']['step']) + 1
        total = 1
        detail = []
        import itertools
        for e, p in cfg['elements'].items():
            pts = 1 if p['start'] == p['end'] else int((p['end'] - p['start']) / p['step']) + 1
            total *= pts
            detail.append(f"  {e}: {pts} pt(s)")
        msg = (f"Preview\n\n"
               f"Elements: {', '.join(sorted(cfg['selected_elements']))}\n"
               f"Compositions: {total:,}\n" + "\n".join(detail) + "\n\n"
                                                                  f"T grid points: {nT}\n"
                                                                  f"Phases: {', '.join(cfg['selected_phases'])}\n"
                                                                  f"Workers (processes): {cfg['workers']}\n"
                                                                  f"Results: "
                                                                  f"{'Fraction ' if cfg['do_fraction'] else ''}"
                                                                  f"{'Composition' if cfg['do_composition'] else ''}\n"
                                                                  f"Output: {self.OUTPUT_FILE}\n"
                                                                  f"Chunk size: {self.CHUNK_SIZE} alloys per intermediate file")
        self._log("=" * 40)
        self._log(msg)
        self._log("=" * 40)
        messagebox.showinfo("Preview", msg)

    def _run(self):
        """Start calculation process"""
        if not self._validate():
            return
        if not TC_PYTHON_AVAILABLE:
            messagebox.showerror("TC-Python Missing", "Install and license TC-Python to run calculations.")
            return
        if not messagebox.askyesno("Confirm", "Start thermodynamic calculations?"):
            return
        self.run_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        self.stop_btn.config(state='normal')
        self.stop_requested = False
        self.pause_requested = False
        self.progress['value'] = 0
        self.time_est_label.config(text="Estimated time: Calculating...")
        self._log("=" * 50)
        self._log("Starting calculations (parallel)...")
        self._log("=" * 50)
        self.calc_thread = threading.Thread(target=self._worker_parallel, daemon=True)
        self.calc_thread.start()

    def _stop(self):
        """Request calculation stop"""
        self.stop_requested = True
        self._log("Stop requested. Waiting for running tasks to finish...")
        self.stop_btn.config(state='disabled')

    def _toggle_pause(self):
        """Toggle between pause and continue states"""
        global pause_event

        if self.pause_requested:
            self.pause_requested = False
            self.pause_btn.config(text="Pause")
            if pause_event is not None:
                pause_event.clear()
            self._log("Calculations resumed.")
        else:
            self.pause_requested = True
            self.pause_btn.config(text="Continue")
            if pause_event is not None:
                pause_event.set()
            self._log("Pausing calculations... (current tasks will finish first)")

    def _format_time_estimate(self, seconds):
        """Format seconds into dd-hh-mm format"""
        if seconds <= 0:
            return "--"
        td = timedelta(seconds=int(seconds))
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{days:02d}-{hours:02d}-{minutes:02d}"

    def _worker_parallel(self):
        """Main parallel calculation worker with chunked saving"""
        try:
            cfg = self._get_config()
            os.makedirs(cfg['cache'], exist_ok=True)
            os.makedirs(self.TEMP_DIR, exist_ok=True)

            # Build composition list
            comp_list = []
            names = sorted([e for e in cfg['selected_elements'] if e != 'Fe'])
            ranges = []
            for e in names:
                p = cfg['elements'][e]
                if p['start'] == p['end']:
                    ranges.append([float(p['start'])])
                else:
                    ranges.append(list(np.arange(float(p['start']),
                                                 float(p['end']) + float(p['step']) / 2.0,
                                                 float(p['step']))))
            import itertools
            for vals in itertools.product(*ranges):
                comp_list.append(dict(zip(names, vals)))

            if not comp_list:
                self._log("No compositions to evaluate.")
                self._reset_buttons()
                return

            static = (cfg['database'], cfg['cache'], cfg['selected_elements'], cfg['selected_phases'],
                      cfg['temperature']['start'], cfg['temperature']['end'], cfg['temperature']['step'],
                      cfg['axis_max_step'], cfg['do_fraction'], cfg['do_composition'])

            total = len(comp_list)
            done = 0
            chunk_counter = 0
            current_chunk_rows = []
            chunk_files = []

            max_workers = max(1, int(cfg['workers']))
            self._log(f"Launching pool with {max_workers} worker process(es)...")
            self._log(f"Total alloys: {total:,} | Chunk size: {self.CHUNK_SIZE}")

            start_time = time.time()
            last_update_time = start_time
            last_log_time = start_time

            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                pending_futures = {}
                submitted = 0

                # Submit initial batch
                initial_batch = min(max_workers * 2, len(comp_list))
                for i in range(initial_batch):
                    if self.stop_requested:
                        break
                    fut = exe.submit(_compute_one_composition, static + (comp_list[i],))
                    pending_futures[fut] = i
                    submitted += 1

                while pending_futures and not self.stop_requested:
                    # Handle pause
                    while self.pause_requested and not self.stop_requested:
                        time.sleep(0.1)
                        self.root.update()

                    if self.stop_requested:
                        break

                    # Get completed future
                    completed = None
                    try:
                        for fut in as_completed(pending_futures.keys(), timeout=0.1):
                            completed = fut
                            break
                    except TimeoutError:
                        continue

                    if completed is None:
                        continue

                    try:
                        rows = completed.result()
                        current_chunk_rows.extend(rows)
                        del pending_futures[completed]
                    except Exception as e:
                        self._log(f"[Worker error] {e}")
                        del pending_futures[completed]

                    done += 1

                    # Submit next composition
                    if submitted < len(comp_list) and not self.pause_requested and not self.stop_requested:
                        new_fut = exe.submit(_compute_one_composition, static + (comp_list[submitted],))
                        pending_futures[new_fut] = submitted
                        submitted += 1

                    # Save chunk and clear memory
                    if len(current_chunk_rows) >= self.CHUNK_SIZE:
                        chunk_counter += 1
                        chunk_file = os.path.join(self.TEMP_DIR, f"chunk_{chunk_counter:04d}.csv")
                        df_chunk = pd.DataFrame(current_chunk_rows)

                        comp_cols = [c for c in df_chunk.columns if c.endswith("_content")]
                        ordered = comp_cols + ["Temperature"] + [c for c in df_chunk.columns if
                                                                 c not in comp_cols + ["Temperature"]]
                        df_chunk = df_chunk[ordered]
                        df_chunk.to_csv(chunk_file, index=False)
                        chunk_files.append(chunk_file)

                        current_chunk_rows.clear()
                        del df_chunk

                        self._log(f"Saved chunk {chunk_counter} to {chunk_file}")

                    # Update progress periodically
                    current_time = time.time()
                    if current_time - last_update_time >= 5.0:
                        pct = (done / total) * 100.0
                        self.root.after(0, lambda p=pct: self.progress.config(value=p))

                        elapsed = current_time - start_time
                        if done > 0 and elapsed > 0:
                            avg_time_per_alloy = elapsed / done
                            remaining = total - done
                            estimated_seconds = remaining * avg_time_per_alloy
                            time_str = self._format_time_estimate(estimated_seconds)
                            rate = done / (elapsed / 60.0)

                            self.root.after(0, lambda t=time_str, r=rate: self.time_est_label.config(
                                text=f"Est. time: {t} | Rate: {r:.1f} alloys/min"))

                        last_update_time = current_time

                    # Log progress periodically
                    if current_time - last_log_time >= 30.0:
                        pct = (done / total) * 100.0
                        elapsed = current_time - start_time
                        rate = done / (elapsed / 60.0) if elapsed > 0 else 0
                        self._log(f"Progress: {done}/{total} ({pct:.1f}%) - {rate:.1f} alloys/min")
                        last_log_time = current_time

            # Save remaining rows
            if current_chunk_rows:
                chunk_counter += 1
                chunk_file = os.path.join(self.TEMP_DIR, f"chunk_{chunk_counter:04d}.csv")
                df_chunk = pd.DataFrame(current_chunk_rows)
                comp_cols = [c for c in df_chunk.columns if c.endswith("_content")]
                ordered = comp_cols + ["Temperature"] + [c for c in df_chunk.columns if
                                                         c not in comp_cols + ["Temperature"]]
                df_chunk = df_chunk[ordered]
                df_chunk.to_csv(chunk_file, index=False)
                chunk_files.append(chunk_file)
                self._log(f"Saved final chunk {chunk_counter}")
                current_chunk_rows.clear()
                del df_chunk

            # Merge chunks
            if chunk_files:
                self._log(f"Merging {len(chunk_files)} chunks...")
                df_final = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
                df_final.to_csv(self.OUTPUT_FILE, index=False)
                self._log(f"Saved {len(df_final):,} rows to {self.OUTPUT_FILE}")

                # Cleanup temporary files
                for chunk_file in chunk_files:
                    try:
                        os.remove(chunk_file)
                    except:
                        pass
                try:
                    if os.path.exists(self.TEMP_DIR) and not os.listdir(self.TEMP_DIR):
                        os.rmdir(self.TEMP_DIR)
                except:
                    pass

            total_time = time.time() - start_time
            self._log(f"Total time: {self._format_time_estimate(total_time)}")
            self._log("Complete!" if not self.stop_requested else "Stopped by user.")

        except Exception as e:
            self._log(f"ERROR: {e}")
            import traceback
            self._log(traceback.format_exc())
        finally:
            self.root.after(0, self._reset_buttons)

    def _reset_buttons(self):
        """Reset button states after calculation"""
        global pause_event
        self.run_btn.config(state='normal')
        self.pause_btn.config(state='disabled', text="Pause")
        self.stop_btn.config(state='disabled')
        self.progress['value'] = 0
        self.time_est_label.config(text="Estimated time: --")
        self.pause_requested = False
        if pause_event is not None:
            pause_event.clear()

    def _log(self, msg):
        """Add message to log window"""
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.root.update_idletasks()


def main():
    """Main entry point"""
    if sys.platform.startswith("win"):
        import multiprocessing as mp
        mp.freeze_support()
    root = tk.Tk()
    app = ThermoCalcGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()