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
import csv
import gc
from collections import OrderedDict
import cProfile
import pstats
import io

# Detect operating system
PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == "Windows"
IS_MACOS = PLATFORM == "Darwin"
IS_LINUX = PLATFORM == "Linux"

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


class PerformanceTimer:
    """Track timing statistics for performance analysis"""
    def __init__(self):
        self.timings = {}
        self.counts = {}

    def time_operation(self, name):
        """Context manager for timing operations"""
        return TimerContext(self, name)

    def add_timing(self, name, duration):
        """Add a timing measurement"""
        if name not in self.timings:
            self.timings[name] = []
            self.counts[name] = 0
        self.timings[name].append(duration)
        self.counts[name] += 1

    def get_stats(self):
        """Get statistics for all operations"""
        stats = {}
        for name, times in self.timings.items():
            if times:
                stats[name] = {
                    'count': len(times),
                    'total': sum(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        return stats


class TimerContext:
    """Context manager for timing operations"""
    def __init__(self, timer, name):
        self.timer = timer
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        duration = time.time() - self.start
        self.timer.add_timing(self.name, duration)


def compute_one_composition(args):
    """
    Worker function with error handling, memory cleanup, 
    and phase existence optimization.
    """
    global pause_event
    if pause_event is not None:
        while pause_event.is_set():
            time.sleep(0.5)

    (database, cache, selected_elements, selected_phases,
     T0, T1, dT, axis_max_step, do_fraction, do_composition,
     composition_dict, enable_profiling) = args

    from tc_python import TCPython, ThermodynamicQuantity, CalculationAxis
    from tc_python.step_or_map_diagrams import AxisType
    import numpy as _np
    from collections import OrderedDict

    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    T_grid = _np.arange(float(T0), float(T1) + 1e-9, float(dT))
    rows = []

    try:
        with TCPython() as sess:
            system = (sess.set_cache_folder(cache)
                     .select_database_and_elements(database, selected_elements)
                     .get_system())
            calc = system.with_property_diagram_calculation()
            for e, wt in composition_dict.items():
                calc = calc.set_condition(f"W({e})", float(wt) / 100.0)
            calc = (calc.set_condition("N", 1).set_condition("P", 1e5)
                   .with_axis(CalculationAxis(ThermodynamicQuantity.temperature())
                   .set_min(float(T0)).set_max(float(T1))
                   .with_axis_type(AxisType.linear()
                   .set_max_step_size(axis_max_step or float(dT)))))

            try:
                res = calc.calculate()
            except Exception:
                return []  # Early exit on error

            # Phase existence masks for optimization
            frac_grid = {}
            phase_exists_mask = {}
            if do_fraction or do_composition:
                for ph in selected_phases:
                    Tx, Fy = res.get_values_of(
                        ThermodynamicQuantity.temperature(),
                        ThermodynamicQuantity.volume_fraction_of_a_phase(ph))
                    Tx = _np.asarray(Tx, float)
                    Fy = _np.asarray(Fy, float)
                    if _np.max(Fy) > 1e-6:
                        frac_grid[ph] = _np.interp(T_grid, Tx, Fy, left=0.0, right=0.0)
                        phase_exists_mask[ph] = frac_grid[ph] > 1e-6
                    else:
                        frac_grid[ph] = _np.zeros_like(T_grid)
                        phase_exists_mask[ph] = _np.zeros_like(T_grid, dtype=bool)

            # Composition extraction with phase mask
            comp_grid = {}
            if do_composition:
                for ph in selected_phases:
                    ph_dict = {}
                    if _np.max(frac_grid[ph]) < 1e-6:
                        for e in selected_elements:
                            ph_dict[e] = _np.full_like(T_grid, _np.nan)
                    else:
                        for e in selected_elements:
                            Tx, Wy = res.get_values_of(
                                ThermodynamicQuantity.temperature(),
                                ThermodynamicQuantity.composition_of_phase_as_weight_fraction(ph, e))
                            Tx = _np.asarray(Tx, float)
                            Wy = _np.asarray(Wy, float)
                            arr = _np.interp(T_grid, Tx, Wy, left=_np.nan, right=_np.nan)
                            arr = _np.where(phase_exists_mask[ph], arr, _np.nan)
                            ph_dict[e] = arr
                    comp_grid[ph] = ph_dict

            # Ordered row construction
            sorted_comp_elements = sorted(composition_dict.keys())
            for k, T in enumerate(T_grid):
                row = OrderedDict()
                for e in sorted_comp_elements:
                    row[f"{e}_content"] = composition_dict[e]
                row["Temperature"] = float(T)
                if do_fraction:
                    for ph in selected_phases:
                        row[f"{ph}_Fraction"] = float(frac_grid[ph][k])
                if do_composition:
                    for ph in selected_phases:
                        for e in selected_elements:
                            row[f"Mass_fraction_{e}_in_{ph}"] = comp_grid[ph][e][k]
                rows.append(row)

    finally:
        # Memory cleanup
        try:
            if 'res' in locals(): del res
            if 'calc' in locals(): del calc
            if 'system' in locals(): del system
            import gc
            gc.collect()
        except: pass
        if enable_profiling:
            profiler.disable()

    return rows
    
def save_chunk_fast(chunk_rows, filename):
    """Optimised CSV writing without Pandas overhead."""
    if not chunk_rows:
        return
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=chunk_rows[0].keys())
        writer.writeheader()
        writer.writerows(chunk_rows)

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


    DEFAULT_PHASES = ["FCC_A1", "BCC_A2", "CEMENTITE_D011", 
                      "FCC_A1#2", "M7C3_D101", "M23C6_D84"]
    
    AXIS_MAX_STEP_SIZE = 5.0  # Changed from 1.0
    OUTPUT_FILE = "Results_PhaseData.csv"
    CHUNK_SIZE = 2000  # Changed from 5000
    TEMP_DIR = "./temp_chunks/"
    PROGRESS_FILE = "./temp_chunks/progress.json"
    COMPOSITIONS_FILE = "./temp_chunks/compositions.json"
    TIMING_FILE = "./temp_chunks/timing_stats.json"
    PROGRESS_SAVE_INTERVAL = 300
    UI_UPDATE_INTERVAL = 20


    def __init__(self, root):
        self.root = root
        platform_name = "Windows" if IS_WINDOWS else ("macOS" if IS_MACOS else "Linux")
        self.root.title(f"Thermodynamic Calculator (Optimised) - {platform_name}")
        self.root.resizable(True, True)

        self.stop_requested = False
        self.pause_requested = False
        self.calc_thread = None

        # Performance tracking
        self.perf_timer = PerformanceTimer()
        self.enable_profiling = tk.BooleanVar(value=False)

        
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

        self.build_ui()
        self.root.after(100, self.size_to_content)

    def build_ui(self):
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

        self.make_elements_section(self.container)
        self.make_temperature_section(self.container)
        self.make_settings_section(self.container)
        self.make_buttons_section(self.container)
        self.make_log_section(self.container)

    def size_to_content(self, h=24, v=80):
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

    def make_elements_section(self, parent):
        """Create the element composition input section"""
        self.el_frame = ttk.LabelFrame(parent, text="Element Composition (wt%)", padding="10")
        self.el_frame.pack(fill=tk.X, pady=5)
        self.rebuild_element_inputs()

    def rebuild_element_inputs(self):
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
                'start': self.num_entry(self.el_frame, r, 1, s),
                'end': self.num_entry(self.el_frame, r, 2, en),
                'step': self.num_entry(self.el_frame, r, 3, st),
            }
            r += 1

    def num_entry(self, parent, row, col, default):
        """Create a numeric entry widget"""
        var = tk.DoubleVar(value=default)
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=col, padx=5, pady=3)
        return var

    def make_temperature_section(self, parent):
        """Create the temperature range input section"""
        f = ttk.LabelFrame(parent, text="Temperature Range (K)", padding="10")
        f.pack(fill=tk.X, pady=5)
        ttk.Label(f, text="Exact output grid (applies to ALL outputs):", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, sticky=tk.W, columnspan=4, pady=(0, 6))
        self.t_start = self.temp_in(f, 1, "Start:", 573)
        self.t_end = self.temp_in(f, 1, "End:", 1073, off=2)
        self.t_step = self.temp_in(f, 2, "Step:", 1)
        ttk.Label(f,
                  text="Solver densification: linear axis with max step size ≈ 1 K; results are resampled to the grid above.",
                  foreground="gray").grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))

    def temp_in(self, parent, row, label, default, off=0):
        """Create a temperature input field"""
        ttk.Label(parent, text=label).grid(row=row, column=off, padx=5, sticky=tk.W)
        var = tk.DoubleVar(value=default)
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=off + 1, padx=5)
        return var

    def make_settings_section(self, parent):
        """Create the settings section"""
        f = ttk.LabelFrame(parent, text="Settings", padding="10")
        f.pack(fill=tk.X, pady=5)
        ttk.Label(f, text="Database:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.database = tk.StringVar(value="TCFE13")
        ttk.Entry(f, textvariable=self.database, width=15).grid(row=0, column=1, padx=5, sticky=tk.W)

        ttk.Label(f, text="Workers (parallel processes):").grid(...)
        # Conservative default: half CPU cores, max 6
        cpu_count = os.cpu_count() or 4
        perf_cores_est = max(1, cpu_count // 2)
        default_workers = min(perf_cores_est, 6)
        self.workers = tk.IntVar(value=default_workers)
        ttk.Spinbox(f, from_=1, to=8, textvariable=self.workers, width=8).grid(...)
        
        # ADD profiling checkbox
        ttk.Checkbutton(f, text="Enable detailed profiling (slower)",
                       variable=self.enable_profiling).grid(row=2, column=0, 
                       columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(f, text="Cache:").grid(row=1, column=0, padx=5, sticky=tk.W, pady=5)
        self.cache = tk.StringVar(value="./cache/")
        ttk.Entry(f, textvariable=self.cache, width=30).grid(row=1, column=1, columnspan=2, padx=5, sticky=tk.W)
        ttk.Button(f, text="Browse", command=self.browse_cache).grid(row=1, column=3, padx=5)

    def browse_cache(self):
        """Open folder browser for cache directory"""
        folder = filedialog.askdirectory(title="Select Cache Folder")
        if folder:
            self.cache.set(folder)

    def make_buttons_section(self, parent):
        """Create the control buttons section"""
        f = ttk.Frame(parent, padding="10")
        f.pack(fill=tk.X, pady=10)

        row1 = ttk.Frame(f)
        row1.pack(fill=tk.X, pady=5)
        ttk.Button(row1, text="Advanced Options", command=self.show_advanced).pack(...)
        ttk.Button(row1, text="Preview", command=self.preview).pack(...)
        ttk.Button(row1, text="Resume Previous", command=self.check_resume).pack(...)
        ttk.Button(row1, text="View Stats", command=self.show_timing_stats).pack(...)

        row2 = ttk.Frame(f)
        row2.pack(fill=tk.X, pady=5)
        self.run_btn = ttk.Button(row2, text="Run Calculations", command=self.run)
        self.run_btn.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        self.pause_btn = ttk.Button(row2, text="Pause", command=self.toggle_pause, state='disabled')
        self.pause_btn.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        self.stop_btn = ttk.Button(row2, text="Stop", command=self.stop, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        self.time_est_label = ttk.Label(f, text="Estimated time: --", font=('Arial', 10))
        self.time_est_label.pack(fill=tk.X, pady=(5, 2))

        self.progress = ttk.Progressbar(f, maximum=100, length=400)
        self.progress.pack(fill=tk.X, pady=5)

    def make_log_section(self, parent):
        """Create the log output section"""
        f = ttk.LabelFrame(parent, text="Log", padding="5")
        f.pack(fill=tk.BOTH, expand=True, pady=5)
        sb = ttk.Scrollbar(f)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log = tk.Text(f, height=8, yscrollcommand=sb.set, wrap=tk.WORD)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.log.yview)

    def show_advanced(self):
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

        self.adv_elements_tab(tab_el)
        self.adv_phases_tab(tab_ph)
        self.adv_results_tab(tab_rs)

        btns = ttk.Frame(win, padding="10")
        btns.pack(fill=tk.X, side=tk.BOTTOM)
        ttk.Button(btns, text="Cancel", command=lambda: self.adv_cancel(win)).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btns, text="OK", command=lambda: self.adv_ok(win)).pack(side=tk.RIGHT, padx=5)

        self.store_adv_state()

    def adv_elements_tab(self, parent):
        """Create the alloying elements selection tab"""
        ttk.Label(parent, text="Select Alloying Elements (Fe is balance)", font=('Arial', 14, 'bold')).pack(
            pady=(0, 10))
        pt = ttk.Frame(parent)
        pt.pack(pady=10)
        self.element_buttons = {}
        self.temp_selected = self.selected_elements.copy()
        for e, (r, c) in self.AVAILABLE_ELEMENTS.items():
            b = tk.Button(pt, text=e, width=4, height=2, font=('Arial', 10, 'bold'),
                          relief=tk.RAISED, borderwidth=2, command=lambda x=e: self.toggle_element(x))
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

    def adv_phases_tab(self, parent):
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

    def adv_results_tab(self, parent):
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

    def toggle_element(self, e):
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

    def store_adv_state(self):
        """Store current advanced options state for cancel functionality"""
        self._orig_elements = self.selected_elements.copy()
        self._orig_phases = self.selected_phases.copy()
        self._orig_results = (self.calc_phase_fraction.get(), self.calc_phase_composition.get())

    def adv_ok(self, win):
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
        self.rebuild_element_inputs()
        win.destroy()
        self.root.after(100, self.size_to_content)
        messagebox.showinfo("Options Updated",
                            f"Elements: {', '.join(sorted(self.selected_elements))}\n"
                            f"Phases: {', '.join(self.selected_phases)}\n"
                            f"Results: "
                            f"{'Fraction ' if self.calc_phase_fraction.get() else ''}"
                            f"{'Composition' if self.calc_phase_composition.get() else ''}")

    def adv_cancel(self, win):
        """Cancel advanced options changes"""
        self.selected_elements = self._orig_elements.copy()
        self.selected_phases = self._orig_phases.copy()
        self.calc_phase_fraction.set(self._orig_results[0])
        self.calc_phase_composition.set(self._orig_results[1])
        win.destroy()

    def validate(self):
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

    def get_config(self):
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
            'enable_profiling': self.enable_profiling.get(),
        }
        cfg['elements'] = {}
        for e, f in self.element_inputs.items():
            cfg['elements'][e] = {'start': f['start'].get(), 'end': f['end'].get(), 'step': f['step'].get()}
        return cfg

    def preview(self):
        """Show calculation preview"""
        if not self.validate():
            return
        cfg = self.get_config()
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
        self.log("=" * 40)
        self.log(msg)
        self.log("=" * 40)
        messagebox.showinfo("Preview", msg)

    def check_resume(self):
        """Check if previous calculation can be resumed"""
        if os.path.exists(self.PROGRESS_FILE):
            try:
                with open(self.PROGRESS_FILE, 'r') as f:
                    progress = json.load(f)
                completed = progress.get('completed', 0)
                total = progress.get('total', 0)
                if completed >= total:
                    messagebox.showinfo("Already Complete", 
                        "Previous calculation was already completed.")
                    return
                msg = (f"Previous calculation found:\n\n"
                       f"Completed: {completed}/{total} alloys "
                       f"({(completed/total*100):.1f}%)\n"
                       f"Remaining: {total-completed} alloys\n\n"
                       f"Resume from where it left off?")
                if messagebox.askyesno("Resume Calculation", msg):
                    self.start_calculation(resume=True)
                return
            except: pass
        
        # Check for orphaned chunks
        if os.path.exists(self.TEMP_DIR):
            chunk_files = sorted([f for f in os.listdir(self.TEMP_DIR) 
                                if f.startswith('chunk_') and f.endswith('.csv')])
            if chunk_files:
                msg = (f"Found {len(chunk_files)} orphaned chunk files.\n\n"
                       f"Options:\n"
                       f"Yes = Merge only\n"
                       f"No = Analyze & Resume\n"
                       f"Cancel = Delete chunks")
                response = messagebox.askyesnocancel("Recovery Mode", msg)
                if response is True:
                    self.merge_orphaned_chunks()
                elif response is False:
                    self.recover_from_chunks()
                elif response is None:
                    if messagebox.askyesno("Confirm Delete", 
                        "Delete all chunk files?"):
                        self.cleanup_chunks()
                return
        messagebox.showinfo("No Previous Run", 
            "No previous calculation found to resume.")

    def show_timing_stats(self):
        """Show timing statistics if available"""
        if os.path.exists(self.TIMING_FILE):
            try:
                with open(self.TIMING_FILE, 'r') as f:
                    timing_data = json.load(f)
                stats_win = tk.Toplevel(self.root)
                stats_win.title("Performance Statistics")
                stats_win.geometry("800x600")
                text_widget = tk.Text(stats_win, wrap=tk.WORD, font=('Courier', 9))
                scrollbar = ttk.Scrollbar(stats_win, command=text_widget.yview)
                text_widget.configure(yscrollcommand=scrollbar.set)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                text_widget.insert('1.0', json.dumps(timing_data, indent=2))
                text_widget.config(state='disabled')
            except Exception as e:
                messagebox.showerror("Error", f"Could not load stats: {e}")
        else:
            messagebox.showinfo("No Stats", 
                "No timing statistics available. Run a calculation first.")

        
    def run(self):
        """Start calculation process"""
        if not self.validate():
            return
        if not TC_PYTHON_AVAILABLE:
            messagebox.showerror("TC-Python Missing", 
                "Install and license TC-Python to run calculations.")
            return
        if not messagebox.askyesno("Confirm", "Start thermodynamic calculations?"):
            return
        self.start_calculation(resume=False)

    def start_calculation(self, resume=False):
        """Start or resume calculation process"""
        if not resume and not self.validate():
            return
        if not TC_PYTHON_AVAILABLE:
            messagebox.showerror("TC-Python Missing", 
                "Install and license TC-Python to run calculations.")
            return
        self.run_btn.config(state='disabled')
        self.pause_btn.config(state='normal')
        self.stop_btn.config(state='normal')
        self.stop_requested = False
        self.pause_requested = False
        self.progress['value'] = 0
        self.time_est_label.config(text="Estimated time: Calculating...")
        self.log("=" * 50)
        if resume:
            self.log("Resuming calculations from previous progress...")
        else:
            self.log("Starting calculations (parallel, optimised)...")
        self.log("=" * 50)
        self.calc_thread = threading.Thread(
            target=lambda: self.worker_parallel(resume=resume), daemon=True)
        self.calc_thread.start()

    def stop(self):
        """Request calculation stop"""
        self.stop_requested = True
        self.log("Stop requested. Waiting for running tasks to finish...")
        self.stop_btn.config(state='disabled')

    def toggle_pause(self):
        """Toggle between pause and continue states"""
        global pause_event

        if self.pause_requested:
            self.pause_requested = False
            self.pause_btn.config(text="Pause")
            if pause_event is not None:
                pause_event.clear()
            self.log("Calculations resumed.")
        else:
            self.pause_requested = True
            self.pause_btn.config(text="Continue")
            if pause_event is not None:
                pause_event.set()
            self.log("Pausing calculations... (current tasks will finish first)")

    def format_time_estimate(self, seconds):
        """Format seconds into dd-hh-mm format"""
        if seconds <= 0:
            return "--"
        td = timedelta(seconds=int(seconds))
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        return f"{days:02d}-{hours:02d}-{minutes:02d}"


    def worker_parallel(self, resume=False):
        """Main parallel calculation worker with resume and progress tracking"""
        overall_start = time.time()
        try:
            cfg = self.get_config()
            # ... initialization code ...
            
            # Resume logic
            comp_list = []
            completed_indices = set()
            chunk_counter = 0
            chunk_files = []
            
            if resume and os.path.exists(self.PROGRESS_FILE):
                self.log("Loading previous progress...")
                with open(self.PROGRESS_FILE, 'r') as f:
                    progress = json.load(f)
                with open(self.COMPOSITIONS_FILE, 'r') as f:
                    comp_data = json.load(f)
                    comp_list = comp_data['compositions']
                completed_indices = set(progress.get('completed_indices', []))
                chunk_counter = progress.get('chunk_counter', 0)
                # ... load existing chunks ...
            else:
                # Build new composition list and save it
                # ... composition generation code ...
                with open(self.COMPOSITIONS_FILE, 'w') as f:
                    json.dump({'compositions': comp_list, 'config': cfg}, f)
            
            # Track composition times for statistics
            composition_times = []
            
            # ... parallel execution with timing ...
            
            # Save timing statistics
            if composition_times:
                timing_stats = {
                    'platform': PLATFORM,
                    'total_time_seconds': total_time,
                    'total_compositions': done - initial_completed,
                    'avg_time_per_composition': sum(composition_times) / len(composition_times),
                    'min_time': min(composition_times),
                    'max_time': max(composition_times),
                    'median_time': sorted(composition_times)[len(composition_times)//2],
                    'workers_used': max_workers,
                    'compositions_per_minute': (done - initial_completed) / (total_time / 60.0),
                }
                with open(self.TIMING_FILE, 'w') as f:
                    json.dump(timing_stats, f, indent=2)
        
        except Exception as e:
            self.log(f"ERROR: {e}")
            import traceback
            self.log(traceback.format_exc())
        finally:
            self.root.after(0, self.reset_buttons)

    def save_progress(self, completed_indices, total, chunk_counter):
        """Save current progress to file"""
        try:
            progress = {
                'completed': len(completed_indices),
                'total': total,
                'completed_indices': list(completed_indices),
                'chunk_counter': chunk_counter,
                'timestamp': time.time(),
                'platform': PLATFORM,
            }
            with open(self.PROGRESS_FILE, 'w') as f:
                json.dump(progress, f)
        except Exception as e:
            self.log(f"Warning: Could not save progress: {e}")

    def merge_orphaned_chunks(self):
        """Merge existing chunk files into final output"""
        # ... see full implementation in updated code ...

    def recover_from_chunks(self):
        """Analyze chunks and resume remaining calculations"""
        # ... see full implementation in updated code ...

    def cleanup_chunks(self):
        """Delete all chunk files and progress data"""
        try:
            if os.path.exists(self.TEMP_DIR):
                for f in os.listdir(self.TEMP_DIR):
                    try:
                        os.remove(os.path.join(self.TEMP_DIR, f))
                    except: pass
                try:
                    os.rmdir(self.TEMP_DIR)
                except: pass
        except Exception as e:
            self.log(f"Error cleaning up: {e}")


    
    def reset_buttons(self):
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

    def log(self, msg):
        """Add message to log window"""
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.root.update_idletasks()


def main():
    """Main entry point"""
    # Platform-specific multiprocessing setup
    if IS_WINDOWS:
        import multiprocessing as mp
        mp.freeze_support()
    
    # Log platform information
    print(f"Starting Thermodynamic Calculator on {PLATFORM}")
    print(f"Python version: {sys.version}")
    print(f"CPU cores: {os.cpu_count()}")
    
    root = tk.Tk()
    app = ThermoCalcGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
