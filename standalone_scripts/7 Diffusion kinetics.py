#!/usr/bin/env python3
"""
DICTRA Diffusion Calculator - TC-Python GUI

A graphical interface for DICTRA diffusion simulations using TC-Python.
Designed for computational materials science research and alloy design.

Features:
    - Multi-region diffusion couple setup
    - Isothermal and non-isothermal calculations  
    - Customizable boundary conditions
    - Batch processing from Excel input
    - Composition profile results and visualization
    - Excel export of simulation results

Requirements:
    - TC-Python (Thermo-Calc Python API)
    - matplotlib (for visualization)
    - pandas, openpyxl (for Excel I/O)

Usage:
    python dictra1.py

License: MIT
"""

# Standard library imports
import csv
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# TC-Python import with availability check
try:
    import tc_python
    from tc_python import TCPython
    from tc_python.diffusion import (
        Region, CompositionProfile,
        CalculatedGrid, LinearGrid, GeometricGrid,
        BoundaryCondition, ConstantProfile, Unit, Options,
        TemperatureProfile, TimestepControl,
        HomogenizationSolver, HomogenizationFunctions
    )
    TC_PYTHON_AVAILABLE = True
except ImportError:
    TC_PYTHON_AVAILABLE = False
    print("WARNING: TC-Python not available. GUI will run but calculations will fail.")

# Optional: Plotting libraries
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Optional: Data processing libraries
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False



class DICTRACalculatorGUI:
    """Main GUI class for DICTRA diffusion calculations."""
    
    # Default database names
    DEFAULT_TDB = "TCFE13"  # Thermo-Calc steel database
    DEFAULT_MDB = "MOBFE8"  # Mobility database for steels
    DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".tc_python_cache")
    
    def __init__(self, root):
        """Initialize the DICTRA calculator GUI."""
        self.root = root
        self.root.title("DICTRA Diffusion Calculator1 - TC-Python")
        self.root.geometry("1200x800")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('Title.TLabel', font=('Helvetica', 12, 'bold'))
        self.style.configure('Section.TLabelframe.Label', font=('Helvetica', 10, 'bold'))
        
        # Initialize variables
        self.init_variables()
        
        # Region storage (must be before create_main_frame)
        self.regions = []  # List of region dictionaries
        
        # Calculation state
        self.calculation_running = False
        self.stop_requested = False
        self.calc_thread = None
        
        # Create main UI
        self.create_menu()
        self.create_main_frame()
        
    def init_variables(self):
        """Initialize all tk variables."""
        # Database paths
        self.tdb_path = tk.StringVar(value=self.DEFAULT_TDB)
        self.mdb_path = tk.StringVar(value=self.DEFAULT_MDB)
        self.cache_path = tk.StringVar(value=self.DEFAULT_CACHE)
        
        # Simulation parameters
        self.temperature = tk.DoubleVar(value=1173.0)  # Kelvin (900°C - stable austenite)
        self.simulation_time = tk.StringVar(value="3600")  # seconds (space-separated for multiple times)
        self.geometry_type = tk.StringVar(value="Planar")
        
        # Calculation type
        self.calc_type = tk.StringVar(value="Isothermal")
        
        # Boundary conditions
        self.left_boundary_type = tk.StringVar(value="Closed System")
        self.right_boundary_type = tk.StringVar(value="Closed System")
        
        # Solver settings
        self.solver_type = tk.StringVar(value="Automatic")
        self.grid_type = tk.StringVar(value="Automatic - Medium")
        self.grid_points = tk.IntVar(value=50)
        
        # Timestep control
        self.min_timestep = tk.DoubleVar(value=1e-10)
        self.max_timestep = tk.DoubleVar(value=1e6)
        self.timestep_increase_factor = tk.DoubleVar(value=1.5)
        
        # Element selection
        self.selected_elements = []
        self.element_vars = {}
        
        # Output settings
        self.output_file = tk.StringVar(value="")
        
        # Non-isothermal thermal profile: list of (time_s, temp_K) tuples
        self.thermal_segments = []
        self.thermal_frame = None  # Will be created in setup tab
        self.isothermal_frame = None  # Frame for isothermal inputs
        
        # Output options
        self.skip_phase_fractions = tk.BooleanVar(value=False)
        
        # Homogenization solver settings
        self.homogenization_function = tk.StringVar(value="Rule of mixtures (upper Wiener bound)")
        self.use_global_minimization = tk.BooleanVar(value=False)
        self.use_interpolation_scheme = tk.BooleanVar(value=True)
        self.interpolation_type = tk.StringVar(value="Logarithmic")
        self.interpolation_steps = tk.IntVar(value=10000)
        self.interpolation_memory = tk.DoubleVar(value=1000.0)
        self.interpolation_memory_unit = tk.StringVar(value="Megabyte")
        
        # Batch mode settings (Excel import)
        self.batch_mode = tk.BooleanVar(value=False)
        self.batch_excel_path = tk.StringVar(value="")
        self.batch_output_dir = tk.StringVar(value="./dictra_results/")
        self.batch_alloys = []  # List of parsed alloy configurations
        self.batch_progress_file = "./dictra_results/progress.json"
        
    def create_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Setup", command=self.new_setup)
        file_menu.add_command(label="Load Setup...", command=self.load_setup)
        file_menu.add_command(label="Save Setup...", command=self.save_setup)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_main_frame(self):
        """Create the main application frame with tabs."""
        # Main container
        main_container = ttk.Frame(self.root, padding="5")
        main_container.pack(fill='both', expand=True)
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill='both', expand=True, pady=(0, 5))
        
        # Create tabs
        self.create_setup_tab()
        self.create_regions_tab()
        self.create_boundary_tab()
        self.create_advanced_tab()
        self.create_results_tab()
        
        # Bottom frame with run controls
        self.create_control_frame(main_container)
        
    def create_setup_tab(self):
        """Create the Setup tab for databases and basic parameters."""
        setup_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(setup_frame, text="Setup")
        
        # === Database Selection ===
        db_frame = ttk.LabelFrame(setup_frame, text="Databases", padding="10", style='Section.TLabelframe')
        db_frame.pack(fill='x', pady=(0, 10))
        
        # TDB file
        ttk.Label(db_frame, text="Thermodynamic Database (TDB):").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(db_frame, textvariable=self.tdb_path, width=60).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(db_frame, text="Browse...", command=lambda: self.browse_file(self.tdb_path, [("TDB files", "*.tdb"), ("All files", "*.*")])).grid(row=0, column=2, pady=2)
        
        # MDB file (mobility database)
        ttk.Label(db_frame, text="Mobility Database (MDB):").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(db_frame, textvariable=self.mdb_path, width=60).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(db_frame, text="Browse...", command=lambda: self.browse_file(self.mdb_path, [("MDB files", "*.mdb"), ("All files", "*.*")])).grid(row=1, column=2, pady=2)
        
        # Cache directory
        ttk.Label(db_frame, text="Cache Directory:").grid(row=2, column=0, sticky='w', pady=2)
        ttk.Entry(db_frame, textvariable=self.cache_path, width=60).grid(row=2, column=1, padx=5, pady=2)
        ttk.Button(db_frame, text="Browse...", command=lambda: self.browse_directory(self.cache_path)).grid(row=2, column=2, pady=2)
        
        # === Element Selection ===
        elem_frame = ttk.LabelFrame(setup_frame, text="Elements", padding="10", style='Section.TLabelframe')
        elem_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(elem_frame, text="Select elements (check the elements in your system):").pack(anchor='w')
        
        # Common elements for diffusion
        self.elements_container = ttk.Frame(elem_frame)
        self.elements_container.pack(fill='x', pady=5)
        
        # Elements matching phases.py AVAILABLE_ELEMENTS
        common_elements = [
            'Fe', 'C', 'Mn', 'Si', 'Al', 'Mo', 'Nb', 'V',  # DEFAULT_ELEMENTS from phases.py
            'B', 'N', 'O', 'Mg', 'P', 'S', 'Ca', 'Sc', 'Ti', 'Cr',  # Additional from phases.py
            'Co', 'Ni', 'Cu', 'Zn', 'Cs', 'Ta', 'W'
        ]
        # Default elements matching phases.py: Fe, C, Mn, Si, Al, Mo, Nb, V
        default_elements = ['Fe', 'C', 'Mn', 'Si', 'Al', 'Mo', 'Nb', 'V']
        for i, elem in enumerate(common_elements):
            var = tk.BooleanVar(value=(elem in default_elements))
            self.element_vars[elem] = var
            # Add trace to sync with regions when element selection changes
            var.trace('w', lambda *args, e=elem: self.sync_elements_to_regions())
            cb = ttk.Checkbutton(self.elements_container, text=elem, variable=var)
            cb.grid(row=i//7, column=i%7, sticky='w', padx=10, pady=2)
        
        # === Simulation Parameters ===
        sim_frame = ttk.LabelFrame(setup_frame, text="Simulation Parameters", padding="10", style='Section.TLabelframe')
        sim_frame.pack(fill='x', pady=(0, 10))
        
        # Calculation type with toggle handler
        ttk.Label(sim_frame, text="Calculation Type:").grid(row=0, column=0, sticky='w', pady=5)
        calc_type_combo = ttk.Combobox(sim_frame, textvariable=self.calc_type, values=["Isothermal", "Non-Isothermal"], 
                     state='readonly', width=20)
        calc_type_combo.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        calc_type_combo.bind('<<ComboboxSelected>>', lambda e: self.toggle_calc_type())
        
        # Geometry (always visible)
        ttk.Label(sim_frame, text="Geometry:").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Combobox(sim_frame, textvariable=self.geometry_type, 
                     values=["Planar", "Cylindrical", "Spherical"], 
                     state='readonly', width=20).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # === Isothermal Frame (shown by default) ===
        self.isothermal_frame = ttk.Frame(sim_frame)
        self.isothermal_frame.grid(row=2, column=0, columnspan=4, sticky='w', pady=5)
        
        ttk.Label(self.isothermal_frame, text="Temperature (K):").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(self.isothermal_frame, textvariable=self.temperature, width=15).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        # Celsius display
        self.temp_c_label = ttk.Label(self.isothermal_frame, text="")
        self.temp_c_label.grid(row=0, column=2, sticky='w', pady=5)
        def update_temp_display(*args):
            try:
                kelvin = self.temperature.get()
                celsius = kelvin - 273.15
                self.temp_c_label.config(text=f"= {celsius:.1f} °C")
            except:
                pass
        self.temperature.trace('w', update_temp_display)
        update_temp_display()
        
        ttk.Label(self.isothermal_frame, text="Simulation Time (s):").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(self.isothermal_frame, textvariable=self.simulation_time, width=15).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # === Non-Isothermal Frame (hidden by default) ===
        self.thermal_frame = ttk.Frame(sim_frame)
        # Don't grid it yet - will be shown when Non-Isothermal is selected
        
        ttk.Label(self.thermal_frame, text="Thermal Profile (Time-Temperature):", 
                  font=('Helvetica', 9, 'bold')).grid(row=0, column=0, columnspan=4, sticky='w', pady=(0, 5))
        
        # Segment table header
        ttk.Label(self.thermal_frame, text="Time (s)").grid(row=1, column=0, padx=5)
        ttk.Label(self.thermal_frame, text="Temperature (K)").grid(row=1, column=1, padx=5)
        ttk.Label(self.thermal_frame, text="°C").grid(row=1, column=2, padx=5)
        
        # Scrollable container for segments
        self.segments_container = ttk.Frame(self.thermal_frame)
        self.segments_container.grid(row=2, column=0, columnspan=4, sticky='w', pady=5)
        self.segment_widgets = []  # List of (time_entry, temp_entry, temp_var, celsius_label) tuples
        
        # Buttons for add/remove
        btn_frame = ttk.Frame(self.thermal_frame)
        btn_frame.grid(row=3, column=0, columnspan=4, sticky='w', pady=5)
        ttk.Button(btn_frame, text="+ Add Point", command=self.add_thermal_segment).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="- Remove Last", command=self.remove_thermal_segment).pack(side='left', padx=5)
        
        # Add two default points for non-isothermal (quench from 660°C to 25°C)
        self.add_thermal_segment(time_val=0, temp_val=933.15)
        self.add_thermal_segment(time_val=6.35, temp_val=298.15)

        ttk.Button(btn_frame, text="Load Profile (CSV)", command=self.load_thermal_profile).pack(side='left', padx=20)

    def load_thermal_profile(self):
        """Load thermal profile from a CSV file (Time, Temperature)."""
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt")])
        if not filename:
            return
            
        try:
            segments = []
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Skip headers or empty lines
                    if not row or not row[0][0].isdigit():
                        continue
                    if len(row) >= 2:
                        t = float(row[0])
                        T = float(row[1])
                        # Check units? Assume Seconds and Kelvin for now
                        segments.append((t, T))
            
            if len(segments) < 2:
                messagebox.showerror("Error", "File must contain at least 2 time-temperature points.")
                return
                
            # Clear existing
            for _, _, time_entry, temp_entry, celsius_label in self.segment_widgets:
                time_entry.destroy()
                temp_entry.destroy()
                celsius_label.destroy()
            self.segment_widgets = []
            
            # Add new points
            segments.sort(key=lambda x: x[0])
            for t, T in segments:
                self.add_thermal_segment(t, T)
                
            self.write_log(f"Loaded {len(segments)} points from {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load profile: {e}\nFormat: Time,Temperature (rows)")
    
    def toggle_calc_type(self):
        """Toggle visibility of isothermal/non-isothermal input frames."""
        if self.calc_type.get() == "Isothermal":
            self.thermal_frame.grid_forget()
            self.isothermal_frame.grid(row=2, column=0, columnspan=4, sticky='w', pady=5)
        else:
            self.isothermal_frame.grid_forget()
            self.thermal_frame.grid(row=2, column=0, columnspan=4, sticky='w', pady=5)
    
    def add_thermal_segment(self, time_val=0, temp_val=1173):
        """Add a temperature-time point to the thermal profile."""
        row = len(self.segment_widgets)
        
        # Time entry
        time_var = tk.DoubleVar(value=time_val)
        time_entry = ttk.Entry(self.segments_container, textvariable=time_var, width=12)
        time_entry.grid(row=row, column=0, padx=5, pady=2)
        
        # Temperature entry
        temp_var = tk.DoubleVar(value=temp_val)
        temp_entry = ttk.Entry(self.segments_container, textvariable=temp_var, width=12)
        temp_entry.grid(row=row, column=1, padx=5, pady=2)
        
        # Celsius display
        celsius_label = ttk.Label(self.segments_container, text=f"= {temp_val - 273.15:.1f}")
        celsius_label.grid(row=row, column=2, padx=5, pady=2)
        
        # Update Celsius when temperature changes
        def update_celsius(*args, lbl=celsius_label, tvar=temp_var):
            try:
                lbl.config(text=f"= {tvar.get() - 273.15:.1f}")
            except:
                pass
        temp_var.trace('w', update_celsius)
        
        self.segment_widgets.append((time_var, temp_var, time_entry, temp_entry, celsius_label))
    
    def remove_thermal_segment(self):
        """Remove the last temperature-time point."""
        if len(self.segment_widgets) > 2:  # Keep at least 2 points
            time_var, temp_var, time_entry, temp_entry, celsius_label = self.segment_widgets.pop()
            time_entry.destroy()
            temp_entry.destroy()
            celsius_label.destroy()
        else:
            messagebox.showinfo("Cannot Remove", "At least 2 temperature points are required.")
    
    def get_thermal_profile(self):
        """Get the thermal profile as a list of (time, temperature) tuples, sorted by time."""
        segments = []
        for time_var, temp_var, *widgets in self.segment_widgets:
            try:
                segments.append((time_var.get(), temp_var.get()))
            except:
                pass
        return sorted(segments, key=lambda x: x[0])
    
    def create_regions_tab(self):
        """Create the Regions tab for defining diffusion couple layers."""
        regions_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(regions_frame, text="Regions")
        
        # Instructions
        ttk.Label(regions_frame, text="Define the regions (layers) of your diffusion couple. "
                  "Each region has a width, phases, and initial composition.", 
                  wraplength=800).pack(anchor='w', pady=(0, 10))
        
        # === Batch Mode Section ===
        batch_frame = ttk.LabelFrame(regions_frame, text="Batch Mode (Excel Import)", padding="10")
        batch_frame.pack(fill='x', pady=(0, 10))
        
        # Toggle for batch mode
        batch_toggle_frame = ttk.Frame(batch_frame)
        batch_toggle_frame.pack(fill='x', pady=2)
        ttk.Checkbutton(batch_toggle_frame, text="Enable Batch Mode (import multiple alloys from Excel)", 
                        variable=self.batch_mode, command=self.toggle_batch_mode).pack(side='left')
        
        # Batch settings (initially hidden unless batch mode enabled)
        self.batch_settings_frame = ttk.Frame(batch_frame)
        
        # Excel file row
        excel_row = ttk.Frame(self.batch_settings_frame)
        excel_row.pack(fill='x', pady=2)
        ttk.Label(excel_row, text="Excel File:").pack(side='left', padx=5)
        ttk.Entry(excel_row, textvariable=self.batch_excel_path, width=50).pack(side='left', padx=5, fill='x', expand=True)
        ttk.Button(excel_row, text="Browse", command=self.browse_batch_excel).pack(side='left', padx=5)
        ttk.Button(excel_row, text="Preview", command=self.preview_batch_excel).pack(side='left', padx=5)
        
        # Settings row
        settings_row = ttk.Frame(self.batch_settings_frame)
        settings_row.pack(fill='x', pady=2)
        ttk.Label(settings_row, text="Output Dir:").pack(side='left', padx=5)
        ttk.Entry(settings_row, textvariable=self.batch_output_dir, width=30).pack(side='left', padx=5)
        ttk.Button(settings_row, text="Browse", command=self.browse_batch_output).pack(side='left', padx=5)
        
        # Status row
        self.batch_status_label = ttk.Label(self.batch_settings_frame, text="No file loaded", foreground='gray')
        self.batch_status_label.pack(anchor='w', pady=5)
        
        # Initially hide batch settings
        if not self.batch_mode.get():
            self.batch_settings_frame.pack_forget()
        else:
            self.batch_settings_frame.pack(fill='x', pady=5)
        
        # === Manual Mode Toolbar ===
        self.manual_toolbar = ttk.Frame(regions_frame)
        self.manual_toolbar.pack(fill='x', pady=(0, 10))
        
        ttk.Button(self.manual_toolbar, text="+ Add Region", command=self.add_region).pack(side='left', padx=5)
        ttk.Button(self.manual_toolbar, text="- Remove Selected", command=self.remove_region).pack(side='left', padx=5)
        ttk.Button(self.manual_toolbar, text="Add Template: Fe-C Couple", command=self.add_fe_c_template).pack(side='left', padx=20)
        
        # Scrollable container for regions
        canvas_frame = ttk.Frame(regions_frame)
        canvas_frame.pack(fill='both', expand=True)
        
        self.regions_canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical', command=self.regions_canvas.yview)
        self.scrollable_regions_frame = ttk.Frame(self.regions_canvas)
        
        self.scrollable_regions_frame.bind(
            "<Configure>",
            lambda e: self.regions_canvas.configure(scrollregion=self.regions_canvas.bbox("all"))
        )
        
        self.regions_canvas.create_window((0, 0), window=self.scrollable_regions_frame, anchor='nw')
        self.regions_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.regions_canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Add initial two regions with MMnS default compositions
        self.add_region(name="Left Region", width=51e-6)
        self.add_region(name="Right Region", width=48e-6)
        
        # Set default compositions for MMnS FCC/BCC couple
        if len(self.regions) >= 2:
            # Left region: FCC (Austenite)
            self.regions[0]['phase'].set("FCC_A1")
            for elem, val in [('C', 0.31), ('Mn', 12.95), ('Si', 1.22), 
                              ('Al', 2.42), ('Mo', 0.28), ('Nb', 1.17e-5), ('V', 0.003)]:
                if elem in self.regions[0]['comp_entries']:
                    self.regions[0]['comp_entries'][elem].set(val)
            
            # Right region: BCC (Ferrite)
            self.regions[1]['phase'].set("BCC_A2")
            for elem, val in [('C', 0.0016), ('Mn', 4.87), ('Si', 0.78), 
                              ('Al', 3.69), ('Mo', 0.62), ('Nb', 1.48e-5), ('V', 0.01)]:
                if elem in self.regions[1]['comp_entries']:
                    self.regions[1]['comp_entries'][elem].set(val)
        
    def add_region(self, name="New Region", width=1e-5):
        """Add a new region to the regions list."""
        region_idx = len(self.regions)
        
        # Create region frame
        region_frame = ttk.LabelFrame(self.scrollable_regions_frame, 
                                       text=f"Region {region_idx + 1}: {name}", 
                                       padding="10")
        region_frame.pack(fill='x', pady=5, padx=5)
        
        # Store region data
        region_data = {
            'frame': region_frame,
            'name': tk.StringVar(value=name),
            'width': tk.DoubleVar(value=width),
            'phase': tk.StringVar(value="FCC_A1"),
            'compositions': {},  # Element -> composition value
            'grid_type': tk.StringVar(value="Linear"),
            'grid_points': tk.IntVar(value=50),
        }
        
        # Region name
        row = 0
        ttk.Label(region_frame, text="Name:").grid(row=row, column=0, sticky='w', pady=2)
        name_entry = ttk.Entry(region_frame, textvariable=region_data['name'], width=20)
        name_entry.grid(row=row, column=1, sticky='w', padx=5, pady=2)
        
        # Update frame title when name changes
        def update_title(*args):
            region_frame.config(text=f"Region {region_idx + 1}: {region_data['name'].get()}")
        region_data['name'].trace('w', update_title)
        
        # Width
        row += 1
        ttk.Label(region_frame, text="Width (m):").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(region_frame, textvariable=region_data['width'], width=15).grid(row=row, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(region_frame, text=f"= {width*1e6:.1f} µm").grid(row=row, column=2, sticky='w', pady=2)
        
        # Phase
        row += 1
        ttk.Label(region_frame, text="Phase:").grid(row=row, column=0, sticky='w', pady=2)
        # Only FCC_A1 and BCC_A2 have mobility data in MOBFE8
        # Using only these phases avoids "DIFFUSION NONE" entries for 100+ other phases
        available_phases = ["FCC_A1", "BCC_A2"]
        ttk.Combobox(region_frame, textvariable=region_data['phase'], 
                     values=available_phases, 
                     width=15).grid(row=row, column=1, sticky='w', padx=5, pady=2)
        
        # Grid type and points
        row += 1
        ttk.Label(region_frame, text="Grid Type:").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Combobox(region_frame, textvariable=region_data['grid_type'], 
                     values=["Linear", "Geometric"],
                     state='readonly', width=12).grid(row=row, column=1, sticky='w', padx=5, pady=2)
        
        # Grid points entry (manual input)
        ttk.Label(region_frame, text="Points:").grid(row=row, column=2, sticky='w', padx=(10, 2), pady=2)
        ttk.Entry(region_frame, textvariable=region_data['grid_points'], width=6).grid(row=row, column=3, sticky='w', pady=2)
        
        # Update grid_type default to just "Linear"
        region_data['grid_type'].set("Linear")
        
        # Compositions header
        row += 1
        ttk.Label(region_frame, text="Initial Compositions (wt%):", font=('Helvetica', 9, 'bold')).grid(
            row=row, column=0, columnspan=4, sticky='w', pady=(10, 5))
        
        # Composition entries for selected elements
        row += 1
        comp_frame = ttk.Frame(region_frame)
        comp_frame.grid(row=row, column=0, columnspan=4, sticky='w')
        
        # Store composition entries
        region_data['comp_frame'] = comp_frame
        region_data['comp_entries'] = {}
        
        # Add default Fe-C compositions
        # Get currently selected elements
        selected = self.get_selected_elements()
        self.update_region_compositions(region_data, selected)
        
        self.regions.append(region_data)
    
    def get_selected_elements(self):
        """Get list of selected elements with Fe first."""
        elements = [elem for elem, var in self.element_vars.items() if var.get()]
        # Ensure Fe is first (it's the balance element)
        if 'Fe' in elements:
            elements.remove('Fe')
            elements.insert(0, 'Fe')
        return elements
    
    def sync_elements_to_regions(self):
        """Sync element selection changes to all regions."""
        selected = self.get_selected_elements()
        for region_data in self.regions:
            # Preserve existing composition values where possible
            old_values = {k: v.get() for k, v in region_data.get('comp_entries', {}).items()}
            self.update_region_compositions(region_data, selected, old_values)
        
    def update_region_compositions(self, region_data, elements, preserve_values=None):
        """Update composition entry fields for a region with Fe as balance."""
        if preserve_values is None:
            preserve_values = {}
            
        comp_frame = region_data['comp_frame']
        
        # Clear existing
        for widget in comp_frame.winfo_children():
            widget.destroy()
        region_data['comp_entries'] = {}
        
        # Fe label (balance, read-only display)
        if 'Fe' in elements:
            ttk.Label(comp_frame, text="Fe (bal):").grid(row=0, column=0, padx=(0, 2))
            fe_var = tk.DoubleVar(value=preserve_values.get('Fe', 99.0))
            fe_label = ttk.Label(comp_frame, textvariable=fe_var, width=8, relief='sunken', anchor='e')
            fe_label.grid(row=0, column=1, padx=(0, 10))
            region_data['comp_entries']['Fe'] = fe_var
            region_data['fe_label'] = fe_label
            col_offset = 2
        else:
            col_offset = 0
        
        # Other elements (editable) - skip Fe
        other_elements = [e for e in elements if e != 'Fe']
        for i, elem in enumerate(other_elements):
            ttk.Label(comp_frame, text=f"{elem}:").grid(row=0, column=col_offset + i*2, padx=(10, 2))
            default_val = preserve_values.get(elem, 0.0)
            var = tk.DoubleVar(value=default_val)
            entry = ttk.Entry(comp_frame, textvariable=var, width=8)
            entry.grid(row=0, column=col_offset + i*2+1, padx=(0, 10))
            region_data['comp_entries'][elem] = var
            
            # Add trace to update Fe balance when any element changes
            var.trace('w', lambda *args, rd=region_data: self.update_fe_balance(rd))
        
        # Calculate initial Fe balance
        self.update_fe_balance(region_data)
    
    def update_fe_balance(self, region_data):
        """Update Fe to be 100 - sum of other elements."""
        if 'Fe' not in region_data['comp_entries']:
            return
        try:
            total_others = sum(
                var.get() for elem, var in region_data['comp_entries'].items() 
                if elem != 'Fe'
            )
            fe_balance = max(0.0, 100.0 - total_others)
            region_data['comp_entries']['Fe'].set(round(fe_balance, 2))
        except:
            pass  # Ignore errors during typing
            
    def remove_region(self):
        """Remove the last region."""
        if len(self.regions) > 1:
            region = self.regions.pop()
            region['frame'].destroy()
        else:
            messagebox.showwarning("Cannot Remove", "At least one region is required.")
            
    def add_fe_c_template(self):
        """Add a template MMnS FCC/BCC diffusion couple."""
        # Clear existing regions
        while self.regions:
            region = self.regions.pop()
            region['frame'].destroy()
            
        # Add two regions for FCC/BCC diffusion couple (MMnS steel)
        self.add_region(name="Left Region", width=51e-6)
        self.add_region(name="Right Region", width=48e-6)
        
        # Set compositions for MMnS diffusion simulation
        if len(self.regions) >= 2:
            # Left region: FCC (Austenite) - higher Mn
            self.regions[0]['phase'].set("FCC_A1")
            for elem, val in [('C', 0.31), ('Mn', 12.95), ('Si', 1.22), 
                              ('Al', 2.42), ('Mo', 0.28), ('Nb', 1.17e-5), ('V', 0.003)]:
                if elem in self.regions[0]['comp_entries']:
                    self.regions[0]['comp_entries'][elem].set(val)
            
            # Right region: BCC (Ferrite) - lower Mn
            self.regions[1]['phase'].set("BCC_A2")
            for elem, val in [('C', 0.0016), ('Mn', 4.87), ('Si', 0.78), 
                              ('Al', 3.69), ('Mo', 0.62), ('Nb', 1.48e-5), ('V', 0.01)]:
                if elem in self.regions[1]['comp_entries']:
                    self.regions[1]['comp_entries'][elem].set(val)
            
        messagebox.showinfo("Template Applied", 
            "MMnS FCC/BCC diffusion couple applied.\n\n"
            "Left: FCC_A1 (Austenite) - high Mn\n"
            "Right: BCC_A2 (Ferrite) - low Mn\n"
            "Temperature: 1173 K (900°C)")
    
    def toggle_batch_mode(self):
        """Toggle between manual and batch mode."""
        if self.batch_mode.get():
            # Show batch settings, hide manual regions
            self.batch_settings_frame.pack(fill='x', pady=5)
            self.manual_toolbar.pack_forget()
            self.regions_canvas.pack_forget()
        else:
            # Hide batch settings, show manual regions
            self.batch_settings_frame.pack_forget()
            self.manual_toolbar.pack(fill='x', pady=(0, 10))
            self.regions_canvas.pack(side='left', fill='both', expand=True)
    
    def browse_batch_excel(self):
        """Browse for batch Excel file."""
        filepath = filedialog.askopenfilename(
            title="Select Batch Alloys Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.batch_excel_path.set(filepath)
            self.preview_batch_excel()
    
    def browse_batch_output(self):
        """Browse for output directory."""
        dirpath = filedialog.askdirectory(title="Select Output Directory")
        if dirpath:
            self.batch_output_dir.set(dirpath)
    
    def preview_batch_excel(self):
        """Preview and parse batch Excel file."""
        filepath = self.batch_excel_path.get()
        if not filepath or not os.path.exists(filepath):
            messagebox.showwarning("File Error", "Please select a valid Excel file.")
            return
        
        try:
            
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Parse the alloy configurations
            alloys, errors = self.parse_dictra_excel(df)
            
            self.batch_alloys = alloys
            
            # Update status
            if alloys:
                status = f"Loaded {len(alloys)} alloys ({len(df)} rows)"
                if errors:
                    status += f" - {len(errors)} warnings"
                self.batch_status_label.config(text=status, foreground='green')
                
                # Show preview
                msg = f"File: {os.path.basename(filepath)}\n"
                msg += f"Total alloys: {len(alloys)}\n\n"
                msg += "Alloys:\n"
                for i, alloy in enumerate(alloys[:10]):  # Show first 10
                    temps = [f"{t}°C/{time}s" for t, time in alloy.get('heat_treatment', [])]
                    msg += f"  {alloy['name']}: {alloy['regions'][0]['phase']}/{alloy['regions'][1]['phase']}, {', '.join(temps)}\n"
                if len(alloys) > 10:
                    msg += f"  ... and {len(alloys) - 10} more\n"
                
                if errors:
                    msg += f"\nWarnings:\n" + "\n".join(errors[:5])
                
                self.write_log(msg)
                messagebox.showinfo("Batch Preview", msg)
            else:
                self.batch_status_label.config(text="No valid alloys found", foreground='red')
                messagebox.showerror("Parse Error", f"No valid alloys found.\n\nErrors:\n" + "\n".join(errors[:10]))
                
        except Exception as e:
            self.batch_status_label.config(text=f"Error: {e}", foreground='red')
            messagebox.showerror("Excel Error", f"Error reading file:\n{str(e)}")
    
    def parse_dictra_excel(self, df):
        """Parse Excel dataframe into alloy configurations.
        
        Expected columns (case-insensitive):
        - Alloy: alloy name (rows with same name form one diffusion case)
        - Region: region number or text like 'region 1', 'region 2'
        - Phase: phase name (FCC_A1, BCC_A2)
        - Width_um: region width in micrometers
        - Grid_type: linear or geometric
        - Points: number of grid points
        - C, Mn, Si, Al, Mo, Nb, V, W, ...: element compositions (wt%)
        - Temp1_C, Time1_s, Temp2_C, Time2_s, ...: heat treatment steps
        """
        
        alloys = {}
        errors = []
        
        # Normalize column names to lowercase for case-insensitive matching
        df.columns = [str(c).strip() for c in df.columns]
        col_map = {c.lower(): c for c in df.columns}  # lowercase -> original
        
        # Helper to get column value case-insensitively
        def get_col(row, name, default=None):
            orig_col = col_map.get(name.lower())
            if orig_col:
                return row.get(orig_col, default)
            return default
        
        # Check required columns (case-insensitive)
        required = ['alloy', 'region', 'phase', 'width_um']
        missing = [c for c in required if c not in col_map]
        if missing:
            errors.append(f"Missing required columns: {missing}")
            return [], errors
        
        # Element columns (case-insensitive)
        element_names = ['C', 'Mn', 'Si', 'Al', 'Mo', 'Nb', 'V', 'W', 'Cr', 'Ni', 'Co', 'Ti', 'Cu', 'B', 'N', 'P', 'S']
        element_cols = [(e, col_map.get(e.lower())) for e in element_names if e.lower() in col_map]
        
        # Heat treatment columns (case-insensitive, e.g., temp1_c, time1_s)
        temp_cols = []
        time_cols = []
        for c in df.columns:
            c_lower = c.lower()
            if c_lower.startswith('temp') and c_lower.endswith('_c'):
                temp_cols.append(c)
            elif c_lower.startswith('time') and c_lower.endswith('_s'):
                time_cols.append(c)
        temp_cols.sort()
        time_cols.sort()
        
        # Helper to parse region number from text like 'region 1' or just '1'
        def parse_region_num(val):
            if pd.isna(val):
                return 1
            val_str = str(val).lower().strip()
            # Try to extract number from text like 'region 1' or 'region 2'
            match = re.search(r'(\d+)', val_str)
            if match:
                return int(match.group(1))
            return int(val)
        
        # Group by alloy name
        alloy_col = col_map.get('alloy')
        for alloy_name, group in df.groupby(alloy_col):
            try:
                regions = []
                heat_treatment = []
                
                for _, row in group.iterrows():
                    region_num = parse_region_num(get_col(row, 'region', 1))
                    
                    # Parse compositions
                    compositions = {}
                    total = 0
                    for elem_name, orig_col in element_cols:
                        val = row.get(orig_col, 0)
                        if pd.notna(val) and float(val) > 0:
                            compositions[elem_name] = float(val)
                            total += float(val)
                    
                    # Fe is balance
                    compositions['Fe'] = 100.0 - total
                    
                    # Get grid settings (case-insensitive)
                    grid_type = str(get_col(row, 'grid_type', 'linear'))
                    points = int(get_col(row, 'points', 50)) if pd.notna(get_col(row, 'points')) else 50
                    
                    region_config = {
                        'region_num': region_num,
                        'phase': str(get_col(row, 'phase', 'FCC_A1')),
                        'width_um': float(get_col(row, 'width_um', 50)),
                        'grid_type': grid_type,
                        'points': points,
                        'compositions': compositions
                    }
                    regions.append(region_config)
                    
                    # Parse heat treatment (only from first row of each alloy)
                    if len(heat_treatment) == 0:
                        for temp_col, time_col in zip(temp_cols, time_cols):
                            temp = row.get(temp_col)
                            time_val = row.get(time_col)
                            if pd.notna(temp) and pd.notna(time_val) and float(temp) > 0 and float(time_val) >= 0:
                                heat_treatment.append((float(temp), float(time_val)))
                
                # Sort regions by region number
                regions.sort(key=lambda r: r['region_num'])
                
                if len(regions) >= 2:
                    alloys[alloy_name] = {
                        'name': str(alloy_name),
                        'regions': regions,
                        'heat_treatment': heat_treatment
                    }
                else:
                    errors.append(f"Alloy '{alloy_name}': needs 2 regions, found {len(regions)}")
                    
            except Exception as e:
                errors.append(f"Alloy '{alloy_name}': {e}")
        
        return list(alloys.values()), errors
            
    def create_boundary_tab(self):
        """Create the Boundary Conditions tab."""
        boundary_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(boundary_frame, text="Boundary Conditions")
        
        # Left boundary
        left_frame = ttk.LabelFrame(boundary_frame, text="Left Boundary (x = 0)", padding="10")
        left_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(left_frame, text="Boundary Type:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Combobox(left_frame, textvariable=self.left_boundary_type,
                     values=["Closed System", "Fixed Composition", "Activity (Carburizing/Nitriding)"],
                     state='readonly', width=35).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(left_frame, text="Closed System: No flux across boundary (default)\n"
                  "Fixed Composition: Maintain constant composition\n"
                  "Activity: Set activity for surface reactions (e.g., carburizing)",
                  foreground='gray').grid(row=1, column=0, columnspan=2, sticky='w', pady=5)
        
        # Right boundary
        right_frame = ttk.LabelFrame(boundary_frame, text="Right Boundary (x = L)", padding="10")
        right_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(right_frame, text="Boundary Type:").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Combobox(right_frame, textvariable=self.right_boundary_type,
                     values=["Closed System", "Fixed Composition", "Activity (Carburizing/Nitriding)"],
                     state='readonly', width=35).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(right_frame, text="For most diffusion couples, use 'Closed System' on both boundaries.",
                  foreground='gray').grid(row=1, column=0, columnspan=2, sticky='w', pady=5)
        
    def create_advanced_tab(self):
        """Create the Advanced settings tab."""
        advanced_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(advanced_frame, text="Advanced")
        
        # Solver settings
        solver_frame = ttk.LabelFrame(advanced_frame, text="Solver Settings", padding="10")
        solver_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(solver_frame, text="Solver Type:").grid(row=0, column=0, sticky='w', pady=5)
        solver_combo = ttk.Combobox(solver_frame, textvariable=self.solver_type,
                     values=["Automatic", "Classic", "Homogenization"],
                     state='readonly', width=20)
        solver_combo.grid(row=0, column=1, sticky='w', padx=5, pady=5)
        solver_combo.bind('<<ComboboxSelected>>', lambda e: self.toggle_homogenization_options())
        
        ttk.Label(solver_frame, text="Automatic: Uses Homogenization if multi-phase, otherwise Classic (recommended)",
                  foreground='gray').grid(row=1, column=0, columnspan=2, sticky='w', pady=2)
        
        # === Homogenization Model Specific Frame ===
        self.homogenization_frame = ttk.LabelFrame(advanced_frame, text="Homogenization Model Specific", padding="10")
        # Initially hidden - will be shown when Homogenization is selected
        
        # Homogenization function dropdown
        ttk.Label(self.homogenization_frame, text="Homogenization function:").grid(row=0, column=0, sticky='w', pady=5)
        homog_functions = [
            "Rule of mixtures (upper Wiener bound)",
            "Inverse rule of mixtures (lower Wiener bound)",
            "General lower Hashin-Shtrikman bound",
            "General upper Hashin-Shtrikman bound",
            "Hashin-Shtrikman bound with majority phase as matrix phase"
        ]
        ttk.Combobox(self.homogenization_frame, textvariable=self.homogenization_function,
                     values=homog_functions, state='readonly', width=50).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        # Global minimization checkbox
        ttk.Checkbutton(self.homogenization_frame, text="Use global minimization",
                        variable=self.use_global_minimization).grid(row=1, column=0, columnspan=2, sticky='w', pady=5)
        
        # Interpolation Scheme section
        interp_frame = ttk.LabelFrame(self.homogenization_frame, text="Interpolation Scheme", padding="10")
        interp_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(10, 0))
        
        # Use interpolation scheme checkbox
        interp_check = ttk.Checkbutton(interp_frame, text="Use interpolation scheme",
                        variable=self.use_interpolation_scheme,
                        command=self.toggle_interpolation_options)
        interp_check.grid(row=0, column=0, columnspan=4, sticky='w', pady=5)
        
        # Interpolation settings row (stored for enabling/disabling)
        self.interp_settings_frame = ttk.Frame(interp_frame)
        self.interp_settings_frame.grid(row=1, column=0, columnspan=4, sticky='w', pady=5)
        
        ttk.Combobox(self.interp_settings_frame, textvariable=self.interpolation_type,
                     values=["Logarithmic", "Linear"], state='readonly', width=12).pack(side='left', padx=(0, 5))
        ttk.Label(self.interp_settings_frame, text="discretization with").pack(side='left', padx=5)
        ttk.Entry(self.interp_settings_frame, textvariable=self.interpolation_steps, width=8).pack(side='left', padx=5)
        ttk.Label(self.interp_settings_frame, text="steps in each dimension").pack(side='left', padx=5)
        
        # Memory row
        self.memory_frame = ttk.Frame(interp_frame)
        self.memory_frame.grid(row=2, column=0, columnspan=4, sticky='w', pady=5)
        
        ttk.Label(self.memory_frame, text="Memory to use:").pack(side='left', padx=(0, 5))
        ttk.Entry(self.memory_frame, textvariable=self.interpolation_memory, width=10).pack(side='left', padx=5)
        ttk.Combobox(self.memory_frame, textvariable=self.interpolation_memory_unit,
                     values=["Megabyte", "Gigabyte"], state='readonly', width=10).pack(side='left', padx=5)
        
        # Timestep control
        ts_frame = ttk.LabelFrame(advanced_frame, text="Timestep Control", padding="10")
        ts_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(ts_frame, text="Min Timestep (s):").grid(row=0, column=0, sticky='w', pady=5)
        ttk.Entry(ts_frame, textvariable=self.min_timestep, width=15).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(ts_frame, text="Max Timestep (s):").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(ts_frame, textvariable=self.max_timestep, width=15).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(ts_frame, text="Increase Factor:").grid(row=2, column=0, sticky='w', pady=5)
        ttk.Entry(ts_frame, textvariable=self.timestep_increase_factor, width=15).grid(row=2, column=1, sticky='w', padx=5, pady=5)
        
        # Output options
        output_frame = ttk.LabelFrame(advanced_frame, text="Output Options", padding="10")
        output_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Checkbutton(output_frame, text="Skip phase fraction extraction (faster, compositions only)",
                        variable=self.skip_phase_fractions).grid(row=0, column=0, sticky='w', pady=5)
    
    def toggle_homogenization_options(self):
        """Show/hide homogenization options based on solver type."""
        if self.solver_type.get() == "Homogenization":
            self.homogenization_frame.pack(fill='x', pady=(0, 10), after=self.homogenization_frame.master.winfo_children()[0])
        else:
            self.homogenization_frame.pack_forget()
    
    def toggle_interpolation_options(self):
        """Enable/disable interpolation settings based on checkbox."""
        state = 'normal' if self.use_interpolation_scheme.get() else 'disabled'
        for child in self.interp_settings_frame.winfo_children():
            if hasattr(child, 'configure'):
                try:
                    child.configure(state=state)
                except:
                    pass
        for child in self.memory_frame.winfo_children():
            if hasattr(child, 'configure'):
                try:
                    child.configure(state=state)
                except:
                    pass
        
    def create_results_tab(self):
        """Create the Results tab for visualization."""
        results_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(results_frame, text="Results")
        
        # Placeholder for results
        ttk.Label(results_frame, text="Results will appear here after calculation.", 
                  font=('Helvetica', 12)).pack(pady=50)
        
        # Store reference for later updates
        self.results_frame = results_frame
        self.results_canvas = None
        
    def create_control_frame(self, parent):
        """Create the bottom control frame with run/stop buttons."""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', pady=5)
        
        # Left side: Run controls
        left_controls = ttk.Frame(control_frame)
        left_controls.pack(side='left')
        
        self.run_btn = ttk.Button(left_controls, text="▶ Run Calculation", command=self.run_calculation)
        self.run_btn.pack(side='left', padx=5)
        
        self.stop_btn = ttk.Button(left_controls, text="■ Stop", command=self.stop_calculation, state='disabled')
        self.stop_btn.pack(side='left', padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate', length=300)
        self.progress.pack(side='left', padx=20)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side='left', padx=10)
        
        # Right side: Log toggle
        ttk.Button(control_frame, text="Show Log", command=self.toggle_log).pack(side='right', padx=5)
        
    def toggle_log(self):
        """Toggle the log window."""
        if hasattr(self, 'log_window') and self.log_window.winfo_exists():
            self.log_window.destroy()
        else:
            self.log_window = tk.Toplevel(self.root)
            self.log_window.title("Log")
            self.log_window.geometry("600x400")
            
            self.log_text = tk.Text(self.log_window, wrap='word')
            self.log_text.pack(fill='both', expand=True)
            
            scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
            self.log_text.config(yscrollcommand=scrollbar.set)
            scrollbar.pack(side='right', fill='y')
            
    def write_log(self, message):
        """Write a message to the log."""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        print(log_message.strip())  # Also print to console
        
        if hasattr(self, 'log_text') and self.log_text.winfo_exists():
            self.log_text.insert('end', log_message)
            self.log_text.see('end')
            
    def browse_file(self, var, filetypes):
        """Browse for a file."""
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            var.set(filename)
            
    def browse_directory(self, var):
        """Browse for a directory."""
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)
            
    def new_setup(self):
        """Reset to a new setup."""
        if messagebox.askyesno("New Setup", "Clear current setup and start fresh?"):
            # Clear regions
            while self.regions:
                region = self.regions.pop()
                region['frame'].destroy()
            # Add default regions
            self.add_region(name="Left Region", width=5e-5)
            self.add_region(name="Right Region", width=5e-5)
            
    def load_setup(self):
        """Load setup from a JSON file."""
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                # Note: Setup application from loaded data not yet implemented
                self.write_log(f"Loaded setup from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load setup: {e}")
                
    def save_setup(self):
        """Save setup to a JSON file."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            try:
                data = self.collect_setup_data()
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                self.write_log(f"Saved setup to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save setup: {e}")
                
    def collect_setup_data(self):
        """Collect all setup data into a dictionary."""
        return {
            'tdb': self.tdb_path.get(),
            'mdb': self.mdb_path.get(),
            'temperature': self.temperature.get(),
            'simulation_time': self.simulation_time.get(),
            'geometry': self.geometry_type.get(),
            'calc_type': self.calc_type.get(),
            'solver': self.solver_type.get(),
            'regions': [
                {
                    'name': r['name'].get(),
                    'width': r['width'].get(),
                    'phase': r['phase'].get(),
                    'compositions': {k: v.get() for k, v in r['comp_entries'].items()}
                }
                for r in self.regions
            ],
            'left_boundary': self.left_boundary_type.get(),
            'right_boundary': self.right_boundary_type.get(),
            # Homogenization settings
            'homogenization_function': self.homogenization_function.get(),
            'use_global_minimization': self.use_global_minimization.get(),
            'use_interpolation_scheme': self.use_interpolation_scheme.get(),
            'interpolation_type': self.interpolation_type.get(),
            'interpolation_steps': self.interpolation_steps.get(),
            'interpolation_memory': self.interpolation_memory.get(),
            'interpolation_memory_unit': self.interpolation_memory_unit.get(),
        }
        
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo("About", 
            "DICTRA Diffusion Calculator\n\n"
            "A GUI for TC-Python DICTRA diffusion simulations.\n\n"
            "Based on TC-Python 2025b")
        
    def run_calculation(self):
        """Start the diffusion calculation."""
        if not TC_PYTHON_AVAILABLE:
            messagebox.showerror("Error", "TC-Python is not available. Please install TC-Python.")
            return
        
        # Check for batch mode
        if self.batch_mode.get():
            self.run_batch_calculations()
            return
            
        # Validate inputs
        if not self.tdb_path.get() or not self.mdb_path.get():
            messagebox.showerror("Error", "Please select both TDB and MDB database files.")
            return
            
        if len(self.regions) < 1:
            messagebox.showerror("Error", "Please define at least one region.")
            return
        
        # Check for phase compatibility (warn if different phases at interfaces)
        if len(self.regions) >= 2:
            phases = [r['phase'].get() for r in self.regions]
            unique_phases = set(phases)
            if len(unique_phases) > 1:
                result = messagebox.askyesno("Phase Warning", 
                    f"Regions use different phases: {unique_phases}\n\n"
                    "Different phases at interfaces can cause 'ERROR UPDATING GRID' "
                    "if they are not thermodynamically stable together.\n\n"
                    "Recommended: Use same phase (e.g., FCC_A1) for all regions.\n\n"
                    "Continue anyway?"
                )
                if not result:
                    return
            
        # Update UI
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress.start(10)
        self.status_label.config(text="Running calculation...")
        self.calculation_running = True
        self.stop_requested = False
        
        # Run in thread
        self.calc_thread = threading.Thread(target=self.run_calculation_thread)
        self.calc_thread.start()
        
    def run_calculation_thread(self):
        """Run the calculation in a separate thread."""
        try:
            self.write_log("Starting diffusion calculation...")
            
            # Get selected elements
            elements = [elem for elem, var in self.element_vars.items() if var.get()]
            self.write_log(f"Elements: {elements}")
            
            # Initialize TC-Python
            with TCPython() as session:
                self.write_log("TC-Python session initialized.")
                
                # Load databases
                session = session.set_cache_folder(self.cache_path.get())
                
                # Select databases with ONLY FCC_A1 and BCC_A2 phases
                # This prevents 100+ phases from being loaded as "DIFFUSION NONE"
                self.write_log(f"Loading TDB: {self.tdb_path.get()}")
                self.write_log(f"Loading MDB: {self.mdb_path.get()}")
                self.write_log("Selecting only FCC_A1 and BCC_A2 phases (have mobility data)...")
                
                # Use without_default_phases() and get_system_for() to control phases
                system = (
                    session
                    .select_thermodynamic_and_kinetic_databases_with_elements(
                        self.tdb_path.get(),
                        self.mdb_path.get(),
                        elements
                    )
                    .without_default_phases()
                    .select_phase("FCC_A1")
                    .select_phase("BCC_A2")
                    .get_system()
                )
                
                self.write_log("Databases loaded with FCC_A1 and BCC_A2 only. Setting up diffusion calculation...")
                
                # Check calculation type
                if self.calc_type.get() == "Isothermal":
                    # Create isothermal calculation
                    self.write_log("Creating isothermal diffusion calculation...")
                    calc = system.with_isothermal_diffusion_calculation()
                    
                    # Set temperature
                    calc = calc.set_temperature(self.temperature.get())
                    self.write_log(f"Temperature: {self.temperature.get()} K ({self.temperature.get()-273.15:.1f} °C)")
                    
                    # Parse simulation times (space-separated)
                    time_str = self.simulation_time.get()
                    sim_times = [float(t.strip()) for t in time_str.split() if t.strip()]
                    if not sim_times:
                        sim_times = [3600.0]
                    sim_times.sort()
                    max_time = max(sim_times)
                    
                else:
                    # Create non-isothermal calculation
                    self.write_log("Creating non-isothermal diffusion calculation...")
                    calc = system.with_non_isothermal_diffusion_calculation()
                    
                    # Debug: Log available methods on calc object
                    available_methods = [m for m in dir(calc) if not m.startswith('_') and 'temp' in m.lower()]
                    self.write_log(f"Available temperature-related methods: {available_methods}")
                    
                    # Get thermal profile from GUI
                    thermal_profile = self.get_thermal_profile()
                    if len(thermal_profile) < 2:
                        raise ValueError("Non-isothermal mode requires at least 2 temperature points")
                    
                    # Ensure profile starts at t=0
                    if thermal_profile[0][0] != 0:
                        initial_temp_K = thermal_profile[0][1]
                        thermal_profile.insert(0, (0, initial_temp_K))
                        self.write_log(f"Added t=0 point with initial temp {initial_temp_K}K")
                    
                    self.write_log(f"Thermal profile: {len(thermal_profile)} points")
                    for t, temp in thermal_profile:
                        self.write_log(f"  t={t}s: {temp}K ({temp-273.15:.1f}°C)")
                    
                    # Create TemperatureProfile object
                    temp_profile = TemperatureProfile()
                    
                    # Add all time-temperature points
                    for t, T in thermal_profile:
                        temp_profile = temp_profile.add_time_temperature(t, T)
                    
                    # Set initial temperature if the method exists (older API compatibility)
                    initial_temp_K = thermal_profile[0][1]
                    if hasattr(calc, 'set_initial_temperature'):
                        calc = calc.set_initial_temperature(initial_temp_K)
                        self.write_log(f"Set initial temperature to {initial_temp_K}K")
                    
                    # Apply temperature profile
                    calc = calc.with_temperature_profile(temp_profile)
                    self.write_log("Successfully set temperature profile!")
                    
                    # Get simulation times from thermal profile
                    sim_times = [thermal_profile[-1][0]]  # End time from last point
                    max_time = thermal_profile[-1][0]
                
                self.write_log(f"Simulation times: {sim_times} seconds")
                
                # Set simulation time to maximum
                calc = calc.set_simulation_time(max_time)
                
                # Set geometry
                if self.geometry_type.get() == "Planar":
                    calc = calc.with_planar_geometry()
                elif self.geometry_type.get() == "Cylindrical":
                    calc = calc.with_cylindrical_geometry()
                elif self.geometry_type.get() == "Spherical":
                    calc = calc.with_spherical_geometry()
                
                # Apply homogenization model settings if selected
                if self.solver_type.get() == "Homogenization":
                    self.write_log("Applying homogenization model settings...")
                    
                    # Try to use with_solver method with a HomogenizationSolver or DiffusionSolver
                    try:
                        # Use HomogenizationSolver.homogenization() factory to create solver
                        solver = HomogenizationSolver.homogenization()
                        
                        # Log available HomogenizationFunctions methods
                        func_methods = [m for m in dir(HomogenizationFunctions) if not m.startswith('_')]
                        self.write_log(f"  HomogenizationFunctions methods: {func_methods}")
                        
                        # Map GUI selection to HomogenizationFunctions factory method
                        selected_func = self.homogenization_function.get()
                        self.write_log(f"  Selected function: {selected_func}")
                        
                        # Get the appropriate function using factory methods
                        homog_func = None
                        if "Rule of mixtures" in selected_func:
                            if hasattr(HomogenizationFunctions, 'rule_of_mixtures'):
                                homog_func = HomogenizationFunctions.rule_of_mixtures()
                            elif hasattr(HomogenizationFunctions, 'upper_wiener'):
                                homog_func = HomogenizationFunctions.upper_wiener()
                        elif "Inverse rule" in selected_func:
                            if hasattr(HomogenizationFunctions, 'inverse_rule_of_mixtures'):
                                homog_func = HomogenizationFunctions.inverse_rule_of_mixtures()
                            elif hasattr(HomogenizationFunctions, 'lower_wiener'):
                                homog_func = HomogenizationFunctions.lower_wiener()
                        elif "lower Hashin" in selected_func:
                            if hasattr(HomogenizationFunctions, 'lower_hashin_shtrikman'):
                                homog_func = HomogenizationFunctions.lower_hashin_shtrikman()
                        elif "upper Hashin" in selected_func:
                            if hasattr(HomogenizationFunctions, 'upper_hashin_shtrikman'):
                                homog_func = HomogenizationFunctions.upper_hashin_shtrikman()
                        elif "majority phase" in selected_func:
                            if hasattr(HomogenizationFunctions, 'hashin_shtrikman_majority_phase'):
                                homog_func = HomogenizationFunctions.hashin_shtrikman_majority_phase()
                        
                        if homog_func:
                            solver = solver.with_function(homog_func)
                            self.write_log(f"  Applied homogenization function: {selected_func}")
                        else:
                            self.write_log(f"  Warning: Could not find factory method for {selected_func}")
                        
                        # Apply global minimization setting
                        if self.use_global_minimization.get():
                            solver = solver.enable_global_minimization()
                            self.write_log("  Enabled global minimization")
                        
                        # Apply interpolation scheme settings
                        if self.use_interpolation_scheme.get():
                            interp_type = self.interpolation_type.get()
                            steps = self.interpolation_steps.get()
                            
                            if interp_type == "Logarithmic":
                                solver = solver.with_logarithmic_interpolation_scheme(steps)
                                self.write_log(f"  Set logarithmic interpolation with {steps} steps")
                            else:
                                solver = solver.with_linear_interpolation_scheme(steps)
                                self.write_log(f"  Set linear interpolation with {steps} steps")
                            
                            # Set memory limit
                            memory = self.interpolation_memory.get()
                            memory_unit = self.interpolation_memory_unit.get()
                            if memory_unit == "Gigabyte":
                                memory_mb = memory * 1024
                            else:
                                memory_mb = memory
                            solver = solver.set_memory_to_use(memory_mb)
                            self.write_log(f"  Set memory: {memory} {memory_unit}")
                        else:
                            solver = solver.disable_interpolation_scheme()
                            self.write_log("  Disabled interpolation scheme")
                        
                        # Apply solver to calculation
                        calc = calc.with_solver(solver)
                        self.write_log("  Applied configured HomogenizationSolver")
                        
                    except Exception as e:
                        self.write_log(f"  Error with solver setup: {e}")
                        self.write_log(traceback.format_exc())
                
                # Add regions
                for i, region_data in enumerate(self.regions):
                    self.write_log(f"Adding region: {region_data['name'].get()}")
                    
                    # Create region
                    region = Region(region_data['name'].get())
                    region = region.set_width(region_data['width'].get())
                    
                    # Set phase - Each region has ONE phase
                    # Homogenization model works with multiple regions, each having one phase
                    # Volume fractions are determined by region widths
                    region = region.add_phase(region_data['phase'].get())
                    self.write_log(f"  Phase: {region_data['phase'].get()}")
                    
                    # Set grid - use CalculatedGrid factory methods
                    grid_type_str = region_data['grid_type'].get()
                    grid_points = region_data['grid_points'].get()  # Direct entry from user
                    
                    self.write_log(f"  Grid: {grid_type_str} with {grid_points} points")
                    
                    if "Geometric" in grid_type_str:
                        # Geometric grid with factor 1.2 for fine resolution at interface
                        grid = CalculatedGrid.geometric(grid_points, 1.2)
                    else:
                        # Default to linear grid (equally spaced)
                        grid = CalculatedGrid.linear(grid_points)
                    
                    region = region.with_grid(grid)
                    
                    # Set composition profile (unit is set on CompositionProfile, not ConstantProfile)
                    profile = CompositionProfile(Unit.MASS_PERCENT)
                    for elem, var in region_data['comp_entries'].items():
                        if elem != 'Fe':  # Fe is the dependent element (balance)
                            profile = profile.add(elem, ConstantProfile(var.get()))
                    
                    region = region.with_composition_profile(profile)
                    
                    # Add region to calculation
                    calc = calc.add_region(region)
                
                # Set boundary conditions
                left_bc = BoundaryCondition.closed_system()
                right_bc = BoundaryCondition.closed_system()
                
                calc = calc.with_left_boundary_condition(left_bc)
                calc = calc.with_right_boundary_condition(right_bc)
                
                # Apply timestep control settings
                try:
                    
                    min_ts = self.min_timestep.get()
                    max_ts = self.max_timestep.get()
                    increase = self.timestep_increase_factor.get()
                    
                    self.write_log(f"Timestep control: min={min_ts}s, max={max_ts}s, factor={increase}")
                    
                    ts_control = TimestepControl()
                    if hasattr(ts_control, 'with_min_timestep'):
                        ts_control = ts_control.with_min_timestep(min_ts)
                    if hasattr(ts_control, 'with_max_timestep'):
                        ts_control = ts_control.with_max_timestep(max_ts)
                    if hasattr(ts_control, 'with_size_increase_factor'):
                        ts_control = ts_control.with_size_increase_factor(increase)
                    
                    calc = calc.with_timestep_control(ts_control)
                    self.write_log("  Applied timestep control")
                except ImportError:
                    self.write_log("  TimestepControl not available")
                except Exception as e:
                    self.write_log(f"  Error setting timestep control: {e}")
                
                self.write_log("Running calculation...")
                
                # CRITICAL: Enable saving results to file during simulation
                # This ensures get_time_steps() and get_mass_fraction_at_time() return data
                options = Options()
                options = options.enable_save_results_to_file(-1)  # Save at every timestep
                calc = calc.with_options(options)
                self.write_log("  Enabled saving results at every timestep")
                
                # Run calculation
                result = calc.calculate()
                
                self.write_log("Calculation completed!")
                
                # Extract results at each time
                self.extract_results(result, elements, sim_times)
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.write_log(f"ERROR: {error_msg}")
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Calculation Error", msg))
            
        finally:
            # Reset UI
            self.root.after(0, self.reset_ui_after_calculation)
            
    def extract_results(self, result, elements, sim_times):
        """Extract and display results from the calculation at multiple times."""
        self.write_log("Extracting results...")
        
        try:
            # Store results: {element: {time: (positions, compositions)}}
            self.result_data = {'compositions': {}, 'phase_fractions': {}, 'metadata': {}}
            self.sim_times = sim_times  # Store for plotting
            self.tc_result = result  # Store for additional queries
            
            # Get regions info
            try:
                regions = result.get_regions()
                self.result_data['metadata']['regions'] = regions
                self.write_log(f"  Regions: {regions}")
            except:
                pass
            
            # Get time steps available - use these for extraction instead of just output times
            extraction_times = sim_times  # Default to user-specified times
            try:
                all_time_steps = result.get_time_steps()
                self.result_data['metadata']['time_steps'] = all_time_steps
                self.write_log(f"  Available time steps: {len(all_time_steps)} points")
                
                # Use all internal timesteps for time-series plots
                # Sample if there are too many (more than 50 points)
                if all_time_steps and len(all_time_steps) > 0:
                    if len(all_time_steps) > 50:
                        # Sample evenly to get ~50 points
                        step = max(1, len(all_time_steps) // 50)
                        extraction_times = all_time_steps[::step]
                        # Always include the last time
                        if all_time_steps[-1] not in extraction_times:
                            extraction_times.append(all_time_steps[-1])
                        self.write_log(f"  Sampling {len(extraction_times)} time points for extraction")
                    else:
                        extraction_times = all_time_steps
                        self.write_log(f"  Using all {len(extraction_times)} time points for extraction")
            except Exception as e:
                self.write_log(f"  Could not get time steps: {e}, using sim_times")
            
            # Update sim_times to include all extraction points
            self.sim_times = extraction_times
            
            # Extract composition profiles for each element at ALL extracted times
            for elem in elements:
                if elem == 'Fe':
                    continue
                self.result_data['compositions'][elem] = {}
                
                for t in extraction_times:
                    try:
                        # Get profile at each time
                        positions, compositions = result.get_mass_fraction_of_component_at_time(elem, t)
                        self.result_data['compositions'][elem][t] = (positions, compositions)
                    except Exception as e:
                        pass  # Some times may not have data
                
                extracted_count = len(self.result_data['compositions'][elem])
                self.write_log(f"  {elem}: extracted {extracted_count} time points")
            
            # Extract phase fractions (unless skipped by user)
            if not self.skip_phase_fractions.get():
                phases = ['FCC_A1', 'BCC_A2', 'CEMENTITE']
                for phase in phases:
                    self.result_data['phase_fractions'][phase] = {}
                    for t in extraction_times:
                        try:
                            positions, fractions = result.get_mass_fraction_of_phase_at_time(phase, t)
                            self.result_data['phase_fractions'][phase][t] = (positions, fractions)
                        except Exception as e:
                            pass  # Phase may not exist
                    extracted_count = len(self.result_data['phase_fractions'][phase])
                    if extracted_count > 0:
                        self.write_log(f"  {phase}: extracted {extracted_count} time points")
            else:
                self.write_log("  Phase fractions: skipped (user option)")
            
            # Calculate system-averaged phase fractions from interface position
            # This is what shows the FCC/BCC ratio based on where the interface is
            self.result_data['system_phase_fractions'] = {}
            try:
                # Get total system size from region widths
                total_width = sum(r['width'].get() for r in self.regions)
                
                # Try to get interface position at each time
                for t in extraction_times:
                    try:
                        # The interface is between the two regions
                        # Get positions from FCC region and find the maximum (interface location)
                        if 'FCC_A1' in self.result_data.get('phase_fractions', {}):
                            fcc_data = self.result_data['phase_fractions']['FCC_A1']
                            if t in fcc_data:
                                positions, _ = fcc_data[t]
                                if positions:
                                    interface_pos = max(positions)  # Right edge of FCC region
                                    fcc_fraction = interface_pos / total_width
                                    bcc_fraction = 1.0 - fcc_fraction
                                    self.result_data['system_phase_fractions'][t] = {
                                        'FCC_A1': fcc_fraction * 100,
                                        'BCC_A2': bcc_fraction * 100,
                                        'interface_position': interface_pos * 1e6  # in µm
                                    }
                                    self.write_log(f"  System fractions at t={t}s: FCC={fcc_fraction*100:.1f}%, BCC={bcc_fraction*100:.1f}%, interface at {interface_pos*1e6:.2f}µm")
                    except Exception as e:
                        self.write_log(f"  Could not calculate system fractions at t={t}s: {e}")
            except Exception as e:
                self.write_log(f"  Error calculating system-averaged fractions: {e}")
                    
            # Update results tab
            self.root.after(0, self.display_results)
            
        except Exception as e:
            self.write_log(f"Error extracting results: {e}")
    def display_results(self):
        """Display results with interactive plot controls like Thermo-Calc's Plot Renderer."""
        if not MATPLOTLIB_AVAILABLE:
            self.write_log("Matplotlib not available for plotting.")
            return
            
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        if not hasattr(self, 'result_data') or not self.result_data:
            ttk.Label(self.results_frame, text="No results to display.").pack(pady=50)
            return
        
        # Store plot controls as instance variables
        self.plot_vars = {}
        
        # === Configuration Panel (Top) ===
        config_frame = ttk.LabelFrame(self.results_frame, text="Plot Configuration", padding="10")
        config_frame.pack(fill='x', pady=(0, 10))
        
        # --- X-Axis Settings ---
        x_frame = ttk.Frame(config_frame)
        x_frame.pack(fill='x', pady=2)
        
        ttk.Label(x_frame, text="X-Axis:", font=('Helvetica', 10, 'bold')).pack(side='left', padx=(0, 10))
        
        ttk.Label(x_frame, text="Variable:").pack(side='left')
        self.plot_vars['x_variable'] = tk.StringVar(value="Distance")
        x_var_combo = ttk.Combobox(x_frame, textvariable=self.plot_vars['x_variable'],
                                    values=["Distance", "Time"], state='readonly', width=12)
        x_var_combo.pack(side='left', padx=5)
        
        ttk.Label(x_frame, text="Unit:").pack(side='left', padx=(10, 0))
        self.plot_vars['x_unit'] = tk.StringVar(value="µm")
        x_unit_combo = ttk.Combobox(x_frame, textvariable=self.plot_vars['x_unit'],
                                     values=["µm", "m", "mm"], state='readonly', width=8)
        x_unit_combo.pack(side='left', padx=5)
        
        ttk.Label(x_frame, text="Region:").pack(side='left', padx=(10, 0))
        self.plot_vars['region'] = tk.StringVar(value="All regions")
        region_names = ["All regions"] + [r['name'].get() for r in self.regions]
        region_combo = ttk.Combobox(x_frame, textvariable=self.plot_vars['region'],
                                     values=region_names, state='readonly', width=15)
        region_combo.pack(side='left', padx=5)
        
        # --- Y-Axis Settings ---
        y_frame = ttk.Frame(config_frame)
        y_frame.pack(fill='x', pady=2)
        
        ttk.Label(y_frame, text="Y-Axis:", font=('Helvetica', 10, 'bold')).pack(side='left', padx=(0, 10))
        
        ttk.Label(y_frame, text="Variable:").pack(side='left')
        self.plot_vars['y_variable'] = tk.StringVar(value="Composition")
        y_var_combo = ttk.Combobox(y_frame, textvariable=self.plot_vars['y_variable'],
                                    values=["Composition", "Phase fraction", "Interface position",
                                            "System phase fractions"], state='readonly', width=18)
        y_var_combo.pack(side='left', padx=5)
        y_var_combo.bind('<<ComboboxSelected>>', self.on_y_variable_changed)
        
        # Element/Phase selector
        ttk.Label(y_frame, text="Element/Phase:").pack(side='left', padx=(10, 0))
        elements = list(self.result_data.get('compositions', {}).keys())
        phases = ['FCC_A1', 'BCC_A2']
        self.plot_vars['element_phase'] = tk.StringVar(value=elements[0] if elements else "C")
        self.element_phase_combo = ttk.Combobox(y_frame, textvariable=self.plot_vars['element_phase'],
                                                 values=elements if elements else ["C", "Mn", "Si", "Al"],
                                                 state='readonly', width=10)
        self.element_phase_combo.pack(side='left', padx=5)
        
        ttk.Label(y_frame, text="Unit:").pack(side='left', padx=(10, 0))
        self.plot_vars['y_unit'] = tk.StringVar(value="Mass %")
        y_unit_combo = ttk.Combobox(y_frame, textvariable=self.plot_vars['y_unit'],
                                     values=["Mass %", "Mole %", "By volume"], state='readonly', width=10)
        y_unit_combo.pack(side='left', padx=5)
        
        # --- Time Settings ---
        time_frame = ttk.Frame(config_frame)
        time_frame.pack(fill='x', pady=2)
        
        ttk.Label(time_frame, text="Time (s):", font=('Helvetica', 10, 'bold')).pack(side='left', padx=(0, 10))
        
        # Get available times from results - default to max simulation time
        available_times = self.result_data.get('metadata', {}).get('time_steps', [])
        if hasattr(self, 'sim_times') and self.sim_times:
            max_time = max(self.sim_times)
            default_times = str(max_time)
        else:
            default_times = "20"
        
        self.plot_vars['times'] = tk.StringVar(value=default_times)
        time_entry = ttk.Entry(time_frame, textvariable=self.plot_vars['times'], width=40)
        time_entry.pack(side='left', padx=5)
        
        ttk.Label(time_frame, text="(use comma or space; leave empty for all available)", foreground='gray').pack(side='left', padx=5)
        
        # --- Y-Axis Limits ---
        limits_frame = ttk.Frame(config_frame)
        limits_frame.pack(fill='x', pady=2)
        
        ttk.Label(limits_frame, text="Y-Limits:", font=('Helvetica', 10, 'bold')).pack(side='left', padx=(0, 10))
        
        self.plot_vars['auto_scale'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(limits_frame, text="Auto-scale", variable=self.plot_vars['auto_scale']).pack(side='left', padx=5)
        
        ttk.Label(limits_frame, text="Min:").pack(side='left', padx=(10, 0))
        self.plot_vars['y_min'] = tk.StringVar(value="")
        y_min_entry = ttk.Entry(limits_frame, textvariable=self.plot_vars['y_min'], width=8)
        y_min_entry.pack(side='left', padx=2)
        
        ttk.Label(limits_frame, text="Max:").pack(side='left', padx=(10, 0))
        self.plot_vars['y_max'] = tk.StringVar(value="")
        y_max_entry = ttk.Entry(limits_frame, textvariable=self.plot_vars['y_max'], width=8)
        y_max_entry.pack(side='left', padx=2)
        
        ttk.Label(limits_frame, text="(leave empty + uncheck Auto for data range)", foreground='gray').pack(side='left', padx=5)
        
        # --- Action Buttons ---
        btn_frame = ttk.Frame(config_frame)
        btn_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(btn_frame, text=" Update Plot", command=self.update_interactive_plot).pack(side='left', padx=5)
        ttk.Button(btn_frame, text=" Export to Excel", command=self.export_results_to_csv).pack(side='left', padx=5)
        ttk.Button(btn_frame, text=" Reset", command=self.reset_plot_config).pack(side='left', padx=5)
        
        # === Plot Area (Bottom) ===
        self.plot_frame = ttk.Frame(self.results_frame)
        self.plot_frame.pack(fill='both', expand=True)
        
        # Initial plot
        self.update_interactive_plot()
        
        self.write_log("Interactive results displayed.")
    
    def on_y_variable_changed(self, event=None):
        """Update element/phase combo based on Y-axis variable selection."""
        y_var = self.plot_vars['y_variable'].get()
        
        if y_var == "Composition":
            elements = list(self.result_data.get('compositions', {}).keys())
            self.element_phase_combo['values'] = elements if elements else ["C", "Mn", "Si", "Al"]
            if elements:
                self.plot_vars['element_phase'].set(elements[0])
        elif y_var == "Phase fraction":
            phases = list(self.result_data.get('phase_fractions', {}).keys())
            self.element_phase_combo['values'] = phases if phases else ["FCC_A1", "BCC_A2"]
            if phases:
                self.plot_vars['element_phase'].set(phases[0])
        elif y_var in ["Interface position", "System phase fractions"]:
            self.element_phase_combo['values'] = ["N/A"]
            self.plot_vars['element_phase'].set("N/A")
    
    def reset_plot_config(self):
        """Reset plot configuration to defaults."""
        self.plot_vars['x_variable'].set("Distance")
        self.plot_vars['x_unit'].set("µm")
        self.plot_vars['region'].set("All regions")
        self.plot_vars['y_variable'].set("Composition")
        self.plot_vars['y_unit'].set("Mass %")
        elements = list(self.result_data.get('compositions', {}).keys())
        if elements:
            self.plot_vars['element_phase'].set(elements[0])
        if hasattr(self, 'sim_times') and self.sim_times:
            max_time = max(self.sim_times)
            default_times = str(max_time)
        else:
            default_times = "20"
        self.plot_vars['times'].set(default_times)
        self.plot_vars['auto_scale'].set(True)
        self.plot_vars['y_min'].set("")
        self.plot_vars['y_max'].set("")
        self.update_interactive_plot()
    
    def update_interactive_plot(self):
        """Update the plot based on current configuration selections."""
        # Clear previous plot
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Get current settings
        x_var = self.plot_vars['x_variable'].get()
        x_unit = self.plot_vars['x_unit'].get()
        y_var = self.plot_vars['y_variable'].get()
        y_unit = self.plot_vars['y_unit'].get()
        elem_phase = self.plot_vars['element_phase'].get()
        region_filter = self.plot_vars['region'].get()
        
        # Parse time values - support both comma and space separators
        try:
            time_str = self.plot_vars['times'].get().strip()
            if time_str:
                # Replace commas with spaces, then split by whitespace
                time_str = time_str.replace(',', ' ')
                times = [float(t.strip()) for t in time_str.split() if t.strip()]
            else:
                times = []
        except:
            times = self.sim_times if hasattr(self, 'sim_times') else [1.0]
        
        # If no times specified, use all available times
        if not times:
            # Get all available times from composition data
            compositions = self.result_data.get('compositions', {})
            if compositions:
                first_elem = list(compositions.keys())[0] if compositions else None
                if first_elem and compositions[first_elem]:
                    times = sorted(compositions[first_elem].keys())
            if not times:
                times = self.sim_times if hasattr(self, 'sim_times') else [1.0]
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Unit conversion factors - positions are stored in METERS
        # µm: multiply by 1e6 to convert m→µm
        # mm: multiply by 1e3 to convert m→mm  
        # m: multiply by 1 (no conversion)
        x_factor = {'µm': 1e6, 'm': 1, 'mm': 1e3}.get(x_unit, 1e6)
        
        try:
            if y_var == "Composition" and x_var == "Distance":
                self.plot_composition_vs_distance(ax, elem_phase, times, x_factor, x_unit)
            elif y_var == "Composition" and x_var == "Time":
                self.plot_composition_vs_time(ax, elem_phase, times)
            elif y_var == "Phase fraction" and x_var == "Distance":
                self.plot_phase_fraction_vs_distance(ax, elem_phase, times, x_factor, x_unit)
            elif y_var == "Phase fraction" and x_var == "Time":
                self.plot_phase_fraction_vs_time(ax, elem_phase, times)
            elif y_var == "Interface position":
                self.plot_interface_position(ax, times)
            elif y_var == "System phase fractions":
                self.plot_system_phase_fractions(ax, times)
            else:
                ax.text(0.5, 0.5, f"Plot type not implemented:\n{y_var} vs {x_var}", 
                       ha='center', va='center', fontsize=14, transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error creating plot:\n{e}", 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes, color='red')
            self.write_log(f"Plot error: {e}")
        
        ax.grid(True, alpha=0.3)
        
        # Apply Y-axis limits
        auto_scale = self.plot_vars.get('auto_scale', tk.BooleanVar(value=True)).get()
        if not auto_scale:
            try:
                y_min_str = self.plot_vars.get('y_min', tk.StringVar()).get().strip()
                y_max_str = self.plot_vars.get('y_max', tk.StringVar()).get().strip()
                
                if y_min_str or y_max_str:
                    current_ylim = ax.get_ylim()
                    y_min = float(y_min_str) if y_min_str else current_ylim[0]
                    y_max = float(y_max_str) if y_max_str else current_ylim[1]
                    ax.set_ylim(y_min, y_max)
                # If both empty and auto-scale off, matplotlib defaults to data range
            except ValueError:
                pass  # Invalid input, keep default
        
        plt.tight_layout()
        
        # Display
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        NavigationToolbar2Tk(canvas, self.plot_frame).update()
    
    # ==================== PLOTTING HELPER METHODS ====================
    
    def _get_times_to_plot(self, data_dict, requested_times):
        """Get valid times to plot, finding closest matches if needed."""
        available = sorted(data_dict.keys()) if data_dict else []
        if not available:
            return []
        
        # Find closest available time for each requested time
        result = []
        for t in requested_times:
            closest = min(available, key=lambda x: abs(x - t))
            if closest not in result:
                result.append(closest)
        return result if result else available
    
    def _show_no_data(self, ax, message):
        """Show 'no data' message centered in plot."""
        ax.text(0.5, 0.5, message, ha='center', va='center', transform=ax.transAxes)
    
    def plot_composition_vs_distance(self, ax, element, times, x_factor, x_unit):
        """Plot composition vs distance for an element at multiple times."""
        data = self.result_data.get('compositions', {}).get(element)
        if not data:
            return self._show_no_data(ax, f"No data for element: {element}")
        
        times_to_plot = self._get_times_to_plot(data, times)
        colors = plt.cm.viridis([i / max(1, len(times_to_plot)-1) for i in range(len(times_to_plot))])
        
        for i, t in enumerate(times_to_plot):
            if t in data:
                pos, comp = data[t]
                ax.plot([p * x_factor for p in pos], [c * 100 for c in comp],
                       color=colors[i], label=f"t={self.format_time(t)}", linewidth=2)
        
        ax.set_xlabel(f"Distance ({x_unit})")
        ax.set_ylabel(f"{element} (wt%)")
        ax.set_title(f"{element} Composition vs Distance")
        ax.legend(title="Time", loc='best')
    
    def plot_composition_vs_time(self, ax, element, times):
        """Plot element composition at interface vs time."""
        data = self.result_data.get('compositions', {}).get(element)
        if not data:
            return self._show_no_data(ax, f"No data for element: {element}")
        
        t_vals, c_vals = [], []
        for t in sorted(data.keys()):
            pos, comp = data[t]
            if comp:
                t_vals.append(t)
                c_vals.append(comp[len(comp)//2] * 100)  # Middle = interface
        
        ax.plot(t_vals, c_vals, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{element} at interface (wt%)")
        ax.set_title(f"{element} at Interface vs Time")
    
    def plot_phase_fraction_vs_distance(self, ax, phase, times, x_factor, x_unit):
        """Plot phase fraction vs distance at multiple times."""
        data = self.result_data.get('phase_fractions', {}).get(phase)
        if not data:
            return self._show_no_data(ax, f"No data for phase: {phase}")
        
        times_to_plot = self._get_times_to_plot(data, times)
        colors = plt.cm.viridis([i / max(1, len(times_to_plot)-1) for i in range(len(times_to_plot))])
        
        for i, t in enumerate(times_to_plot):
            if t in data:
                pos, frac = data[t]
                ax.plot([p * x_factor for p in pos], [f * 100 for f in frac],
                       color=colors[i], label=f"t={self.format_time(t)}", linewidth=2)
        
        ax.set_xlabel(f"Distance ({x_unit})")
        ax.set_ylabel(f"{phase} Fraction (%)")
        ax.set_title(f"{phase} vs Distance")
        ax.legend(title="Time", loc='best')
    
    def plot_phase_fraction_vs_time(self, ax, phase, times):
        """Plot system-level phase fraction vs time."""
        sys_frac = self.result_data.get('system_phase_fractions', {})
        if not sys_frac:
            return self._show_no_data(ax, "No system phase fraction data")
        
        t_vals = [t for t in sorted(sys_frac.keys()) if phase in sys_frac[t]]
        p_vals = [sys_frac[t][phase] for t in t_vals]
        color = 'blue' if 'FCC' in phase else 'red'
        
        ax.plot(t_vals, p_vals, '-o', linewidth=2, markersize=6, color=color)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{phase} Fraction (%)")
        ax.set_title(f"Total {phase} vs Time")
    
    def plot_interface_position(self, ax, times):
        """Plot interface position vs time."""
        sys_frac = self.result_data.get('system_phase_fractions', {})
        if not sys_frac:
            return self._show_no_data(ax, "No interface position data")
        
        t_vals = [t for t in sorted(sys_frac.keys()) if 'interface_position' in sys_frac[t]]
        pos_vals = [sys_frac[t]['interface_position'] for t in t_vals]
        
        ax.plot(t_vals, pos_vals, 'g-o', linewidth=2, markersize=6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Interface Position (µm)")
        ax.set_title("Phase Interface Position vs Time")
    
    def plot_system_phase_fractions(self, ax, times):
        """Plot bar chart of final phase fractions."""
        sys_frac = self.result_data.get('system_phase_fractions', {})
        if not sys_frac:
            return self._show_no_data(ax, "No system phase fraction data")
        
        latest_t = max(sys_frac.keys())
        latest = sys_frac[latest_t]
        
        phases = ['FCC_A1\n(Austenite)', 'BCC_A2\n(Ferrite)']
        fracs = [latest.get('FCC_A1', 0), latest.get('BCC_A2', 0)]
        
        bars = ax.bar(phases, fracs, color=['blue', 'red'], edgecolor='black', linewidth=2)
        for bar, f in zip(bars, fracs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{f:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_ylabel("Phase Fraction (%)")
        ax.set_title(f"Phase Fractions at t={self.format_time(latest_t)}")
    
    def format_time(self, t):
        """Format time value nicely."""
        if t >= 3600:
            return f"{t/3600:.1f}h"
        elif t >= 60:
            return f"{t/60:.1f}min"
        elif t >= 1:
            return f"{t:.1f}s"
        elif t >= 0.001:
            return f"{t*1000:.1f}ms"
        elif t >= 0.000001:
            return f"{t*1e6:.1f}µs"
        else:
            return f"{t:.2e}s"
        
    def reset_ui_after_calculation(self):
        """Reset UI after calculation completes."""
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress.stop()
        self.status_label.config(text="Ready")
        self.calculation_running = False
        
    def stop_calculation(self):
        """Stop the running calculation."""
        self.stop_requested = True
        self.write_log("Stop requested...")
        self.reset_ui_after_calculation()
    
    def run_batch_calculations(self):
        """Run batch calculations for multiple alloys from Excel import."""
        if not self.batch_alloys:
            messagebox.showerror("No Alloys", "No alloys loaded. Please import an Excel file first.")
            return
        
        # Validate inputs
        if not self.tdb_path.get() or not self.mdb_path.get():
            messagebox.showerror("Error", "Please select both TDB and MDB database files.")
            return
        
        # Create output directory
        output_dir = self.batch_output_dir.get()
        os.makedirs(output_dir, exist_ok=True)
        
        # Load progress (for resume)
        progress_file = os.path.join(output_dir, "progress.json")
        completed_alloys = set()
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                    completed_alloys = set(progress.get('completed', []))
                if completed_alloys:
                    result = messagebox.askyesno("Resume",
                        f"Found {len(completed_alloys)} completed alloys.\n\n"
                        "Resume from where it left off?")
                    if not result:
                        completed_alloys = set()
            except:
                pass
        
        # Filter alloys to process
        alloys_to_run = [a for a in self.batch_alloys if a['name'] not in completed_alloys]
        total_alloys = len(alloys_to_run)
        
        if total_alloys == 0:
            messagebox.showinfo("Complete", "All alloys have already been processed!")
            return
        
        # Update UI
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress.start(10)
        self.status_label.config(text=f"Processing {total_alloys} alloys...")
        self.calculation_running = True
        self.stop_requested = False
        
        # Start batch processing in thread
        def batch_thread():
            try:
                self.write_log(f"Starting batch processing: {total_alloys} alloys (sequential)")
                self.write_log(f"Output directory: {output_dir}")
                
                completed = list(completed_alloys)
                failed = []
                start_time = time.time()
                
                # Prepare common arguments
                common_args = {
                    'tdb_path': self.tdb_path.get(),
                    'mdb_path': self.mdb_path.get(),
                    'cache_path': self.cache_path.get(),
                    'elements': [elem for elem, var in self.element_vars.items() if var.get()],
                    'geometry': self.geometry_type.get(),
                    'output_dir': output_dir,
                    # Solver settings for homogenization support
                    'solver_type': self.solver_type.get(),
                    'homogenization_function': self.homogenization_function.get(),
                    'use_global_minimization': self.use_global_minimization.get(),
                }
                
                # Process alloys sequentially
                for idx, alloy in enumerate(alloys_to_run):
                    if self.stop_requested:
                        self.write_log("Batch processing stopped by user.")
                        break
                    
                    alloy_name = alloy['name']
                    self.write_log(f"\n[{idx+1}/{total_alloys}] Processing: {alloy_name}")
                    
                    # Update progress in UI
                    self.root.after(0, lambda n=alloy_name, i=idx: self.status_label.config(
                        text=f"Processing {i+1}/{total_alloys}: {n}"))
                    
                    try:
                        result = self.run_single_alloy(alloy, common_args)
                        
                        if result.get('success'):
                            completed.append(alloy_name)
                            self.write_log(f"  ✓ {alloy_name} completed successfully")
                            
                            # Save result to Excel file
                            result_file = os.path.join(output_dir, f"{alloy_name}.xlsx")
                            self.save_alloy_result(result, result_file)
                            self.write_log(f"  Saved to: {result_file}")
                        else:
                            failed.append((alloy_name, result.get('error', 'Unknown error')))
                            self.write_log(f"  ✗ {alloy_name} failed: {result.get('error')}")
                        
                    except Exception as e:
                        failed.append((alloy_name, str(e)))
                        self.write_log(f"  ✗ {alloy_name} error: {e}")
                    
                    # Save progress after each alloy
                    with open(progress_file, 'w') as f:
                        json.dump({
                            'completed': completed,
                            'failed': [{'name': n, 'error': e} for n, e in failed],
                            'total': len(self.batch_alloys),
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                        }, f, indent=2)
                
                # Summary
                elapsed = time.time() - start_time
                self.write_log(f"\n{'='*50}")
                self.write_log(f"Batch processing complete!")
                self.write_log(f"  Completed: {len(completed)}")
                self.write_log(f"  Failed: {len(failed)}")
                self.write_log(f"  Time: {elapsed/60:.1f} minutes")
                
                self.root.after(0, lambda: messagebox.showinfo("Batch Complete",
                    f"Processed {len(completed)} alloys successfully.\n"
                    f"Failed: {len(failed)}\n"
                    f"Time: {elapsed/60:.1f} minutes\n\n"
                    f"Results saved to:\n{output_dir}"))
                
            except Exception as e:
                self.write_log(f"Batch error: {e}")
                self.root.after(0, lambda: messagebox.showerror("Batch Error", str(e)))
            finally:
                self.root.after(0, self.reset_ui_after_calculation)
        
        self.calc_thread = threading.Thread(target=batch_thread)
        self.calc_thread.start()
    
    def run_single_alloy(self, alloy_config, common_args):
        """Run DICTRA calculation for a single alloy configuration."""
        result = {'success': False, 'data': {}, 'error': None}
        
        try:
            with TCPython() as session:
                session = session.set_cache_folder(common_args['cache_path'])
                
                # Get elements from alloy compositions
                alloy_elements = set()
                for region in alloy_config['regions']:
                    alloy_elements.update(region['compositions'].keys())
                elements = list(alloy_elements)
                
                # Load databases
                system = (
                    session
                    .select_thermodynamic_and_kinetic_databases_with_elements(
                        common_args['tdb_path'],
                        common_args['mdb_path'],
                        elements
                    )
                    .without_default_phases()
                    .select_phase("FCC_A1")
                    .select_phase("BCC_A2")
                    .get_system()
                )
                
                # Determine calculation type from heat treatment
                heat_treatment = alloy_config.get('heat_treatment', [])
                
                if len(heat_treatment) <= 1:
                    # Isothermal
                    calc = system.with_isothermal_diffusion_calculation()
                    temp_K = heat_treatment[0][0] + 273.15 if heat_treatment else 1173.0
                    calc = calc.set_temperature(temp_K)
                    sim_time = heat_treatment[0][1] if heat_treatment else 3600.0
                else:
                    # Non-isothermal - use TemperatureProfile
                    calc = system.with_non_isothermal_diffusion_calculation()
                    
                    # Build thermal profile with time-temperature points
                    # heat_treatment is list of (temp_C, time_s) tuples
                    heat_treatment_sorted = sorted(heat_treatment, key=lambda x: x[1])
                    
                    temp_profile = TemperatureProfile()
                    for temp_c, time_s in heat_treatment_sorted:
                        temp_k = temp_c + 273.15
                        temp_profile = temp_profile.add_time_temperature(time_s, temp_k)
                    
                    calc = calc.with_temperature_profile(temp_profile)
                    
                    # Simulation time is the last time point
                    sim_time = heat_treatment_sorted[-1][1]
                
                # Add regions
                for region_config in alloy_config['regions']:
                    width_m = region_config['width_um'] * 1e-6
                    phase = region_config['phase']
                    grid_points = region_config['points']
                    
                    # Create grid using CalculatedGrid (correct API)
                    if region_config['grid_type'].lower() == 'geometric':
                        grid = CalculatedGrid.geometric(grid_points, 1.2)
                    else:
                        grid = CalculatedGrid.linear(grid_points)
                    
                    # Create composition profile
                    compositions = region_config['compositions']
                    profile = CompositionProfile(Unit.MASS_PERCENT)
                    for elem, wt_pct in compositions.items():
                        if elem != 'Fe':
                            profile = profile.add(elem, ConstantProfile(wt_pct))
                    
                    # Build region correctly: Region(name), set_width, add_phase, with_grid, with_composition_profile
                    region_name = region_config.get('name', f"{phase}_region")
                    region = Region(region_name)
                    region = region.set_width(width_m)
                    region = region.add_phase(phase)
                    region = region.with_grid(grid)
                    region = region.with_composition_profile(profile)
                    
                    calc = calc.add_region(region)
                
                # Set simulation time
                calc = calc.set_simulation_time(sim_time)
                
                # CRITICAL: Enable saving results to file during simulation
                # This ensures get_time_steps() and get_mass_fraction_at_time() return data
                options = Options()
                options = options.enable_save_results_to_file(-1)  # Save at every timestep
                calc = calc.with_options(options)
                
                # Calculate
                dictra_result = calc.calculate()
                
                # Extract results (similar to single calculation)
                result['data'] = {
                    'alloy_name': alloy_config['name'],
                    'heat_treatment': heat_treatment,
                    'simulation_time': sim_time,
                    'compositions': {},
                    'phase_fractions': {},
                    'system_phase_fractions': {}
                }
                
                # Get all time steps and sample up to 100 points for detailed results
                all_time_steps = dictra_result.get_time_steps()
                result['data']['time_steps'] = list(all_time_steps) if all_time_steps else []
                
                # Sample time points: use ~100 evenly spaced points if too many
                if all_time_steps and len(all_time_steps) > 100:
                    step = max(1, len(all_time_steps) // 100)
                    extraction_times = list(all_time_steps[::step])
                    if all_time_steps[-1] not in extraction_times:
                        extraction_times.append(all_time_steps[-1])
                else:
                    extraction_times = list(all_time_steps) if all_time_steps else []
                
                result['data']['extraction_times'] = extraction_times
                
                # Extract composition profiles for each element at all extraction times
                for elem in elements:
                    if elem == 'Fe':
                        continue
                    result['data']['compositions'][elem] = {}
                    for t in extraction_times:
                        try:
                            positions, comps = dictra_result.get_mass_fraction_of_component_at_time(elem, t)
                            result['data']['compositions'][elem][t] = (list(positions), list(comps))
                        except:
                            pass
                
                # Extract phase fractions for FCC_A1 and BCC_A2
                for phase in ['FCC_A1', 'BCC_A2']:
                    result['data']['phase_fractions'][phase] = {}
                    for t in extraction_times:
                        try:
                            positions, fractions = dictra_result.get_mass_fraction_of_phase_at_time(phase, t)
                            result['data']['phase_fractions'][phase][t] = (list(positions), list(fractions))
                        except:
                            pass
                
                # Calculate system-level phase fractions based on interface position
                total_width_m = sum(r['width_um'] * 1e-6 for r in alloy_config['regions'])
                result['data']['total_width_um'] = total_width_m * 1e6
                
                for t in extraction_times:
                    try:
                        fcc_data = result['data']['phase_fractions'].get('FCC_A1', {})
                        if t in fcc_data:
                            positions, _ = fcc_data[t]
                            if positions:
                                interface_pos = max(positions)
                                fcc_fraction = interface_pos / total_width_m
                                result['data']['system_phase_fractions'][t] = {
                                    'FCC_A1': fcc_fraction * 100,
                                    'BCC_A2': (1.0 - fcc_fraction) * 100,
                                    'interface_position_um': interface_pos * 1e6
                                }
                    except:
                        pass
                
                result['success'] = True
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def save_alloy_result(self, result, filepath):
        """Save single alloy result to Excel file."""
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                data = result.get('data', {})
                
                # Info sheet with more details
                time_steps = data.get('time_steps', [])
                extraction_times = data.get('extraction_times', [])
                info = {
                    'Alloy': [data.get('alloy_name', 'Unknown')],
                    'Simulation Time (s)': [data.get('simulation_time', 0)],
                    'Heat Treatment': [str(data.get('heat_treatment', []))],
                    'Total Time Steps': [len(time_steps)],
                    'Extracted Time Points': [len(extraction_times)]
                }
                pd.DataFrame(info).to_excel(writer, sheet_name='Info', index=False)
                
                # Compositions sheet
                compositions = data.get('compositions', {})
                if compositions:
                    rows = []
                    for elem, time_data in compositions.items():
                        for t, (positions, comps) in time_data.items():
                            for pos, comp in zip(positions, comps):
                                rows.append({
                                    'Time_s': t,
                                    'Position_um': pos * 1e6,
                                    'Element': elem,
                                    'wt%': comp * 100
                                })
                    if rows:
                        pd.DataFrame(rows).to_excel(writer, sheet_name='Compositions', index=False)
                
                # Phase Fractions sheet (local within each region - kept for reference)
                phase_fractions = data.get('phase_fractions', {})
                if phase_fractions:
                    rows = []
                    for phase, time_data in phase_fractions.items():
                        for t, (positions, fractions) in time_data.items():
                            for pos, frac in zip(positions, fractions):
                                rows.append({
                                    'Time_s': t,
                                    'Position_um': pos * 1e6,
                                    'Phase': phase,
                                    'Fraction': frac
                                })
                    if rows:
                        pd.DataFrame(rows).to_excel(writer, sheet_name='Local_Phase_Fractions', index=False)
                
                # System-level Phase Fractions (based on interface position / total width)
                system_phase_fractions = data.get('system_phase_fractions', {})
                if system_phase_fractions:
                    rows = []
                    total_width = data.get('total_width_um', 0)
                    for t, fracs in system_phase_fractions.items():
                        rows.append({
                            'Time_s': t,
                            'FCC_A1_%': fracs.get('FCC_A1', 0),
                            'BCC_A2_%': fracs.get('BCC_A2', 0),
                            'Interface_Position_um': fracs.get('interface_position_um', 0),
                            'Total_Width_um': total_width
                        })
                    if rows:
                        # Sort by time
                        df = pd.DataFrame(rows).sort_values('Time_s')
                        df.to_excel(writer, sheet_name='System_Phase_Fractions', index=False)
                        
        except Exception as e:
            self.write_log(f"Error saving {filepath}: {e}")
    
    def export_results_to_csv(self):
        """Export results to Excel file with multiple sheets."""
        if not hasattr(self, 'result_data') or not self.result_data:
            messagebox.showwarning("No Data", "No results to export.")
            return
        if not PANDAS_AVAILABLE:
            messagebox.showerror("Missing Package", "pandas required: pip install pandas openpyxl")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Save Results to Excel"
        )
        if not filename:
            return
        
        def build_profile_df(data_dict, value_multiplier=100, col_suffix=''):
            """Build DataFrame from {key: {time: (positions, values)}} data."""
            keys = list(data_dict.keys())
            all_times = set()
            for k in keys:
                all_times.update(data_dict[k].keys())
            
            rows = []
            for t in sorted(all_times):
                # Get positions from first key that has data at this time
                positions = None
                for k in keys:
                    if t in data_dict[k]:
                        positions, _ = data_dict[k][t]
                        if positions:
                            break
                
                if positions:
                    for i, pos in enumerate(positions):
                        row = {'Time_s': t, 'Position_um': pos * 1e6}
                        for k in keys:
                            val = None
                            if t in data_dict[k]:
                                _, values = data_dict[k][t]
                                if values and i < len(values):
                                    val = values[i] * value_multiplier
                            row[f'{k}{col_suffix}'] = val
                        rows.append(row)
            return pd.DataFrame(rows) if rows else None
        
        try:
            sheets = []
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Compositions
                comps = self.result_data.get('compositions', {})
                if comps:
                    df = build_profile_df(comps, value_multiplier=100, col_suffix='_wt%')
                    if df is not None:
                        df.to_excel(writer, sheet_name='Compositions', index=False)
                        sheets.append('Compositions')
                
                # Phase fractions
                phases = {p: d for p, d in self.result_data.get('phase_fractions', {}).items() if d}
                if phases:
                    df = build_profile_df(phases, value_multiplier=100, col_suffix='_%')
                    if df is not None:
                        df.to_excel(writer, sheet_name='Phase_Fractions', index=False)
                        sheets.append('Phase_Fractions')
                
                # System fractions (simpler structure)
                sys_frac = self.result_data.get('system_phase_fractions', {})
                if sys_frac:
                    rows = [{'Time_s': t, 
                             'FCC_A1_%': sf.get('FCC_A1'), 
                             'BCC_A2_%': sf.get('BCC_A2'),
                             'Interface_um': sf.get('interface_position')}
                            for t, sf in sorted(sys_frac.items())]
                    pd.DataFrame(rows).to_excel(writer, sheet_name='System_Fractions', index=False)
                    sheets.append('System_Fractions')
                
                if not sheets:
                    pd.DataFrame({'Message': ['No data']}).to_excel(writer, sheet_name='Info', index=False)
            
            self.write_log(f"Export complete: {filename}")
            messagebox.showinfo("Export Complete", f"Saved: {filename}\nSheets: {', '.join(sheets)}")
            
        except Exception as e:
            self.write_log(f"Export error: {e}")
            messagebox.showerror("Export Error", str(e))


def main():
    """Main entry point."""
    root = tk.Tk()
    app = DICTRACalculatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
