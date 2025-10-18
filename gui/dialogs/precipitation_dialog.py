"""
Precipitation settings configuration dialog
"""

import tkinter as tk
from tkinter import ttk, messagebox


class PrecipitationDialog:
    """Dialog for configuring precipitation kinetics settings."""

    def __init__(self, app):
        self.app = app
        self.window = None
        self.setup_ui()

    def setup_ui(self):
        """Create the precipitation settings window."""
        self.window = tk.Toplevel(self.app.root)
        self.window.title("Precipitation Settings")

        main_frame = ttk.Frame(self.window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Precipitation parameters
        params_frame = ttk.LabelFrame(main_frame, text="Precipitation Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=5)

        ttk.Label(params_frame, text="Simulation Time (seconds):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.precip_time = tk.DoubleVar(value=self.app.precipitation_params['sim_time'])
        ttk.Entry(params_frame, textvariable=self.precip_time).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(params_frame, text="Matrix Phase:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.matrix_phase = tk.StringVar(value=self.app.precipitation_params['matrix_phase'])
        ttk.Entry(params_frame, textvariable=self.matrix_phase).grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(params_frame, text="Precipitate Phase:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.precipitate_phase = tk.StringVar(value=self.app.precipitation_params['precipitate_phase'])
        ttk.Entry(params_frame, textvariable=self.precipitate_phase).grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

        # Configure grid weights for params_frame
        params_frame.grid_columnconfigure(0, weight=1)
        params_frame.grid_columnconfigure(1, weight=2, uniform="x")

        # Database settings
        db_frame = ttk.LabelFrame(main_frame, text="Database Settings", padding="10")
        db_frame.pack(fill=tk.X, pady=5)

        ttk.Label(db_frame, text="Thermodynamic DB:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.tdb = tk.StringVar(value=self.app.precipitation_params['tdb'])
        ttk.Entry(db_frame, textvariable=self.tdb).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(db_frame, text="Kinetic DB:").grid(row=0, column=2, padx=15, pady=5, sticky=tk.W)
        self.kdb = tk.StringVar(value=self.app.precipitation_params['kdb'])
        ttk.Entry(db_frame, textvariable=self.kdb).grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

        # Configure grid weights for db_frame
        for c in range(4):
            weight = 1 if c % 2 == 0 else 2
            db_frame.grid_columnconfigure(c, weight=weight, uniform="x")

        # Growth rate model
        growth_frame = ttk.LabelFrame(main_frame, text="Growth Rate Model", padding="10")
        growth_frame.pack(fill=tk.X, pady=5)

        ttk.Label(growth_frame, text="Model:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.growth_model = tk.StringVar(value=self.app.precipitation_params['growth_model'])
        models = ["Simplified", "General", "Advanced", "Para_eq", "NPLE", "PE_AUTOMATIC"]
        ttk.Combobox(growth_frame, textvariable=self.growth_model, values=models, state='readonly').grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        # Configure grid weights for growth_frame
        growth_frame.grid_columnconfigure(0, weight=1)
        growth_frame.grid_columnconfigure(1, weight=2, uniform="x")

        # Nucleation sites
        nucleation_frame = ttk.LabelFrame(main_frame, text="Nucleation Sites", padding="10")
        nucleation_frame.pack(fill=tk.X, pady=5)

        ttk.Label(nucleation_frame, text="Site:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.nucleation_site = tk.StringVar(value=self.app.precipitation_params['nucleation_site'])
        sites = ["Bulk", "Grain boundaries", "Grain edges", "Grain corners", "Dislocations"]
        ttk.Combobox(nucleation_frame, textvariable=self.nucleation_site, values=sites, state='readonly').grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        # Configure grid weights for nucleation_frame
        nucleation_frame.grid_columnconfigure(0, weight=1)
        nucleation_frame.grid_columnconfigure(1, weight=2, uniform="x")

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=15)

        ttk.Button(
            button_frame,
            text="Save Configuration",
            command=self.save_config
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            button_frame,
            text="Cancel",
            command=self.window.destroy
        ).pack(side=tk.RIGHT, padx=5)

        # Auto-size window to content
        self.window.update_idletasks()
        self.window.minsize(self.window.winfo_reqwidth(), self.window.winfo_reqheight())
        self.window.resizable(True, True)