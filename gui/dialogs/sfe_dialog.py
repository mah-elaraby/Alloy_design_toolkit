"""
SFE settings configuration dialog
"""

import tkinter as tk
from tkinter import ttk, messagebox


class SFEDialog:
    """Dialog for configuring SFE calculations."""

    def __init__(self, app):
        self.app = app
        self.window = None
        self.setup_ui()

    def setup_ui(self):
        """Create the SFE settings window."""
        self.window = tk.Toplevel(self.app.root)
        self.window.title("Stacking Fault Energy Settings")

        main_frame = ttk.Frame(self.window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Advanced parameters
        adv_frame = ttk.LabelFrame(main_frame, text="Advanced Parameters", padding="10")
        adv_frame.pack(fill=tk.X, pady=10)

        params = [
            ("Interfacial energy σ (mJ/m²)", 'sigma', self.app.sfe_params['sigma']),
            ("Grain Size (μm)", 'grain_size', self.app.sfe_params['grain_size']),
            ("Lattice parameter a (m)", 'lattice_param', self.app.sfe_params['lattice_param']),
            ("Temperature (K)", 'temperature', self.app.sfe_params['temperature'])
        ]

        self.sfe_entries = {}
        for i, (label, key, default) in enumerate(params):
            ttk.Label(adv_frame, text=label).grid(row=i, column=0, padx=5, pady=5, sticky=tk.W)
            var = tk.DoubleVar(value=default)
            ttk.Entry(adv_frame, textvariable=var).grid(row=i, column=1, padx=5, pady=5, sticky=tk.EW)
            self.sfe_entries[key] = var

        # Configure grid weights for adv_frame
        adv_frame.grid_columnconfigure(0, weight=1)
        adv_frame.grid_columnconfigure(1, weight=2, uniform="x")

        # Note about Excel import
        note_frame = ttk.Frame(main_frame)
        note_frame.pack(fill=tk.X, pady=10)

        ttk.Label(
            note_frame,
            text="Note: SFE will be calculated automatically from Phase Calculation results",
            font=('Arial', 10, 'italic'),
            foreground='blue',
            wraplength=450
        ).pack()

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

    def save_config(self):
        """Save SFE configuration."""
        try:
            for key, var in self.sfe_entries.items():
                self.app.sfe_params[key] = var.get()

            self.app.update_log("SFE configuration saved.")
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")