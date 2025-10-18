"""
MOO algorithm settings configuration dialog
"""

import tkinter as tk
from tkinter import ttk, messagebox


class MOODialog:
    """Dialog for configuring MOO algorithm settings."""

    def __init__(self, app):
        self.app = app
        self.window = None
        self.setup_ui()

    def setup_ui(self):
        """Create the MOO settings window."""
        self.window = tk.Toplevel(self.app.root)
        self.window.title("Optimization Algorithm Settings")

        main_frame = ttk.Frame(self.window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Martensite parameters
        ma_frame = ttk.LabelFrame(main_frame, text="Martensite/Austenite Parameters", padding="10")
        ma_frame.pack(fill=tk.X, pady=10)

        ttk.Label(ma_frame, text="Quenching Temperature (°C):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.quench_temp_var = tk.DoubleVar(value=self.app.quenching_temp)
        ttk.Entry(ma_frame, textvariable=self.quench_temp_var).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(ma_frame, text="(Used for Fm and RA calculations)", font=('Arial', 8, 'italic'),
                  foreground='gray', wraplength=200).grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

        # Configure grid weights for ma_frame
        ma_frame.grid_columnconfigure(0, weight=1)
        ma_frame.grid_columnconfigure(1, weight=2, uniform="x")
        ma_frame.grid_columnconfigure(2, weight=1)

        # Alloy design constraints
        constraints_frame = ttk.LabelFrame(main_frame, text="Alloy Design Constraints", padding="10")
        constraints_frame.pack(fill=tk.X, pady=10)

        constraints = [
            ('Ms min (°C)', 'Ms_min', -50.0),
            ('Ms max (°C)', 'Ms_max', 0.0),
            ('RA min', 'Ra_min', 0.4),
            ('RA max', 'Ra_max', 0.55),
            ('SFE min (mJ/m²)', 'SFE_min', 15.0),
            ('SFE max (mJ/m²)', 'SFE_max', 25.0),
            ('Min ΔT (°C)', 'min_deltaT', 10.0),
            ('Cementite limit', 'Cementite_max', 0.0)
        ]

        self.moo_entries = {}
        for i, (label, key, default) in enumerate(constraints):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(constraints_frame, text=label).grid(row=row, column=col, padx=5, pady=5, sticky=tk.W)
            var = tk.DoubleVar(value=self.app.moo_constraints.get(key, default))
            ttk.Entry(constraints_frame, textvariable=var).grid(row=row, column=col + 1, padx=5, pady=5, sticky=tk.EW)
            self.moo_entries[key] = var

        # Configure grid weights for constraints_frame
        for c in range(4):
            weight = 1 if c % 2 == 0 else 2
            constraints_frame.grid_columnconfigure(c, weight=weight, uniform="x")

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
        """Save MOO configuration."""
        try:
            # Save quenching temperature
            self.app.quenching_temp = self.quench_temp_var.get()

            # Save MOO constraints
            for key, var in self.moo_entries.items():
                self.app.moo_constraints[key] = var.get()

            self.app.update_log("MOO constraints and M/A parameters saved.")
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")