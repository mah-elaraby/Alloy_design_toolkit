"""
Phase calculations configuration dialog
"""

import tkinter as tk
from tkinter import ttk, messagebox


class PhaseDialog:
    """Dialog for configuring phase calculations."""

    def __init__(self, app):
        self.app = app
        self.window = None
        self.setup_ui()

    def setup_ui(self):
        """Create the phase calculations configuration window."""
        self.window = tk.Toplevel(self.app.root)
        self.window.title("Phase Calculations Configuration")

        # Main container
        main_frame = ttk.Frame(self.window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Element Composition
        elem_frame = ttk.LabelFrame(main_frame, text="Element Composition (wt%)", padding="10")
        elem_frame.pack(fill=tk.X, pady=5)

        # Create element inputs (excluding Fe)
        self.elem_entries = {}
        headers = ["Element", "Start", "End", "Step"]
        for i, h in enumerate(headers):
            ttk.Label(elem_frame, text=h, font=('Arial', 9, 'bold')).grid(row=0, column=i, padx=5, pady=5, sticky=tk.W)

        row = 1
        for elem in sorted([e for e in self.app.selected_elements if e != 'Fe']):
            ttk.Label(elem_frame, text=elem).grid(row=row, column=0, padx=5, pady=3, sticky=tk.W)

            ranges = self.app.element_ranges.get(elem, {'start': 0.1, 'end': 0.1, 'step': 0.1})
            start_var = tk.DoubleVar(value=ranges['start'])
            end_var = tk.DoubleVar(value=ranges['end'])
            step_var = tk.DoubleVar(value=ranges['step'])

            ttk.Entry(elem_frame, textvariable=start_var).grid(row=row, column=1, padx=5, pady=3, sticky=tk.EW)
            ttk.Entry(elem_frame, textvariable=end_var).grid(row=row, column=2, padx=5, pady=3, sticky=tk.EW)
            ttk.Entry(elem_frame, textvariable=step_var).grid(row=row, column=3, padx=5, pady=3, sticky=tk.EW)

            self.elem_entries[elem] = {'start': start_var, 'end': end_var, 'step': step_var}
            row += 1

        # Configure grid weights for elem_frame
        cols, rows = elem_frame.grid_size()
        for c in range(cols):
            weight = 1 if c == 0 else 2  # Label column narrower, input columns wider
            elem_frame.grid_columnconfigure(c, weight=weight, uniform="x")

        # Temperature Range
        temp_frame = ttk.LabelFrame(main_frame, text="Temperature Range (K)", padding="10")
        temp_frame.pack(fill=tk.X, pady=5)

        ttk.Label(temp_frame, text="Start:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.temp_start_var = tk.DoubleVar(value=self.app.temp_start)
        ttk.Entry(temp_frame, textvariable=self.temp_start_var).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(temp_frame, text="End:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.temp_end_var = tk.DoubleVar(value=self.app.temp_end)
        ttk.Entry(temp_frame, textvariable=self.temp_end_var).grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(temp_frame, text="Step:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.temp_step_var = tk.DoubleVar(value=self.app.temp_step)
        ttk.Entry(temp_frame, textvariable=self.temp_step_var).grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        # Configure grid weights for temp_frame
        for c in range(4):
            weight = 1 if c % 2 == 0 else 2
            temp_frame.grid_columnconfigure(c, weight=weight, uniform="x")

        # Phases Selection
        phases_frame = ttk.LabelFrame(main_frame, text="Phases", padding="10")
        phases_frame.pack(fill=tk.X, pady=5)

        ttk.Label(phases_frame, text="Selected phases for calculation:").pack(anchor=tk.W, pady=(0, 10))

        self.phase_vars = {}
        for phase in ["FCC_A1", "BCC_A2", "CEMENTITE_D011", "FCC_A1#2", "HCP_A3"]:
            var = tk.BooleanVar(value=(phase in self.app.selected_phases))
            self.phase_vars[phase] = var
            ttk.Checkbutton(phases_frame, text=phase, variable=var).pack(anchor=tk.W, pady=2)

        # Results Options
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.X, pady=5)

        self.calc_fraction_var = tk.BooleanVar(value=self.app.calc_fraction)
        self.calc_composition_var = tk.BooleanVar(value=self.app.calc_composition)

        ttk.Checkbutton(results_frame, text="Phase Fraction", variable=self.calc_fraction_var).pack(anchor=tk.W, pady=3)
        ttk.Checkbutton(results_frame, text="Phase Composition", variable=self.calc_composition_var).pack(anchor=tk.W, pady=3)

        # Database settings
        db_frame = ttk.LabelFrame(main_frame, text="Database", padding="10")
        db_frame.pack(fill=tk.X, pady=5)

        ttk.Label(db_frame, text="Database:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.phase_database_var = tk.StringVar(value=self.app.phase_database)
        ttk.Entry(db_frame, textvariable=self.phase_database_var).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(db_frame, text="Workers:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.workers_var = tk.IntVar(value=self.app.workers)
        ttk.Spinbox(db_frame, from_=1, to=16, textvariable=self.workers_var, width=8).grid(row=0, column=3, padx=5, pady=5, sticky=tk.EW)

        # Configure grid weights for db_frame
        for c in range(4):
            weight = 1 if c % 2 == 0 else 2
            db_frame.grid_columnconfigure(c, weight=weight, uniform="x")

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
        """Save phase calculation configuration."""
        try:
            # Update element ranges
            for elem, entries in self.elem_entries.items():
                self.app.element_ranges[elem] = {
                    'start': entries['start'].get(),
                    'end': entries['end'].get(),
                    'step': entries['step'].get()
                }

            # Update temperature settings
            self.app.temp_start = self.temp_start_var.get()
            self.app.temp_end = self.temp_end_var.get()
            self.app.temp_step = self.temp_step_var.get()

            # Update selected phases
            self.app.selected_phases = [phase for phase, var in self.phase_vars.items() if var.get()]

            # Update results settings
            self.app.calc_fraction = self.calc_fraction_var.get()
            self.app.calc_composition = self.calc_composition_var.get()

            # Update database settings
            self.app.phase_database = self.phase_database_var.get()
            self.app.workers = self.workers_var.get()

            self.app.update_log("Phase calculation configuration saved.")
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")