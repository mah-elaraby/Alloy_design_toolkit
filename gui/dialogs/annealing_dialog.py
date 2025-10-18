"""
Optimal annealing time settings configuration dialog
"""

import tkinter as tk
from tkinter import ttk, messagebox


class AnnealingDialog:
    """Dialog for configuring optimal annealing time settings."""

    def __init__(self, app):
        self.app = app
        self.window = None
        self.setup_ui()

    def setup_ui(self):
        """Create the optimal annealing time settings window."""
        self.window = tk.Toplevel(self.app.root)
        self.window.title("Optimal Annealing Time Settings")

        main_frame = ttk.Frame(self.window, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Objectives selection frame
        objectives_frame = ttk.LabelFrame(main_frame, text="Select objectives for ranking", padding="10")
        objectives_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        ttk.Label(
            objectives_frame,
            text="Select which objectives contribute to the Overall Score (Geometric Mean):",
            font=('Arial', 10),
            wraplength=600
        ).pack(anchor=tk.W, pady=(0, 15))

        # Create checkboxes for objectives
        self.gm_objective_vars = {}
        for obj_name, obj_config in self.app.gm_objectives.items():
            var = tk.BooleanVar(value=obj_config['selected'])
            self.gm_objective_vars[obj_name] = var

            text = f"{obj_name} ({obj_config['goal']})"
            ttk.Checkbutton(objectives_frame, text=text, variable=var).pack(anchor=tk.W, pady=3)

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
        """Save annealing time optimization configuration."""
        try:
            for obj_name, var in self.gm_objective_vars.items():
                self.app.gm_objectives[obj_name]['selected'] = var.get()

            self.app.update_log("Annealing optimization configuration saved.")
            self.window.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")