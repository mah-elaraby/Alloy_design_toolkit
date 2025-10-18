"""
Models tab for individual model execution
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os


class ModelsTab:
    """Tab for launching individual models independently."""

    def __init__(self, app, notebook):
        self.app = app
        self.setup_ui(notebook)

    def setup_ui(self, notebook):
        """Create the Models tab UI."""
        self.frame = ttk.Frame(notebook, padding="20")
        notebook.add(self.frame, text="Standalone_scripts")

        # Title
        ttk.Label(
            self.frame,
            text="Individual Model Execution",
            font=('Arial', 14, 'bold')
        ).pack(pady=10)

        # Instructions
        ttk.Label(
            self.frame,
            text="Click on a button to run the corresponding model independently:",
            font=('Arial', 11, 'italic')
        ).pack(pady=(0, 20))

        # Model buttons
        models_info = [
            ("Phase composition and fraction", "1 Phase fraction and composition.py"),
            ("Martensite - Retained austenite", "2 retained austenite model.py"),
            ("Stacking fault energy", "3 SFE model.py"),
            ("Multi-objective optimization", "4 MOO algorithm.py"),
            ("Precipitation kinetics", "5 Precipitation kinetics.py"),
            ("PRISMA GM sorting", "6 PRISMA GM sorting.py")
        ]

        button_frame = ttk.Frame(self.frame)
        button_frame.pack(pady=20)

        for i, (name, script) in enumerate(models_info):
            row = i
            col = 0
            ttk.Button(
                button_frame,
                text=name,
                width=30,
                command=lambda s=script: self.launch_model(s)
            ).grid(row=row, column=col, padx=10, pady=5)

    def launch_model(self, script_name):
        """Launch individual model script."""
        script_path = os.path.join('Standalone_scripts', script_name)

        if not os.path.exists(script_path):
            messagebox.showerror(
                "Script Not Found",
                f"The script '{script_name}' was not found in the standalone_scripts directory."
            )
            return

        try:
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )

            messagebox.showinfo(
                "Model Launched",
                f"'{script_name}' has been launched in a new window."
            )

        except Exception as e:
            messagebox.showerror(
                "Launch Error",
                f"Failed to launch '{script_name}':\n{str(e)}"
            )