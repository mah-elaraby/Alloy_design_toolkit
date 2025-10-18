"""
Workflow tab implementation
"""

import tkinter as tk
from tkinter import ttk


class WorkflowTab:
    """Computational Workflow"""

    def __init__(self, app, notebook):
        self.app = app
        self.setup_ui(notebook)

    def setup_ui(self, notebook):
        """Create the workflow tab UI."""
        self.frame = ttk.Frame(notebook, padding="20")
        notebook.add(self.frame, text="Computational Workflow")

        # Title
        ttk.Label(
            self.frame,
            text="Automatic Workflow Configuration",
            font=('Arial', 14, 'bold')
        ).pack(pady=10)

        # Workflow steps
        self.create_workflow_steps()

        # Control buttons
        self.create_control_buttons()

        # Progress bar
        self.create_progress_bar()

        # Log display
        self.create_log_display()

    def create_workflow_steps(self):
        """Create workflow steps frame."""
        steps_frame = ttk.LabelFrame(self.frame, text="Workflow Steps", padding="15")
        steps_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        steps = [
            ("Phase Calculations", self.app.open_phase_calculations,
             "Configure phase calculations"),
            ("Stacking Fault Energy", self.app.open_sfe_settings,
             "Configure SFE calculation parameters"),
            ("Multi-Objective Optimization", self.app.open_moo_settings,
             "Set alloy design criteria"),
            ("Precipitation kinetics", self.app.open_precipitation_settings,
             "Configure precipitation kinetics parameters"),
            ("Optimal Annealing Time", self.app.open_annealing_settings,
             "Configure precipitation objectives")
        ]

        for i, (name, command, description) in enumerate(steps):
            step_frame = ttk.Frame(steps_frame)
            step_frame.pack(fill=tk.X, pady=5)

            ttk.Label(step_frame, text=f"{i + 1}.", width=3).pack(side=tk.LEFT)
            ttk.Button(
                step_frame,
                text=name,
                command=command,
                width=25
            ).pack(side=tk.LEFT, padx=5)
            ttk.Label(step_frame, text=description).pack(side=tk.LEFT, padx=10)

    def create_control_buttons(self):
        """Create control buttons frame."""
        control_frame = ttk.Frame(self.frame)
        control_frame.pack(fill=tk.X, pady=20)

        self.run_button = ttk.Button(
            control_frame,
            text="Run Workflow",
            command=self.app.workflow_runner.run_workflow,
            width=20
        )
        self.run_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop",
            command=self.app.workflow_runner.stop_workflow,
            width=15,
            state='disabled'
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

    def create_progress_bar(self):
        """Create progress bar."""
        self.progress = ttk.Progressbar(self.frame, maximum=100)
        self.progress.pack(fill=tk.X, pady=10)

    def create_log_display(self):
        """Create log display area."""
        log_frame = ttk.LabelFrame(self.frame, text="Workflow Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(
            log_frame,
            height=10,
            yscrollcommand=scrollbar.set,
            wrap=tk.WORD
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)