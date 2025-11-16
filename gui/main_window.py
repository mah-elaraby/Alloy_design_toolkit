"""
Main application window - COMPLETE VERSION
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading

from gui.tabs.models_tab import ModelsTab
from gui.tabs.workflow_tab import WorkflowTab
from core.workflow_runner import WorkflowRunner
from gui.tabs.about_tab import AboutTab
from gui.dialogs.phase_dialog import PhaseDialog
from gui.dialogs.sfe_dialog import SFEDialog
from gui.dialogs.moo_dialog import MOODialog
from gui.dialogs.precipitation_dialog import PrecipitationDialog
from gui.dialogs.annealing_dialog import AnnealingDialog

class AlloyDesignWorkflow:
    """Main workflow application for automatic alloy design."""

    def __init__(self, root):
        self.root = root
        self.root.title("Alloy Design Toolkit - Automatic Workflow")

        # Initialize components
        self.workflow_runner = WorkflowRunner(self)
        self.init_default_parameters()
        self.setup_ui()

        # Auto-size main window to content
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_reqwidth(), self.root.winfo_reqheight())
        self.root.resizable(True, True)

    def init_default_parameters(self):
        """Initialize default parameters from config."""
        from config.settings import DEFAULT_PARAMETERS
        self.__dict__.update(DEFAULT_PARAMETERS)

    def setup_ui(self):
        """Build the main user interface."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs
        self.models_tab = ModelsTab(self, self.notebook)
        self.workflow_tab = WorkflowTab(self, self.notebook)
        self.about_tab = AboutTab(self, self.notebook)

        # Status bar
        self.create_status_bar()

    def create_status_bar(self):
        """Create status bar at bottom of window."""
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN, padding="2")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(
            status_frame,
            text="Ready",
            font=('Arial', 9)
        )
        self.status_label.pack(side=tk.LEFT)

        ttk.Label(
            status_frame,
            text="Developed by Mahmoud Elaraby",
            font=('Arial', 9, 'italic')
        ).pack(side=tk.RIGHT)

    # Dialog methods
    def open_phase_calculations(self):
        """Open phase calculations configuration dialog."""
        PhaseDialog(self)

    def open_sfe_settings(self):
        """Open SFE settings dialog."""
        SFEDialog(self)

    def open_moo_settings(self):
        """Open MOO settings dialog."""
        MOODialog(self)

    def open_precipitation_settings(self):
        """Open precipitation settings dialog."""
        PrecipitationDialog(self)

    def open_annealing_settings(self):
        """Open annealing settings dialog."""
        AnnealingDialog(self)

    def update_log(self, message):
        """Update the log display from worker thread."""
        self.root.after(0, lambda: self.update_log_ui(message))

    def update_log_ui(self, message):
        """Update log in UI thread."""
        self.workflow_tab.log_text.insert(tk.END, message + "\n")
        self.workflow_tab.log_text.see(tk.END)
        self.root.update_idletasks()

    def update_progress(self, value):
        """Update progress bar from worker thread."""
        self.root.after(0, lambda: self.workflow_tab.progress.config(value=value))

    def update_status(self, message):
        """Update status bar."""
        self.root.after(0, lambda: self.status_label.config(text=message))
