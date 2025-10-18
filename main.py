#!/usr/bin/env python3
"""
Automatic Alloy Design Workflow GUI - MODULAR VERSION
Main entry point
"""

import sys
import os
import tkinter as tk

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from gui.main_window import AlloyDesignWorkflow

def main():
    """Launch the alloy design workflow application."""
    if sys.platform.startswith("win"):
        import multiprocessing as mp
        mp.freeze_support()

    root = tk.Tk()
    app = AlloyDesignWorkflow(root)
    root.mainloop()

if __name__ == "__main__":
    main()