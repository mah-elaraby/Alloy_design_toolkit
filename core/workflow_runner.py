"""
Workflow execution engine - UPDATED VERSION
"""

import threading
import pandas as pd
import numpy as np

from core.phase_calculator import PhaseCalculator
from core.martensite_calculator import MartensiteCalculator
from core.sfe_calculator import SFECalculator
from core.moo_optimizer import MOOOptimizer
from core.precipitation_calculator import PrecipitationCalculator
from core.gm_sorter import GMSorter


class WorkflowRunner:
    """Orchestrates the execution of the complete workflow."""

    def __init__(self, app):
        self.app = app
        self.stop_requested = False
        self.current_thread = None

        # Initialize calculators
        self.phase_calculator = PhaseCalculator(app)
        self.martensite_calculator = MartensiteCalculator(app)
        self.sfe_calculator = SFECalculator(app)
        self.moo_optimizer = MOOOptimizer(app)
        self.precipitation_calculator = PrecipitationCalculator(app)
        self.gm_sorter = GMSorter(app)

        # Store intermediate results
        self.workflow_data = {
            'phase_results': None,
            'sfe_results': None,
            'moo_results': None,
            'precipitation_results': None,
            'precipitation_timeseries': None,  # Add this for time-series data
            'gm_sorted_results': None
        }

    def run_workflow(self):
        """Execute the complete automatic workflow."""
        if self.current_thread and self.current_thread.is_alive():
            self.app.update_log("Workflow is already running.")
            return False

        self.stop_requested = False
        self.current_thread = threading.Thread(target=self.execute_workflow, daemon=True)
        self.current_thread.start()
        return True

    def execute_workflow(self):
        """Execute the workflow steps sequentially."""
        try:
            total_steps = 6

            # Step 1: Phase Calculations
            self.app.update_log("Step 1/6: Running phase calculations...")
            self.app.update_progress(0)

            phase_results = self.phase_calculator.calculate()
            if self.stop_requested or phase_results is None:
                return
            self.workflow_data['phase_results'] = phase_results
            self.app.update_progress(100 / total_steps)

            # Step 2: Martensite/Austenite Calculations
            self.app.update_log("Step 2/6: Calculating martensite and retained austenite...")
            ma_results = self.martensite_calculator.calculate(phase_results)
            if self.stop_requested or ma_results is None:
                return
            self.app.update_progress(200 / total_steps)

            # Step 3: SFE Calculations
            self.app.update_log("Step 3/6: Calculating stacking fault energy...")
            sfe_results = self.sfe_calculator.calculate(ma_results)
            if self.stop_requested or sfe_results is None:
                return
            self.workflow_data['sfe_results'] = sfe_results
            self.app.update_progress(300 / total_steps)

            # Step 4: Multi-objective Optimization
            self.app.update_log("Step 4/6: Running multi-objective optimization...")
            moo_results = self.moo_optimizer.optimize(sfe_results)
            if self.stop_requested or moo_results is None:
                return
            self.workflow_data['moo_results'] = moo_results
            self.app.update_progress(400 / total_steps)

            # Step 5: Precipitation Kinetics
            self.app.update_log("Step 5/6: Calculating precipitation kinetics...")
            precip_results = self.precipitation_calculator.calculate(moo_results)
            if self.stop_requested or precip_results is None:
                return
            self.workflow_data['precipitation_results'] = precip_results
            self.app.update_progress(500 / total_steps)

            # Step 6: Optimal Annealing Time (GM Sorting)
            self.app.update_log("Step 6/6: Determining optimal annealing time...")

            # Pass the complete precipitation time-series data to GM sorter
            annealing_results = self.gm_sorter.optimize_annealing(precip_results)
            if self.stop_requested or annealing_results is None:
                return

            # Store the complete GM-sorted precipitation data
            self.workflow_data['gm_sorted_results'] = annealing_results
            self.app.update_progress(100)

            # Save final results
            self.save_results()
            self.app.update_log("\nWorkflow completed successfully!")

        except Exception as e:
            self.app.update_log(f"ERROR: {str(e)}")
            import traceback
            self.app.update_log(traceback.format_exc())

    def save_results(self):
        """Save all workflow results to Excel."""
        from utils.excel_handler import save_workflow_results
        save_workflow_results(self.workflow_data)

    def stop_workflow(self):
        """Stop the running workflow."""
        self.stop_requested = True
        self.app.update_log("Stop requested. Waiting for current step to complete...")