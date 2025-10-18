#!/usr/bin/env python3
"""
Multi-Objective Optimization for Medium Manganese Steel Alloy Design
Using NSGA-II-inspired sorting mechanisms for discrete data
Based on:
- NSGA-II algorithm (Deb et al. 2002)
- Discrete MOO for materials HTS
- Four objectives: Ra, -Ms, SFE, ΔT (process window width)
Author: Generated for Materials Science Research
Python Version: 3.6+
"""
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
from typing import List, Tuple, Dict, Optional


class MediumMnSteelMOO:
    """
    Multi-objective optimization class for medium manganese steel alloy design.
    Implements NSGA-II sorting mechanisms for discrete data.
    """

    def __init__(self):
        self.data = None
        self.filtered_data = None
        self.pareto_results = None
        self.constraints = {
            'Ms_min': -50.0,
            'Ms_max': 0.0,
            'Ra_min': 0.4,
            'Ra_max': 0.55,
            'SFE_min': 15.0,
            'SFE_max': 25.0,
            'min_deltaT': 10.0,
            'Cementite_max': 0.0
        }
        self.composition_cols = ['C', 'Mn', 'Si', 'Al', 'Mo', 'Nb', 'V']
        self.objective_names = ['J1_Ra', 'J2_neg_Ms', 'J3_SFE', 'J4_deltaT']

    def load_excel_data(self, filepath: str, sheet_name: str = "Data with 0 cementite") -> bool:
        """
        Load Excel data from specified sheet.
        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name to load (default: "Data with 0 cementite")
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.data = pd.read_excel(filepath, sheet_name=sheet_name)
            # Ensure numeric types for key columns
            numeric_cols = self.composition_cols + ['Temperature', 'Ms', 'Ra', 'SFE', 'Fm', 'Cementite', 'Gamma']
            for col in numeric_cols:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            # Drop rows with NaN values in critical columns
            critical_cols = self.composition_cols + ['Temperature', 'Ms', 'Ra', 'SFE', 'Fm', 'Cementite']
            self.data = self.data.dropna(subset=critical_cols)
            print(f"Loaded {len(self.data)} rows from {sheet_name}")
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False

    def compute_process_window(self, group_data: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Compute process window (ΔT) for a composition group.
        Args:
            group_data: DataFrame with filtered data for one composition
        Returns:
            Tuple[T_min, T_max, deltaT]: Process window boundaries and width
        """
        if len(group_data) == 0:
            return 0.0, 0.0, 0.0
        temps = sorted(group_data['Temperature'].unique())
        if len(temps) == 1:
            # Single temperature point
            return temps[0], temps[0], 5.0  # Assume 5°C step for single point
        # Find contiguous temperature ranges or use overall span
        temp_step = 5.0  # Assume 5°C steps based on manuscript
        # For simplicity, use overall span (can be modified for contiguous ranges)
        T_min = min(temps)
        T_max = max(temps)
        deltaT = T_max - T_min
        return T_min, T_max, deltaT

    def select_optimal_temperature(self, group_data: pd.DataFrame,
                                   T_min: float, T_max: float) -> Tuple[float, dict]:
        """
        Pick T_opt as the accepted temperature closest to the midpoint (T_min+T_max)/2.
        Ties are broken by choosing the higher temperature.
        """
        if len(group_data) == 0:
            return 0.0, {}

        T_center = 0.5 * (T_min + T_max)

        # Distance from each accepted setpoint to the window center
        diffs = (group_data['Temperature'] - T_center).abs()
        min_diff = diffs.min()

        # All candidates equally close to the center
        candidates = group_data.loc[diffs == min_diff]
        # Tie-break: choose the higher temperature
        row = candidates.sort_values('Temperature', ascending=False).iloc[0]

        T_opt = float(row['Temperature'])
        objectives = {
            'J1_Ra': float(row['Ra']),
            'J2_neg_Ms': float(-row['Ms']),
            'J3_SFE': float(row['SFE']),
            'T_opt': T_opt
        }
        return T_opt, objectives

    def filter_and_process_compositions(self) -> pd.DataFrame:
        """
        Filter compositions based on constraints and compute objectives.
        Returns:
            DataFrame with valid compositions and their objectives
        """
        if self.data is None:
            return pd.DataFrame()
        valid_compositions = []
        # Group by composition
        grouped = self.data.groupby(self.composition_cols)
        for comp_values, group_data in grouped:
            # Apply constraints: Fm==0, Cementite<=max, and user-defined ranges
            filtered_group = group_data[
                (group_data['Fm'] == 0) &
                (group_data['Cementite'] <= self.constraints['Cementite_max']) &
                (group_data['Ms'] >= self.constraints['Ms_min']) &
                (group_data['Ms'] <= self.constraints['Ms_max']) &
                (group_data['Ra'] >= self.constraints['Ra_min']) &
                (group_data['Ra'] <= self.constraints['Ra_max']) &
                (group_data['SFE'] >= self.constraints['SFE_min']) &
                (group_data['SFE'] <= self.constraints['SFE_max'])
                ]
            if len(filtered_group) == 0:
                continue
            # Compute process window
            T_min, T_max, deltaT = self.compute_process_window(filtered_group)
            if deltaT < self.constraints['min_deltaT']:
                continue
            # Select optimal temperature
            T_opt, objectives = self.select_optimal_temperature(filtered_group, T_min, T_max)
            if not objectives:
                continue
            # Create composition record
            comp_record = {f'{col}': comp_values[i] for i, col in enumerate(self.composition_cols)}
            comp_record.update({
                'T_min': T_min,
                'T_max': T_max,
                'J4_deltaT': deltaT,
                'T_opt': T_opt,
                'J1_Ra': objectives['J1_Ra'],
                'J2_neg_Ms': objectives['J2_neg_Ms'],
                'J3_SFE': objectives['J3_SFE']
            })
            valid_compositions.append(comp_record)
        return pd.DataFrame(valid_compositions)

    def dominates(self, solution_a: np.ndarray, solution_b: np.ndarray) -> bool:
        """
        Check if solution A dominates solution B (for maximization).
        Args:
            solution_a, solution_b: Arrays of objective values
        Returns:
            bool: True if A dominates B
        """
        # For maximization: A dominates B if A >= B in all objectives and > in at least one
        return np.all(solution_a >= solution_b) and np.any(solution_a > solution_b)

    def non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
        """
        Perform non-dominated sorting to assign ranks.
        Args:
            objectives: 2D array of objective values (n_solutions x n_objectives)
        Returns:
            List of fronts, where each front contains indices of solutions
        """
        n_solutions = len(objectives)
        dominated_solutions = [[] for _ in range(n_solutions)]  # Solutions dominated by i
        domination_count = np.zeros(n_solutions, dtype=int)  # Number of solutions dominating i
        # Compute domination relationships
        for i in range(n_solutions):
            for j in range(i + 1, n_solutions):
                if self.dominates(objectives[i], objectives[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self.dominates(objectives[j], objectives[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1
        # Find first front (non-dominated solutions)
        fronts = []
        first_front = []
        for i in range(n_solutions):
            if domination_count[i] == 0:
                first_front.append(i)
        fronts.append(first_front)
        # Find subsequent fronts
        while len(fronts[-1]) > 0:
            next_front = []
            for i in fronts[-1]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if len(next_front) > 0:
                fronts.append(next_front)
            else:
                break
        return fronts[:-1] if len(fronts) > 1 and len(fronts[-1]) == 0 else fronts

    def crowding_distance(self, objectives: np.ndarray, front_indices: List[int]) -> np.ndarray:
        """
        Compute crowding distance for solutions in a front.
        Args:
            objectives: 2D array of objective values
            front_indices: Indices of solutions in this front
        Returns:
            Array of crowding distances for each solution in the front
        """
        n_solutions = len(front_indices)
        n_objectives = objectives.shape[1]
        if n_solutions <= 2:
            return np.full(n_solutions, np.inf)
        distances = np.zeros(n_solutions)
        front_objectives = objectives[front_indices]
        # Normalize objectives for crowding distance calculation
        obj_min = np.min(front_objectives, axis=0)
        obj_max = np.max(front_objectives, axis=0)
        obj_range = obj_max - obj_min
        # Avoid division by zero
        obj_range = np.where(obj_range == 0, 1.0, obj_range)
        for obj_idx in range(n_objectives):
            # Sort solutions by this objective
            sorted_indices = np.argsort(front_objectives[:, obj_idx])
            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf
            # Compute distances for intermediate solutions
            for i in range(1, n_solutions - 1):
                if distances[sorted_indices[i]] != np.inf:
                    distance = (front_objectives[sorted_indices[i + 1], obj_idx] -
                                front_objectives[sorted_indices[i - 1], obj_idx]) / obj_range[obj_idx]
                    distances[sorted_indices[i]] += distance
        return distances

    def compute_geometric_mean_ranking(self, compositions_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Compute geometric mean (GM) ranking for the top N alloys.
        Args:
            compositions_df: DataFrame with results including objectives
            top_n: Number of top alloys to apply GM ranking to
        Returns:
            DataFrame with GM ranking and normalized objectives added
        """
        if len(compositions_df) == 0:
            return compositions_df
        result_df = compositions_df.copy()

        # Get top N alloys (first N rows from NSGA-II sorted results)
        top_alloys = result_df.head(top_n).copy()

        if len(top_alloys) == 0:
            # No top alloys, assign NaN GM_Rank
            result_df['GM_Score'] = np.nan
            result_df['GM_Rank'] = np.nan
            # Initialize normalized objective columns with NaN
            for obj_name in self.objective_names:
                result_df[f'Norm_{obj_name}'] = np.nan
            return result_df

        # Extract objectives for top N alloys
        objectives = top_alloys[self.objective_names].values
        # Normalize objectives to [0, 1] range for geometric mean calculation
        # Find global min/max across all top N alloys
        obj_min = np.min(objectives, axis=0)
        obj_max = np.max(objectives, axis=0)
        obj_range = obj_max - obj_min
        # Avoid division by zero
        obj_range = np.where(obj_range == 0, 1.0, obj_range)
        # Normalize to [0, 1] range
        normalized_objectives = (objectives - obj_min) / obj_range
        # Add small epsilon to avoid zero values in geometric mean
        epsilon = 1e-6
        normalized_objectives_for_gm = normalized_objectives + epsilon
        # Compute geometric mean for each solution
        # GM = (x1 * x2 * x3 * x4)^(1/4)
        geometric_means = np.exp(np.mean(np.log(normalized_objectives_for_gm), axis=1))

        # Initialize GM columns for all rows
        result_df['GM_Score'] = np.nan
        result_df['GM_Rank'] = np.nan

        # Initialize normalized objective columns for all rows
        for obj_name in self.objective_names:
            result_df[f'Norm_{obj_name}'] = np.nan

        # Assign GM scores and normalized objectives to top N alloys
        top_indices = top_alloys.index
        result_df.loc[top_indices, 'GM_Score'] = np.round(geometric_means, 6)

        # Assign normalized objective values (without epsilon) to display
        for i, obj_name in enumerate(self.objective_names):
            result_df.loc[top_indices, f'Norm_{obj_name}'] = np.round(normalized_objectives[:, i], 4)

        # Rank by GM
        result_df.loc[top_indices, 'GM_Rank'] = np.argsort(-geometric_means) + 1  # Start ranking from 1

        # Re-sort the dataframe: top N alloys sorted by GM rank, then remaining alloys
        top_gm_sorted = result_df.loc[top_indices].sort_values('GM_Rank', ascending=True)
        remaining_alloys = result_df.drop(top_indices)
        result_df = pd.concat([top_gm_sorted, remaining_alloys], ignore_index=True)

        return result_df

    def nsga_ii_sort(self, compositions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply NSGA-II sorting (non-dominated sorting + crowding distance).
        Args:
            compositions_df: DataFrame with valid compositions and objectives
        Returns:
            DataFrame sorted by NSGA-II criteria with rank and crowding distance
        """
        if len(compositions_df) == 0:
            return compositions_df
        # Extract objectives matrix
        objectives = compositions_df[self.objective_names].values
        # Perform non-dominated sorting
        fronts = self.non_dominated_sort(objectives)
        # Assign ranks and compute crowding distances
        ranks = np.zeros(len(compositions_df))
        crowding_distances = np.zeros(len(compositions_df))
        for rank, front in enumerate(fronts):
            ranks[front] = rank + 1  # Rank starts from 1
            if len(front) > 0:
                distances = self.crowding_distance(objectives, front)
                crowding_distances[front] = distances
        # Add rank and crowding distance to dataframe
        result_df = compositions_df.copy()
        result_df['Rank'] = ranks.astype(int)
        result_df['Crowding_Distance'] = np.round(crowding_distances, 4)
        # Sort by rank (ascending) then by crowding distance (descending)
        result_df = result_df.sort_values(['Rank', 'Crowding_Distance'],
                                          ascending=[True, False]).reset_index(drop=True)
        return result_df

    def process_moo(self) -> pd.DataFrame:
        """
        Main processing function for multi-objective optimization.
        Now automatically applies GM ranking to the first 20 alloys.
        Returns:
            DataFrame with ranked Pareto solutions including GM ranking for top 20
        """
        # Filter and process compositions
        print("Filtering compositions and computing objectives...")
        valid_compositions = self.filter_and_process_compositions()
        if len(valid_compositions) == 0:
            print("No valid compositions found!")
            return pd.DataFrame()
        print(f"Found {len(valid_compositions)} valid compositions")

        # Apply NSGA-II sorting
        print("Applying NSGA-II sorting...")
        ranked_results = self.nsga_ii_sort(valid_compositions)

        # Automatically apply GM ranking to first 20 alloys
        print("Applying GM ranking to top 20 alloys...")
        ranked_results = self.compute_geometric_mean_ranking(ranked_results, top_n=20)

        # Round numerical values for display
        numeric_columns = ['T_min', 'T_max', 'T_opt', 'J1_Ra', 'J2_neg_Ms', 'J3_SFE', 'J4_deltaT']
        for col in numeric_columns:
            if col in ranked_results.columns:
                ranked_results[col] = np.round(ranked_results[col], 2)

        # Report GM ranking results
        gm_count = len(ranked_results.dropna(subset=['GM_Rank']))
        print(f"GM ranking applied to {gm_count} alloys")

        return ranked_results

    def plot_pareto_front(self, results_df: pd.DataFrame, max_rank: int = 3):
        """
        Plot 2D projections of Pareto front.
        Args:
            results_df: Results dataframe with rankings
            max_rank: Maximum rank to include in plot
        """
        if len(results_df) == 0:
            return
        # Filter to first few ranks
        plot_data = results_df[results_df['Rank'] <= max_rank]
        if len(plot_data) == 0:
            return
        # Create 2x3 subplot for different objective pairs
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Pareto Front Projections - Medium Mn Steel MOO', fontsize=16)
        objective_pairs = [
            ('J1_Ra', 'J2_neg_Ms'),
            ('J1_Ra', 'J3_SFE'),
            ('J1_Ra', 'J4_deltaT'),
            ('J2_neg_Ms', 'J3_SFE'),
            ('J2_neg_Ms', 'J4_deltaT'),
            ('J3_SFE', 'J4_deltaT')
        ]
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for idx, (obj1, obj2) in enumerate(objective_pairs):
            ax = axes[idx // 3, idx % 3]
            for rank in range(1, max_rank + 1):
                rank_data = plot_data[plot_data['Rank'] == rank]
                if len(rank_data) > 0:
                    ax.scatter(rank_data[obj1], rank_data[obj2],
                               c=colors[rank - 1], label=f'Rank {rank}', alpha=0.7)
            ax.set_xlabel(obj1)
            ax.set_ylabel(obj2)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class MnSteelMOOGUI:
    """GUI application for Medium Manganese Steel MOO."""

    def __init__(self):
        self.moo = MediumMnSteelMOO()
        self.results = None
        self.gm_sorted = False  # Track if results are GM sorted
        # Create main window
        self.root = tk.Tk()
        self.root.title("Medium Mn Steel Multi-Objective Optimization")
        self.root.geometry("1600x800")  # Increased width to accommodate new columns
        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI components."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        # File loading section
        file_frame = ttk.LabelFrame(main_frame, text="Data Loading", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(file_frame, text="Load Excel File",
                   command=self.load_file).grid(row=0, column=0, padx=5)
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.grid(row=0, column=1, padx=10, sticky=tk.W)
        # Constraints section
        constraints_frame = ttk.LabelFrame(main_frame, text="Alloy design constraints", padding="5")
        constraints_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        # Create constraint input fields
        self.constraint_vars = {}
        constraints = [
            ('Ms_min', 'Ms min (°C)', -50.0),
            ('Ms_max', 'Ms max (°C)', 0.0),
            ('Ra_min', 'RA min', 0.4),
            ('Ra_max', 'RA max', 0.55),
            ('SFE_min', 'SFE min (mJ/m²)', 15.0),
            ('SFE_max', 'SFE max (mJ/m²)', 25.0),
            ('min_deltaT', 'Min ΔT (°C)', 10.0),
            ('Cementite_max', 'Cementite limit', 0.0)
        ]
        for i, (key, label, default) in enumerate(constraints):
            row = i // 4
            col = (i % 4) * 2
            ttk.Label(constraints_frame, text=label).grid(row=row, column=col, padx=5, pady=2, sticky=tk.W)
            var = tk.DoubleVar(value=default)
            self.constraint_vars[key] = var
            entry = ttk.Entry(constraints_frame, textvariable=var, width=10)
            entry.grid(row=row, column=col + 1, padx=5, pady=2)
        # Processing section
        process_frame = ttk.Frame(main_frame)
        process_frame.grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(process_frame, text="Process MOO (Auto GM Ranking)",
                   command=self.process_moo).grid(row=0, column=0, padx=5)
        # Remove GM Ranking button since it's now automatic
        ttk.Button(process_frame, text="Plot Pareto Front",
                   command=self.plot_results).grid(row=0, column=1, padx=5)
        ttk.Button(process_frame, text="Export Results",
                   command=self.export_results).grid(row=0, column=2, padx=5)
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=5)
        # Results table
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        # Create Treeview with scrollbars
        self.tree_frame = ttk.Frame(results_frame)
        self.tree_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.tree_frame.columnconfigure(0, weight=1)
        self.tree_frame.rowconfigure(0, weight=1)
        # Updated Treeview columns to include GM ranking and normalized objectives
        columns = ['Rank', 'GM_Rank', 'GM_Score', 'Norm_J1_Ra', 'Norm_J2_neg_Ms', 'Norm_J3_SFE', 'Norm_J4_deltaT',
                   'Crowding_Distance', 'C', 'Mn', 'Al', 'Mo', 'Nb', 'V',
                   'T_opt', 'J1_Ra', 'J2_neg_Ms', 'J3_SFE', 'J4_deltaT']
        self.tree = ttk.Treeview(self.tree_frame, columns=columns, show='headings', height=15)
        # Configure column headings and widths
        column_widths = {
            'Rank': 50, 'GM_Rank': 70, 'GM_Score': 80,
            'Norm_J1_Ra': 80, 'Norm_J2_neg_Ms': 90, 'Norm_J3_SFE': 80, 'Norm_J4_deltaT': 90,
            'Crowding_Distance': 100,
            'C': 60, 'Mn': 60, 'Al': 60, 'Mo': 60, 'Nb': 60, 'V': 60,
            'T_opt': 80, 'J1_Ra': 80, 'J2_neg_Ms': 80, 'J3_SFE': 80, 'J4_deltaT': 80
        }

        column_headers = {
            'Rank': 'Rank', 'GM_Rank': 'GM Rank', 'GM_Score': 'GM Score',
            'Norm_J1_Ra': 'Norm Ra', 'Norm_J2_neg_Ms': 'Norm -Ms',
            'Norm_J3_SFE': 'Norm SFE', 'Norm_J4_deltaT': 'Norm ΔT',
            'Crowding_Distance': 'Crowding Dist',
            'C': 'C', 'Mn': 'Mn', 'Al': 'Al', 'Mo': 'Mo', 'Nb': 'Nb', 'V': 'V',
            'T_opt': 'T_opt', 'J1_Ra': 'Ra', 'J2_neg_Ms': '-Ms', 'J3_SFE': 'SFE', 'J4_deltaT': 'ΔT'
        }

        for col in columns:
            self.tree.heading(col, text=column_headers.get(col, col), command=lambda c=col: self.sort_treeview(c))
            self.tree.column(col, width=column_widths.get(col, 80), anchor=tk.CENTER)
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        # Grid layout
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))

    def load_file(self):
        """Load Excel file dialog."""
        filepath = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filepath:
            if self.moo.load_excel_data(filepath):
                self.file_label.config(text=f"Loaded: {os.path.basename(filepath)}")
                self.status_label.config(text=f"Data loaded: {len(self.moo.data)} rows")
            else:
                messagebox.showerror("Error", "Failed to load Excel file")

    def process_moo(self):
        """Process multi-objective optimization with automatic GM ranking."""
        if self.moo.data is None:
            messagebox.showerror("Error", "Please load data first")
            return
        # Update constraints
        for key, var in self.constraint_vars.items():
            self.moo.constraints[key] = var.get()
        self.status_label.config(text="Processing MOO with automatic GM ranking...")
        self.root.update()
        try:
            self.results = self.moo.process_moo()
            if len(self.results) == 0:
                messagebox.showwarning("Warning", "No valid solutions found with current constraints")
                self.status_label.config(text="No solutions found")
                return
            # GM ranking is now applied automatically
            self.gm_sorted = True
            self.update_results_table()
            rank_1_count = len(self.results[self.results['Rank'] == 1])
            gm_count = len(self.results.dropna(subset=['GM_Rank']))
            self.status_label.config(
                text=f"Found {len(self.results)} solutions, {rank_1_count} in Pareto front, GM ranking applied to top {gm_count}")
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            self.status_label.config(text="Processing failed")

    def update_results_table(self):
        """Update the results table with current results."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        if self.results is None or len(self.results) == 0:
            return
        # Display top 113 results to show more solutions
        display_results = self.results.head(113)
        for _, row in display_results.iterrows():
            # Format GM values - check if columns exist first
            if 'GM_Rank' in row and not pd.isna(row['GM_Rank']):
                gm_rank = int(row['GM_Rank'])
            else:
                gm_rank = '-'
            if 'GM_Score' in row and not pd.isna(row['GM_Score']):
                gm_score = f"{row['GM_Score']:.4f}"
            else:
                gm_score = '-'
            # Format normalized objective values
            norm_values = []
            for obj_name in self.moo.objective_names:
                norm_col = f'Norm_{obj_name}'
                if norm_col in row and not pd.isna(row[norm_col]):
                    norm_values.append(f"{row[norm_col]:.3f}")
                else:
                    norm_values.append('-')
            values = [
                int(row['Rank']),
                gm_rank,
                gm_score,
                norm_values[0],  # Norm_J1_Ra
                norm_values[1],  # Norm_J2_neg_Ms
                norm_values[2],  # Norm_J3_SFE
                norm_values[3],  # Norm_J4_deltaT
                f"{row['Crowding_Distance']:.4f}",
                f"{row['C']:.3f}",
                f"{row['Mn']:.2f}",
                f"{row['Al']:.3f}",
                f"{row['Mo']:.3f}",
                f"{row['Nb']:.4f}",
                f"{row['V']:.4f}",
                f"{row['T_opt']:.0f}",
                f"{row['J1_Ra']:.3f}",
                f"{row['J2_neg_Ms']:.1f}",
                f"{row['J3_SFE']:.1f}",
                f"{row['J4_deltaT']:.1f}"
            ]
            # Color code rows with GM ranking
            item = self.tree.insert('', 'end', values=values)
            if row['Rank'] == 1:
                self.tree.set(item, 'Rank', f"⭐ {int(row['Rank'])}")
            # Highlight GM ranked alloys
            if not pd.isna(row.get('GM_Rank', np.nan)):
                # Add a special marker for GM ranked alloys
                current_gm_rank = self.tree.item(item)['values'][1]
                if current_gm_rank != '-':
                    self.tree.set(item, 'GM_Rank', f"🏆 {current_gm_rank}")

    def sort_treeview(self, col):
        """Sort treeview by column."""
        if not self.tree.get_children():
            return
        # Get data and sort
        data = [(self.tree.item(child)["values"], child) for child in self.tree.get_children()]
        try:
            # Get column index
            columns = ['Rank', 'GM_Rank', 'GM_Score', 'Norm_J1_Ra', 'Norm_J2_neg_Ms', 'Norm_J3_SFE', 'Norm_J4_deltaT',
                       'Crowding_Distance', 'C', 'Mn', 'Al', 'Mo', 'Nb', 'V',
                       'T_opt', 'J1_Ra', 'J2_neg_Ms', 'J3_SFE', 'J4_deltaT']
            col_index = columns.index(col)
            # Handle special sorting for GM_Rank and normalized columns (convert '-' to high number for sorting)
            if col in ['GM_Rank'] + [f'Norm_{obj}' for obj in self.moo.objective_names]:
                def sort_key(x):
                    val = str(x[0][col_index])
                    # Remove special markers
                    if '🏆' in val:
                        val = val.replace('🏆 ', '')
                    if val == '-':
                        return float('inf')
                    try:
                        return float(val)
                    except:
                        return float('inf')

                data.sort(key=sort_key)
            else:
                # Try numeric sort first
                def sort_key(x):
                    val = str(x[0][col_index])
                    # Remove star emoji for Rank column
                    if col == 'Rank' and '⭐' in val:
                        val = val.replace('⭐ ', '')
                    try:
                        return float(val)
                    except:
                        return str(val)

                data.sort(key=sort_key)
        except (ValueError, IndexError):
            # Fall back to string sort
            data.sort(key=lambda x: str(x[0][col_index]) if col_index < len(x[0]) else '')
        # Reorder items
        for index, (values, child) in enumerate(data):
            self.tree.move(child, '', index)

    def plot_results(self):
        """Plot Pareto front."""
        if self.results is None or len(self.results) == 0:
            messagebox.showwarning("Warning", "No results to plot")
            return
        try:
            self.moo.plot_pareto_front(self.results)
        except Exception as e:
            messagebox.showerror("Error", f"Plotting failed: {str(e)}")

    def export_results(self):
        """Export results to Excel/CSV."""
        if self.results is None or len(self.results) == 0:
            messagebox.showwarning("Warning", "No results to export")
            return
        filepath = filedialog.asksaveasfilename(
            title="Save results",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            try:
                if filepath.endswith('.csv'):
                    self.results.to_csv(filepath, index=False)
                else:
                    # Export to Excel with multiple sheets
                    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                        # All results
                        self.results.to_excel(writer, sheet_name='All_Results', index=False)
                        # Pareto front only (Rank 1)
                        pareto_front = self.results[self.results['Rank'] == 1]
                        if len(pareto_front) > 0:
                            pareto_front.to_excel(writer, sheet_name='Pareto_Front', index=False)
                        # GM ranked alloys (top 20 with GM ranking)
                        gm_ranked = self.results.dropna(subset=['GM_Rank']).sort_values('GM_Rank')
                        if len(gm_ranked) > 0:
                            gm_ranked.to_excel(writer, sheet_name='GM_Ranked_Top20', index=False)
                        # Top 30 results
                        top_30 = self.results.head(30)
                        top_30.to_excel(writer, sheet_name='Top_30', index=False)
                messagebox.showinfo("Success", f"Results exported to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")

    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


class MOOBatchProcessor:
    """Batch processor for multiple MOO runs with different constraint sets."""

    def __init__(self, moo_instance: MediumMnSteelMOO):
        self.moo = moo_instance

    def run_sensitivity_analysis(self) -> Dict:
        """
        Run sensitivity analysis by varying constraints.
        Returns:
            Dictionary with results for different constraint sets
        """
        if self.moo.data is None:
            raise ValueError("No data loaded")
        # Define constraint variations
        constraint_sets = {
            'Base': {'Ms_min': -50, 'Ms_max': 0, 'Ra_min': 0.4, 'Ra_max': 0.55,
                     'SFE_min': 15, 'SFE_max': 25, 'min_deltaT': 10, 'Cementite_max': 0.0},
            'Relaxed_Ms': {'Ms_min': -70, 'Ms_max': 10, 'Ra_min': 0.4, 'Ra_max': 0.55,
                           'SFE_min': 15, 'SFE_max': 25, 'min_deltaT': 10, 'Cementite_max': 0.0},
            'Relaxed_Ra': {'Ms_min': -50, 'Ms_max': 0, 'Ra_min': 0.35, 'Ra_max': 0.6,
                           'SFE_min': 15, 'SFE_max': 25, 'min_deltaT': 10, 'Cementite_max': 0.0},
            'Tight_SFE': {'Ms_min': -50, 'Ms_max': 0, 'Ra_min': 0.4, 'Ra_max': 0.55,
                          'SFE_min': 18, 'SFE_max': 22, 'min_deltaT': 10, 'Cementite_max': 0.0},
            'Large_ProcessWindow': {'Ms_min': -50, 'Ms_max': 0, 'Ra_min': 0.4, 'Ra_max': 0.55,
                                    'SFE_min': 15, 'SFE_max': 25, 'min_deltaT': 20, 'Cementite_max': 0.0},
            'Relaxed_Cementite': {'Ms_min': -50, 'Ms_max': 0, 'Ra_min': 0.4, 'Ra_max': 0.55,
                                  'SFE_min': 15, 'SFE_max': 25, 'min_deltaT': 10, 'Cementite_max': 5.0}
        }
        results = {}
        for name, constraints in constraint_sets.items():
            print(f"Processing constraint set: {name}")
            # Update constraints
            self.moo.constraints.update(constraints)
            # Run MOO (now includes automatic GM ranking)
            result = self.moo.process_moo()
            # Store results with summary statistics
            results[name] = {
                'results': result,
                'n_solutions': len(result),
                'n_pareto': len(result[result['Rank'] == 1]) if len(result) > 0 else 0,
                'n_gm_ranked': len(result.dropna(subset=['GM_Rank'])) if len(result) > 0 else 0,
                'constraints': constraints.copy()
            }
            print(
                f"  Found {results[name]['n_solutions']} solutions, {results[name]['n_pareto']} in Pareto front, {results[name]['n_gm_ranked']} GM ranked")
        return results

    def export_sensitivity_results(self, results: Dict, filepath: str):
        """Export sensitivity analysis results to Excel."""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for name, data in results.items():
                summary_row = {
                    'Constraint_Set': name,
                    'N_Solutions': data['n_solutions'],
                    'N_Pareto': data['n_pareto'],
                    'N_GM_Ranked': data['n_gm_ranked'],
                    **data['constraints']
                }
                summary_data.append(summary_row)
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            # Individual result sheets
            for name, data in results.items():
                if len(data['results']) > 0:
                    sheet_name = name.replace('_', ' ')[:31]  # Excel sheet name limit
                    data['results'].to_excel(writer, sheet_name=sheet_name, index=False)


def main():
    """Main function to run the application."""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Run test with sample data
        print("Running MOO algorithm test...")
        # test_results = test_moo_algorithm()  # Commented out as function not provided
        return
    # Run GUI application
    print("Starting Medium Mn Steel MOO GUI with automatic GM ranking and Cementite filter...")
    app = MnSteelMOOGUI()
    app.run()


if __name__ == "__main__":
    main()