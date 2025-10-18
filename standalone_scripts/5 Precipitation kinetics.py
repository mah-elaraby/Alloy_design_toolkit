"""
PRISMA Precipitation Calculator GUI
====================================

A simple graphical interface for running isothermal precipitation calculations
on multi-component Fe-based alloys using TC-Python PRISMA module.

Features:
- Load alloy compositions from Excel file
- Configure precipitation parameters (temperature, time, phases)
- Select alloying elements from periodic table
- Run isothermal precipitation calculations
- Export detailed results to Excel (volume fraction, mean radius, composition)
- Real-time progress tracking

Requirements:
- TC-Python with PRISMA module
- tkinter (standard Python GUI library)
- pandas, openpyxl for Excel handling


"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import threading
import pandas as pd

# Import TC-Python libraries for precipitation calculations
try:
    from tc_python import TCPython
    from tc_python.precipitation import (
        MatrixPhase,
        PrecipitatePhase,
        CompositionUnit,
        GrowthRateModel,
    )
    TC_PYTHON_AVAILABLE = True
except ImportError:
    TC_PYTHON_AVAILABLE = False
    print("Warning: TC-Python not found. Calculations will not work without it.")


class PrismaCalculatorGUI:
    """
    Main GUI application for PRISMA precipitation calculations.

    This interface allows users to configure and run isothermal precipitation
    calculations for Fe-based alloys, reading compositions from Excel files
    and exporting detailed time-dependent results.
    """

    # Default elements
    DEFAULT_ELEMENTS = ["Fe", "C", "Mn", "Si", "Al", "Mo", "Nb", "V"]

    # Available elements for selection (elements shown in black in periodic table)
    AVAILABLE_ELEMENTS = {
        'B': (1, 12), 'C': (1, 13), 'N': (1, 14),
        'O': (1, 15), 'Mg': (2, 1), 'Al': (2, 12), 'Si': (2, 13), 'P': (2, 14),
        'S': (2, 15), 'Ca': (3, 1), 'Sc': (3, 2), 'Ti': (3, 3), 'V': (3, 4),
        'Cr': (3, 5), 'Mn': (3, 6), 'Fe': (3, 7), 'Co': (3, 8), 'Ni': (3, 9),
        'Cu': (3, 10), 'Zn': (3, 11), 'Nb': (4, 4),
        'Mo': (4, 5), 'Cs': (5, 0), 'Ta': (5, 4), 'W': (5, 5)

    }

    GROWTH_MODELS = [
        "Simplified",
        "General",
        "Advanced",
        "Para_eq",
        "NPLE",
        "PE_AUTOMATIC"
    ]

    NUCLEATION_SITES = [
        "Bulk",
        "Grain boundaries",
        "Grain edges",
        "Grain corners",
        "Dislocations"
    ]

    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("PRISMA Precipitation Calculator")

        # Fit-to-content behavior (allow user resizing but don't force maximize)
        self.root.resizable(True, True)

        # Control flags for calculations
        self.stop_requested = False
        self.calc_thread = None

        # Selected elements (default)
        self.selected_elements = self.DEFAULT_ELEMENTS.copy()

        # Advanced options with default values
        self.growth_model = tk.StringVar(value="PE_AUTOMATIC")
        self.nucleation_site = tk.StringVar(value="Dislocations")

        # Results options with default values (all enabled)
        self.calc_volume_fraction = tk.BooleanVar(value=True)
        self.calc_mean_radius = tk.BooleanVar(value=True)
        self.calc_number_density = tk.BooleanVar(value=True)
        self.calc_nucleation_rate = tk.BooleanVar(value=True)
        self.calc_matrix_composition = tk.BooleanVar(value=True)
        self.calc_precipitate_composition = tk.BooleanVar(value=True)

        # Build the user interface
        self.setup_ui()

    def setup_ui(self):
        """Build all user interface components."""
        # Create main container with scrollbar for smaller screens
        self.main_canvas = tk.Canvas(self.root, highlightthickness=0)
        self.vscrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.main_container = ttk.Frame(self.main_canvas, padding="10")

        self.main_container.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )

        self.main_canvas.create_window((0, 0), window=self.main_container, anchor="nw")
        self.main_canvas.configure(yscrollcommand=self.vscrollbar.set)

        self.main_canvas.pack(side="left", fill="both", expand=True)
        self.vscrollbar.pack(side="right", fill="y")

        # Application title
        title_label = ttk.Label(
            self.main_container,
            text="PRISMA Precipitation Calculator",
            font=('Arial', 16, 'bold')
        )
        title_label.pack(pady=15)

        # Create each section of the interface
        self.create_file_section(self.main_container)
        self.create_precipitation_section(self.main_container)
        self.create_database_section(self.main_container)
        self.create_buttons_section(self.main_container)
        self.create_log_section(self.main_container)

        # After all widgets are realized, size window to content (and hide scrollbar if not needed)
        self.root.after(0, self.size_to_content)

    def size_to_content(self, horiz_margin=24, vert_margin=80):
        """
        Resize main window to fit content if it fits on screen; otherwise keep scrollbar visible.
        Also auto-hide/show the vertical scrollbar as needed.
        """
        self.root.update_idletasks()

        # Natural requested size of the content frame (inside the canvas)
        req_w = self.main_container.winfo_reqwidth()
        req_h = self.main_container.winfo_reqheight()

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        # Cap target size by screen size (leave margins for WM decorations/taskbar)
        target_w = min(req_w, max(300, screen_w - horiz_margin))
        target_h = min(req_h, max(300, screen_h - vert_margin))

        # Determine whether vertical scrolling is needed
        needs_vscroll = req_h > target_h

        # Estimate scrollbar width
        sbw = self.vscrollbar.winfo_reqwidth() or 18

        # If we need a scrollbar, ensure total window width accounts for it
        if needs_vscroll:
            target_w = min(req_w + sbw, max(300, screen_w - horiz_margin))
            canvas_w = max(100, target_w - sbw)
        else:
            canvas_w = target_w

        # Apply sizes
        self.main_canvas.config(width=canvas_w, height=target_h)
        self.root.geometry(f"{int(target_w)}x{int(target_h)}")

        # Show/hide scrollbar
        if needs_vscroll:
            if not self.vscrollbar.winfo_ismapped():
                self.vscrollbar.pack(side="right", fill="y")
        else:
            if self.vscrollbar.winfo_ismapped():
                self.vscrollbar.pack_forget()

    def create_file_section(self, parent):
        """Create file selection section for input/output files."""
        frame = ttk.LabelFrame(parent, text="Files", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        # Input file (Excel with compositions)
        ttk.Label(frame, text="Input Excel:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )

        self.input_file = tk.StringVar(value="Pareto data for PRISMA 103 alloy.xlsx")
        ttk.Entry(frame, textvariable=self.input_file, width=50).grid(
            row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E)
        )

        ttk.Button(frame, text="Browse", command=self.browse_input).grid(
            row=0, column=2, padx=5, pady=5
        )

        # Output file
        ttk.Label(frame, text="Output Excel:").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )

        self.output_file = tk.StringVar(value="precipitation_results.xlsx")
        ttk.Entry(frame, textvariable=self.output_file, width=50).grid(
            row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E)
        )

        ttk.Button(frame, text="Browse", command=self.browse_output).grid(
            row=1, column=2, padx=5, pady=5
        )

        frame.columnconfigure(1, weight=1)

    def create_precipitation_section(self, parent):
        """Create precipitation parameter inputs."""
        frame = ttk.LabelFrame(parent, text="Precipitation Parameters", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        # Simulation time
        ttk.Label(frame, text="Simulation Time (seconds):").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )

        self.sim_time = tk.DoubleVar(value=60)
        ttk.Entry(frame, textvariable=self.sim_time, width=20).grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.W
        )

        # Matrix phase name
        ttk.Label(frame, text="Matrix Phase:").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )

        self.matrix_phase = tk.StringVar(value="FCC_A1")
        ttk.Entry(frame, textvariable=self.matrix_phase, width=20).grid(
            row=1, column=1, padx=5, pady=5, sticky=tk.W
        )

        # Precipitate phase name
        ttk.Label(frame, text="Precipitate Phase:").grid(
            row=2, column=0, padx=5, pady=5, sticky=tk.W
        )

        self.precipitate_phase = tk.StringVar(value="FCC_A1#2")
        ttk.Entry(frame, textvariable=self.precipitate_phase, width=20).grid(
            row=2, column=1, padx=5, pady=5, sticky=tk.W
        )

        # Note about temperature
        ttk.Label(
            frame,
            text="Note: Temperature is read from 'Temperature' column in Excel file",
            font=('Arial', 9, 'italic'),
            foreground='gray'
        ).grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)

    def create_database_section(self, parent):
        """Create database selection inputs."""
        frame = ttk.LabelFrame(parent, text="Database Settings", padding="10")
        frame.pack(fill=tk.X, pady=5, padx=10)

        # Thermodynamic database
        ttk.Label(frame, text="Thermodynamic DB:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )

        self.tdb = tk.StringVar(value="TCFE13")
        ttk.Entry(frame, textvariable=self.tdb, width=20).grid(
            row=0, column=1, padx=5, pady=5, sticky=tk.W
        )

        # Kinetic database
        ttk.Label(frame, text="Kinetic DB:").grid(
            row=0, column=2, padx=15, pady=5, sticky=tk.W
        )

        self.kdb = tk.StringVar(value="MOBFE8")
        ttk.Entry(frame, textvariable=self.kdb, width=20).grid(
            row=0, column=3, padx=5, pady=5, sticky=tk.W
        )

        # Cache folder
        ttk.Label(frame, text="Cache Folder:").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )

        self.cache = tk.StringVar(value="./cache/")
        ttk.Entry(frame, textvariable=self.cache, width=40).grid(
            row=1, column=1, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E)
        )

        ttk.Button(frame, text="Browse", command=self.browse_cache).grid(
            row=1, column=3, padx=5, pady=5
        )

        frame.columnconfigure(1, weight=1)

    def create_buttons_section(self, parent):
        """Create control buttons and progress display."""
        frame = ttk.Frame(parent, padding="10")
        frame.pack(fill=tk.X, pady=10, padx=10)

        # Buttons frame
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=5)

        # Advanced Options button
        ttk.Button(
            button_frame,
            text="Advanced Options",
            command=self.show_advanced_options,
            width=20
        ).pack(side=tk.LEFT, padx=5)

        # Preview button
        ttk.Button(
            button_frame,
            text="Preview Calculation",
            command=self.show_preview,
            width=20
        ).pack(side=tk.LEFT, padx=5)

        # Spacer
        ttk.Frame(button_frame).pack(side=tk.LEFT, expand=True)

        # Run button
        self.run_btn = ttk.Button(
            button_frame,
            text="Run Calculations",
            command=self.run_calculations,
            width=20
        )
        self.run_btn.pack(side=tk.LEFT, padx=5)

        # Stop button
        self.stop_btn = ttk.Button(
            button_frame,
            text="Stop",
            command=self.stop_calculations,
            state='disabled',
            width=15
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(frame, maximum=100)
        self.progress.pack(fill=tk.X, pady=10)

    def create_log_section(self, parent):
        """Create status log display area."""
        frame = ttk.LabelFrame(parent, text="Log", padding="5")
        frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=10)

        # Create scrollbar
        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create text widget
        self.log = tk.Text(
            frame,
            height=12,
            yscrollcommand=scrollbar.set,
            wrap=tk.WORD,
            font=('Courier', 13)
        )
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=self.log.yview)

    def write_log(self, message):
        """Add a message to the status log."""
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.root.update_idletasks()

    def browse_input(self):
        """Open file dialog to select input Excel file."""
        filename = filedialog.askopenfilename(
            title="Select Input Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
            self.size_to_content()

    def browse_output(self):
        """Open file dialog to select output Excel file."""
        filename = filedialog.asksaveasfilename(
            title="Select Output Excel File",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.output_file.set(filename)
            self.size_to_content()

    def browse_cache(self):
        """Open file dialog to select cache folder."""
        folder = filedialog.askdirectory(title="Select Cache Folder")
        if folder:
            self.cache.set(folder)
            self.size_to_content()

    def show_element_selector(self):
        """Open element selector window."""
        elem_window = tk.Toplevel(self.root)
        elem_window.title("Select Alloying Elements")
        elem_window.geometry("1000x470")
        elem_window.resizable(False, False)

        # Main frame
        main_frame = ttk.Frame(elem_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(
            main_frame,
            text="Select Alloying Elements",
            font=('Arial', 14, 'bold')
        ).pack(pady=(0, 10))

        # Instructions
        ttk.Label(
            main_frame,
            text="Click on elements to select/deselect. Fe (Iron) is always included as base element.",
            font=('Arial', 10)
        ).pack(pady=(0, 15))

        # Periodic table frame
        pt_frame = ttk.Frame(main_frame)
        pt_frame.pack(pady=10)

        # Store button references and temporary selection
        self.element_buttons = {}
        self.temp_selected = self.selected_elements.copy()

        # Create periodic table buttons
        for element, (row, col) in self.AVAILABLE_ELEMENTS.items():
            btn = tk.Button(
                pt_frame,
                text=element,
                width=4,
                height=2,
                font=('Arial', 10, 'bold'),
                relief=tk.RAISED,
                borderwidth=2,
                command=lambda e=element: self.toggle_element(e)
            )

            # Color selected elements
            if element in self.temp_selected:
                btn.config(bg='#4CAF50', fg='white', relief=tk.SUNKEN)
            else:
                btn.config(bg='white', fg='black')

            # Disable Fe (always included)
            if element == 'Fe':
                btn.config(state='disabled', bg='#E0E0E0')

            btn.grid(row=row, column=col, padx=2, pady=2, sticky='nsew')
            self.element_buttons[element] = btn

        # Add lanthanide label
        ttk.Label(pt_frame, text="* Lanthanides", font=('Arial', 9)).grid(
            row=7, column=0, columnspan=2, sticky=tk.W
        )

        # Selected elements display
        selected_frame = ttk.LabelFrame(main_frame, text="Selected Elements", padding="10")
        selected_frame.pack(fill=tk.X, pady=15)

        self.selected_label = ttk.Label(
            selected_frame,
            text=", ".join(sorted(self.temp_selected)),
            font=('Arial', 11),
            wraplength=800
        )
        self.selected_label.pack()

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)

        ttk.Button(
            button_frame,
            text="Reset to Default",
            command=lambda: self.reset_elements_to_default(elem_window)
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Cancel",
            command=elem_window.destroy
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            button_frame,
            text="OK",
            command=lambda: self.save_element_selection(elem_window)
        ).pack(side=tk.RIGHT, padx=5)

    def toggle_element(self, element):
        """Toggle element selection."""
        if element == 'Fe':
            return  # Fe is always selected

        if element in self.temp_selected:
            self.temp_selected.remove(element)
            self.element_buttons[element].config(bg='white', fg='black', relief=tk.RAISED)
        else:
            self.temp_selected.append(element)
            self.element_buttons[element].config(bg='#4CAF50', fg='white', relief=tk.SUNKEN)

        # Update selected elements display
        self.selected_label.config(text=", ".join(sorted(self.temp_selected)))

    def reset_elements_to_default(self, window):
        """Reset elements to default selection."""
        self.temp_selected = self.DEFAULT_ELEMENTS.copy()

        # Update all button states
        for element, btn in self.element_buttons.items():
            if element == 'Fe':
                continue
            if element in self.temp_selected:
                btn.config(bg='#4CAF50', fg='white', relief=tk.SUNKEN)
            else:
                btn.config(bg='white', fg='black', relief=tk.RAISED)

        # Update display
        self.selected_label.config(text=", ".join(sorted(self.temp_selected)))

    def save_element_selection(self, window):
        """Save element selection and close window."""
        if len(self.temp_selected) < 2:
            messagebox.showwarning(
                "Insufficient Elements",
                "Please select at least one alloying element in addition to Fe."
            )
            return

        self.selected_elements = self.temp_selected.copy()
        window.destroy()
        messagebox.showinfo(
            "Elements Updated",
            f"Selected elements: {', '.join(sorted(self.selected_elements))}\n\n"
            "Note: Make sure your input Excel file contains columns for these elements."
        )

    def show_preview(self):
        """Show preview of calculation parameters and number of calculations."""
        try:
            # Validate inputs first
            if not self.validate_inputs():
                return

            # Read input file to get number of compositions
            input_file = self.input_file.get()
            compositions_df = pd.read_excel(input_file)
            num_compositions = len(compositions_df)

            # Get selected results
            selected_results = []
            if self.calc_volume_fraction.get():
                selected_results.append("Volume fraction")
            if self.calc_mean_radius.get():
                selected_results.append("Mean radius")
            if self.calc_number_density.get():
                selected_results.append("Number density")
            if self.calc_nucleation_rate.get():
                selected_results.append("Nucleation rate")
            if self.calc_matrix_composition.get():
                selected_results.append("Matrix composition")
            if self.calc_precipitate_composition.get():
                selected_results.append("Precipitate composition")

            # Build preview message
            preview_msg = f"""
CALCULATION PREVIEW
{'=' * 50}

INPUT PARAMETERS:
  • Input file: {os.path.basename(input_file)}
  • Number of compositions: {num_compositions}
  • Simulation time: {self.sim_time.get():.0f} seconds
  • Temperature: Read from Excel file

ALLOYING ELEMENTS:
  • Selected: {', '.join(sorted(self.selected_elements))}

PHASE CONFIGURATION:
  • Matrix phase: {self.matrix_phase.get()}
  • Precipitate phase: {self.precipitate_phase.get()}

DATABASE SETTINGS:
  • Thermodynamic DB: {self.tdb.get()}
  • Kinetic DB: {self.kdb.get()}

ADVANCED OPTIONS:
  • Growth model: {self.growth_model.get()}
  • Nucleation site: {self.nucleation_site.get()}

RESULTS TO CALCULATE:
"""
            for result in selected_results:
                preview_msg += f"  ✓ {result}\n"

            preview_msg += f"""
{'=' * 50}
TOTAL CALCULATIONS: {num_compositions}

Output will be saved to:
{self.output_file.get()}
"""

            # Create preview window
            preview_window = tk.Toplevel(self.root)
            preview_window.title("Calculation Preview")
            preview_window.geometry("620x650")

            # Text widget for preview
            text_widget = tk.Text(
                preview_window,
                wrap=tk.WORD,
                font=('Courier', 12),
                padx=20,
                pady=20
            )
            text_widget.pack(fill=tk.BOTH, expand=True)
            text_widget.insert('1.0', preview_msg)
            text_widget.config(state='disabled')

            # Close button
            ttk.Button(
                preview_window,
                text="Close",
                command=preview_window.destroy
            ).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Preview Error", f"Cannot generate preview:\n{str(e)}")

    def show_advanced_options(self):
        """Open a new window with advanced precipitation options."""
        # Create new window
        advanced_window = tk.Toplevel(self.root)
        advanced_window.title("Advanced Options")
        advanced_window.geometry("500x750")
        advanced_window.resizable(False, False)

        # Main frame with padding
        content_frame = ttk.Frame(advanced_window, padding="20")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(
            content_frame,
            text="Advanced Precipitation Options",
            font=('Arial', 14, 'bold')
        ).pack(pady=(0, 20))

        # Element Selection Section
        element_frame = ttk.LabelFrame(content_frame, text="Alloying Elements", padding="15")
        element_frame.pack(fill=tk.X, pady=10)

        ttk.Label(
            element_frame,
            text=f"Current elements: {', '.join(sorted(self.selected_elements))}",
            font=('Arial', 10),
            wraplength=450
        ).pack(pady=(0, 10))

        ttk.Button(
            element_frame,
            text="Select Elements",
            command=self.show_element_selector
        ).pack()

        # Growth Model Section
        growth_frame = ttk.LabelFrame(content_frame, text="Growth Rate Model", padding="15")
        growth_frame.pack(fill=tk.X, pady=10)

        ttk.Label(growth_frame, text="Select Model:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )

        ttk.Combobox(
            growth_frame,
            textvariable=self.growth_model,
            values=self.GROWTH_MODELS,
            state='readonly',
            width=25
        ).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(
            growth_frame,
            text="PE_AUTOMATIC enables driving force approximation",
            font=('Arial', 9, 'italic'),
            foreground='gray'
        ).grid(row=1, column=0, columnspan=2, padx=5, pady=(5, 0), sticky=tk.W)

        # Nucleation Site Section
        nucleation_frame = ttk.LabelFrame(content_frame, text="Nucleation Sites", padding="15")
        nucleation_frame.pack(fill=tk.X, pady=10)

        ttk.Label(nucleation_frame, text="Select Site:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )

        ttk.Combobox(
            nucleation_frame,
            textvariable=self.nucleation_site,
            values=self.NUCLEATION_SITES,
            state='readonly',
            width=25
        ).grid(row=0, column=1, padx=5, pady=5)

        # Results Section
        results_frame = ttk.LabelFrame(content_frame, text="Results to Calculate", padding="15")
        results_frame.pack(fill=tk.X, pady=10)

        ttk.Label(
            results_frame,
            text="Select which results to calculate and export:",
            font=('Arial', 10)
        ).pack(anchor=tk.W, pady=(0, 10))

        # Checkboxes for results
        result_options = [
            ("Volume fraction", self.calc_volume_fraction),
            ("Mean radius", self.calc_mean_radius),
            ("Number density", self.calc_number_density),
            ("Nucleation rate", self.calc_nucleation_rate),
            ("Matrix composition", self.calc_matrix_composition),
            ("Precipitate composition", self.calc_precipitate_composition)
        ]

        for text, var in result_options:
            ttk.Checkbutton(results_frame, text=text, variable=var).pack(
                anchor=tk.W, pady=3
            )

        ttk.Label(
            results_frame,
            text="Note: At least one result must be selected",
            font=('Arial', 9, 'italic'),
            foreground='gray'
        ).pack(anchor=tk.W, pady=(10, 0))

        # Buttons
        button_frame = ttk.Frame(content_frame)
        button_frame.pack(fill=tk.X, pady=20)

        ttk.Button(
            button_frame,
            text="Cancel",
            command=lambda: self.cancel_advanced_options(advanced_window)
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(
            button_frame,
            text="OK",
            command=lambda: self.save_advanced_options(advanced_window)
        ).pack(side=tk.RIGHT, padx=5)

        # Store original values for cancel
        self._store_original_values()

    def _store_original_values(self):
        """Store original values of advanced options for cancel functionality."""
        self.original_growth_model = self.growth_model.get()
        self.original_nucleation_site = self.nucleation_site.get()
        self.original_calc_volume_fraction = self.calc_volume_fraction.get()
        self.original_calc_mean_radius = self.calc_mean_radius.get()
        self.original_calc_number_density = self.calc_number_density.get()
        self.original_calc_nucleation_rate = self.calc_nucleation_rate.get()
        self.original_calc_matrix_composition = self.calc_matrix_composition.get()
        self.original_calc_precipitate_composition = self.calc_precipitate_composition.get()

    def save_advanced_options(self, window):
        """Validate and save advanced options."""
        # Check that at least one result is selected
        if not any([
            self.calc_volume_fraction.get(),
            self.calc_mean_radius.get(),
            self.calc_number_density.get(),
            self.calc_nucleation_rate.get(),
            self.calc_matrix_composition.get(),
            self.calc_precipitate_composition.get()
        ]):
            messagebox.showwarning(
                "No Results Selected",
                "Please select at least one result to calculate."
            )
            return

        window.destroy()
        # Re-evaluate fit in case label sizes changed
        self.size_to_content()

    def cancel_advanced_options(self, window):
        """Restore original values and close advanced options window."""
        self.growth_model.set(self.original_growth_model)
        self.nucleation_site.set(self.original_nucleation_site)
        self.calc_volume_fraction.set(self.original_calc_volume_fraction)
        self.calc_mean_radius.set(self.original_calc_mean_radius)
        self.calc_number_density.set(self.original_calc_number_density)
        self.calc_nucleation_rate.set(self.original_calc_nucleation_rate)
        self.calc_matrix_composition.set(self.original_calc_matrix_composition)
        self.calc_precipitate_composition.set(self.original_calc_precipitate_composition)
        window.destroy()

    def get_required_columns(self):
        """Get required columns based on selected elements."""
        # Temperature is always required, plus all non-Fe elements
        columns = [elem for elem in self.selected_elements if elem != "Fe"]
        columns.append("Temperature")
        return columns

    def validate_inputs(self):
        """Validate all user inputs before running calculations."""
        try:
            # Check input file exists
            if not os.path.exists(self.input_file.get()):
                raise ValueError("Input Excel file does not exist")

            # Check simulation time is positive
            if self.sim_time.get() <= 0:
                raise ValueError("Simulation time must be positive")

            # Check phase names are not empty
            if not self.matrix_phase.get().strip():
                raise ValueError("Matrix phase name cannot be empty")

            if not self.precipitate_phase.get().strip():
                raise ValueError("Precipitate phase name cannot be empty")

            # Check database names are not empty
            if not self.tdb.get().strip():
                raise ValueError("Thermodynamic database name cannot be empty")

            if not self.kdb.get().strip():
                raise ValueError("Kinetic database name cannot be empty")

            return True

        except ValueError as error:
            messagebox.showerror("Invalid Input", str(error))
            return False

    def run_calculations(self):
        """Start precipitation calculations in a background thread."""
        if not self.validate_inputs():
            return

        # Check TC-Python availability
        if not TC_PYTHON_AVAILABLE:
            messagebox.showerror(
                "TC-Python Not Found",
                "TC-Python is not installed or not available.\n"
                "Please install TC-Python to run calculations."
            )
            return

        confirmation_message = (
            "Start precipitation calculations?\n\n"
            "This may take a long time depending on the number\n"
            "of compositions and simulation time.\n\n"
            "Continue?"
        )
        user_confirmed = messagebox.askyesno("Confirm", confirmation_message)

        if not user_confirmed:
            return

        # Update button states
        self.run_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.stop_requested = False
        self.progress['value'] = 0

        # Log start message
        separator = "=" * 70
        self.write_log(separator)
        self.write_log("Starting precipitation calculations...")
        self.write_log(f"Selected elements: {', '.join(sorted(self.selected_elements))}")
        self.write_log(separator)

        # Start calculation thread
        self.calc_thread = threading.Thread(target=self.do_calculations, daemon=True)
        self.calc_thread.start()

    def stop_calculations(self):
        """Request to stop ongoing calculations."""
        self.stop_requested = True
        self.write_log("Stop requested, finishing current calculation...")
        self.stop_btn.config(state='disabled')

    def do_calculations(self):
        """Main calculation worker method (runs in background thread)."""
        try:
            # Get configuration
            input_file = self.input_file.get()
            output_file = self.output_file.get()
            sim_time = self.sim_time.get()
            matrix_phase = self.matrix_phase.get()
            precipitate_phase = self.precipitate_phase.get()
            tdb = self.tdb.get()
            kdb = self.kdb.get()
            cache_folder = self.cache.get()

            # Create cache folder if needed
            os.makedirs(cache_folder, exist_ok=True)

            # Read compositions from Excel
            self.write_log(f"Reading compositions from: {input_file}")
            try:
                compositions_df = pd.read_excel(input_file)
            except Exception as error:
                raise RuntimeError(f"Unable to read Excel file: {error}")

            # Check required columns
            required_columns = self.get_required_columns()
            missing_columns = [
                col for col in required_columns
                if col not in compositions_df.columns
            ]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

            total_compositions = len(compositions_df)
            self.write_log(f"Found {total_compositions} composition(s) to calculate")

            # Store all results
            all_results = []

            # Start TC-Python session
            with TCPython() as session:
                self.write_log("Initializing TC-Python session...")

                # Setup system with databases and elements
                system = (
                    session
                    .set_cache_folder(cache_folder)
                    .select_thermodynamic_and_kinetic_databases_with_elements(
                        tdb, kdb, self.selected_elements
                    )
                    .get_system()
                )

                self.write_log("System initialized successfully")

                # Loop over each composition
                for comp_index, comp_row in compositions_df.iterrows():
                    if self.stop_requested:
                        break

                    # Extract compositions and process
                    result = self._process_composition(
                        system, comp_row, comp_index, total_compositions,
                        matrix_phase, precipitate_phase, sim_time
                    )

                    if result:
                        all_results.extend(result)

                    # Update progress
                    progress_percent = ((comp_index + 1) / total_compositions) * 100
                    self.root.after(0, lambda p=progress_percent: self.progress.config(value=p))

            # Save results to Excel
            if all_results and not self.stop_requested:
                self._save_results(all_results, output_file)

            # Log completion
            if self.stop_requested:
                self.write_log("\nCalculations stopped by user")
            else:
                self.write_log("\nAll calculations completed successfully!")

        except Exception as error:
            self.write_log(f"\nERROR: {str(error)}")
            import traceback
            self.write_log(traceback.format_exc())

        finally:
            # Reset UI controls
            self.root.after(0, self.reset_buttons)

    def _process_composition(self, system, comp_row, comp_index, total_compositions,
                            matrix_phase, precipitate_phase, sim_time):
        """Process a single composition and return results."""
        try:
            # Extract element compositions (mass percent) for selected elements
            composition = {}
            for elem in self.selected_elements:
                if elem != "Fe":  # Fe is balance, don't set explicitly
                    composition[elem] = float(comp_row.get(elem, 0.0) or 0.0)

            # Extract temperature and convert to Kelvin
            try:
                temp_celsius = float(comp_row.get("Temperature"))
                temperature_kelvin = temp_celsius + 273.15
            except Exception:
                temperature_kelvin = 273.15
                temp_celsius = 0.0

            # Log current calculation
            comp_summary = f"Composition {comp_index + 1}/{total_compositions}: "
            comp_parts = [f"{elem}={composition[elem]:.3f}" for elem in sorted(composition.keys())[:3]]
            comp_summary += ", ".join(comp_parts)
            comp_summary += f", Temp={temp_celsius:.0f}°C"
            self.write_log(comp_summary)

            # Build and run calculation
            precip_calc = self._build_calculation(
                system, composition, temperature_kelvin, sim_time,
                matrix_phase, precipitate_phase
            )

            # Extract results
            results = self._extract_results(
                precip_calc, precipitate_phase, composition, temp_celsius
            )

            self.write_log(f"  Completed: {len(results)} time points extracted")
            return results

        except Exception as error:
            self.write_log(f"  Error: {str(error)}")
            return None

    def _build_calculation(self, system, composition, temperature_kelvin, sim_time,
                          matrix_phase, precipitate_phase):
        """Build the precipitation calculation with all parameters."""
        # Create precipitate phase with growth model
        precipitate_phase_obj = PrecipitatePhase(precipitate_phase)

        # Apply growth model
        growth_model_name = self.growth_model.get()
        growth_model_map = {
            "Simplified": GrowthRateModel.SIMPLIFIED,
            "General": GrowthRateModel.GENERAL,
            "Advanced": GrowthRateModel.ADVANCED,
            "Para_eq": GrowthRateModel.PARA_EQ,
            "NPLE": GrowthRateModel.NPLE,
        }

        if growth_model_name == "PE_AUTOMATIC":
            precipitate_phase_obj = (
                precipitate_phase_obj
                .with_growth_rate_model(GrowthRateModel.PE_AUTOMATIC)
                .enable_driving_force_approximation()
            )
        elif growth_model_name in growth_model_map:
            precipitate_phase_obj = precipitate_phase_obj.with_growth_rate_model(
                growth_model_map[growth_model_name]
            )

        # Apply nucleation site
        nucleation_site_name = self.nucleation_site.get()
        nucleation_methods = {
            "Bulk": precipitate_phase_obj.set_nucleation_in_bulk,
            "Grain boundaries": precipitate_phase_obj.set_nucleation_at_grain_boundaries,
            "Grain edges": precipitate_phase_obj.set_nucleation_at_grain_edges,
            "Grain corners": precipitate_phase_obj.set_nucleation_at_grain_corners,
            "Dislocations": precipitate_phase_obj.set_nucleation_at_dislocations
        }

        if nucleation_site_name in nucleation_methods:
            precipitate_phase_obj = nucleation_methods[nucleation_site_name]()

        # Build precipitation calculation
        calc = (
            system
            .with_isothermal_precipitation_calculation()
            .set_composition_unit(CompositionUnit.MASS_PERCENT)
        )

        # Set composition for each selected element (except Fe)
        for elem, value in composition.items():
            calc = calc.set_composition(elem, value)

        # Complete calculation setup
        precip_calc = (
            calc
            .set_temperature(temperature_kelvin)
            .set_simulation_time(sim_time)
            .with_matrix_phase(
                MatrixPhase(matrix_phase).add_precipitate_phase(precipitate_phase_obj)
            )
            .calculate()
        )

        return precip_calc

    def _extract_results(self, precip_calc, precipitate_phase, composition, temp_celsius):
        """Extract all requested results from the calculation."""
        # Get time points (always needed)
        times, vol_fracs = precip_calc.get_volume_fraction_of(precipitate_phase)

        # Initialize result containers
        mean_radii = None
        number_densities = None
        nucleation_rates = None
        precip_comp = {}
        matrix_comp = {}

        # Extract optional results
        if self.calc_mean_radius.get():
            _, mean_radii = precip_calc.get_mean_radius_of(precipitate_phase)

        if self.calc_number_density.get():
            _, number_densities = precip_calc.get_number_density_of(precipitate_phase)

        if self.calc_nucleation_rate.get():
            _, nucleation_rates = precip_calc.get_nucleation_rate_of(precipitate_phase)

        if self.calc_precipitate_composition.get():
            for elem in self.selected_elements:
                if elem != "Fe":
                    try:
                        _, comp_values = precip_calc.get_precipitate_composition_in_weight_fraction_of(
                            precipitate_phase, elem
                        )
                        precip_comp[elem] = comp_values
                    except Exception:
                        precip_comp[elem] = [None] * len(times)

        if self.calc_matrix_composition.get():
            for elem in self.selected_elements:
                if elem != "Fe":
                    try:
                        _, comp_values = precip_calc.get_matrix_composition_in_weight_fraction_of(elem)
                        matrix_comp[elem] = comp_values
                    except Exception:
                        matrix_comp[elem] = [None] * len(times)

        # Build result rows
        results = []
        for time_index, time_value in enumerate(times):
            result_row = {}

            # Add input composition for each selected element
            for elem, value in composition.items():
                result_row[f"{elem}_wt_percent"] = value

            result_row["Temperature_C"] = temp_celsius
            result_row["Time_s"] = time_value

            # Add volume fraction if calculated
            if self.calc_volume_fraction.get() and vol_fracs is not None:
                result_row["Precipitate_volume_fraction"] = vol_fracs[time_index]

            # Add mean radius if calculated
            if self.calc_mean_radius.get() and mean_radii is not None:
                result_row["Mean_radius_m"] = mean_radii[time_index]

            # Add number density if calculated
            if self.calc_number_density.get() and number_densities is not None:
                result_row["Number_density_m3"] = number_densities[time_index]

            # Add nucleation rate if calculated
            if self.calc_nucleation_rate.get() and nucleation_rates is not None:
                result_row["Nucleation_rate_m3s"] = nucleation_rates[time_index]

            # Add precipitate composition if calculated
            if self.calc_precipitate_composition.get() and precip_comp:
                for elem in self.selected_elements:
                    if elem != "Fe" and elem in precip_comp:
                        result_row[f"Precipitate_wtfrac_{elem}"] = precip_comp[elem][time_index]

            # Add matrix composition if calculated
            if self.calc_matrix_composition.get() and matrix_comp:
                for elem in self.selected_elements:
                    if elem != "Fe" and elem in matrix_comp:
                        result_row[f"Matrix_wtfrac_{elem}"] = matrix_comp[elem][time_index]

            results.append(result_row)

        return results

    def _save_results(self, all_results, output_file):
        """Save all results to Excel file."""
        self.write_log("\nSaving results to Excel...")
        results_df = pd.DataFrame(all_results)

        # Ensure output has .xlsx extension
        if not output_file.lower().endswith(".xlsx"):
            output_file = output_file.rsplit(".", 1)[0] + ".xlsx"

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name="Results", index=False)

        self.write_log(f"Results saved: {output_file}")
        self.write_log(f"Total rows: {len(results_df)}")

    def reset_buttons(self):
        """Reset button states after calculations complete or stop."""
        self.run_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress['value'] = 0


def main():
    """Launch the PRISMA precipitation calculator application."""
    root = tk.Tk()
    app = PrismaCalculatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()