"""
About tab displaying application information
"""

import tkinter as tk
from tkinter import ttk

class AboutTab:
    """About tab with application description and information."""

    def __init__(self, app, notebook):
        self.app = app
        self.setup_ui(notebook)

    def setup_ui(self, notebook):
        """Create the About tab UI."""
        self.frame = ttk.Frame(notebook, padding="20")
        notebook.add(self.frame, text="About")

        # Application title
        ttk.Label(
            self.frame,
            text="Alloy Design Toolkit",
            font=('Arial', 18, 'bold')
        ).pack(pady=(10, 5))

        ttk.Label(
            self.frame,
            text="Automatic Workflow for Advanced Steel Alloy Design",
            font=('Arial', 12, 'italic')
        ).pack(pady=(0, 20))

        # Version and developer info
        info_frame = ttk.Frame(self.frame)
        info_frame.pack(fill=tk.X, pady=10)

        ttk.Label(
            info_frame,
            text="Version: 1.0.0",
            font=('Arial', 10)
        ).pack()

        ttk.Label(
            info_frame,
            text="Developed by: Mahmoud Elaraby - mahmoud.elaraby@oulu.fi",
            font=('Arial', 10)
        ).pack()

        # Create buttons frame
        buttons_frame = ttk.Frame(self.frame)
        buttons_frame.pack(expand=True, pady=30)

        # Define button data
        button_data = [
            ("Overview", self.show_overview),
            ("Computational Modules", self.show_computational_modules),
            ("Workflow Description", self.show_workflow_description)
        ]

        # Create buttons
        for text, command in button_data:
            btn = ttk.Button(
                buttons_frame,
                text=text,
                command=command,
                width=30
            )
            btn.pack(pady=10)

    def show_popup(self, title, content):
        """Create a popup window with the given title and content."""
        popup = tk.Toplevel(self.frame)
        popup.title(title)
        popup.geometry("700x500")

        # Create scrollable text widget
        text_frame = ttk.Frame(popup)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text widget
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=('Times New Roman', 14),
            yscrollcommand=scrollbar.set,
            padx=10,
            pady=10
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        # Insert content
        text_widget.insert(1.0, content.strip())
        text_widget.config(state=tk.DISABLED)

        # Close button
        close_btn = ttk.Button(
            popup,
            text="Close",
            command=popup.destroy
        )
        close_btn.pack(pady=10)

        # Center the popup window
        popup.update_idletasks()
        width = popup.winfo_width()
        height = popup.winfo_height()
        x = (popup.winfo_screenwidth() // 2) - (width // 2)
        y = (popup.winfo_screenheight() // 2) - (height // 2)
        popup.geometry(f'{width}x{height}+{x}+{y}')

    def show_overview(self):
        """Show overview popup."""
        content = """This application provides an integrated computational workflow for designing advanced high-strength steel alloys, particularly medium-manganese (Med-Mn) steels. It combines thermodynamic calculations, kinetic modeling, and multi-objective optimization to predict optimal alloy compositions and processing conditions.

The toolkit is designed to streamline the alloy design process by automating the calculation of critical material properties and identifying optimal compositions through multi-objective optimization algorithms. It integrates multiple computational modules that work together to provide comprehensive insights into alloy behavior during processing and in service.

The application serves as a bridge between thermodynamic databases (Thermo-Calc) and materials design, enabling researchers and engineers to rapidly screen composition spaces and predict the performance of novel steel alloys."""

        self.show_popup("Overview", content)


    def show_computational_modules(self):
        """Show computational modules popup."""
        content = """1. Phase Calculations 
   - Uses TC-Python API to calculate phase equilibria
   - Parallel processing for multiple compositions
   - Exports phase fractions and compositions vs. temperature


2. Retained Austenite prediction Model 
   - Based on Koistinen-Marburger equation
   - Predicts Ms temperature using imported data from first module
   - Predicts retained austenite fraction based on quench temperature

3. Stacking Fault Energy Model 
   - Thermodynamic-based SFE calculation
   - Considers temperature, grain size, and composition
   - Predicts TRIP vs. TWIP behavior
   - Includes magnetic and chemical contributions to SFE
   - Temperature-dependent calculations

4. Multi-Objective Algorithm  
   - Non-dominated sorting genetic algorithm II (NSGA-II)
   - Optimizes for strength, ductility, and processing window
   - Applies design constraints (Ms, RA, SFE ranges, prcessing window width ΔT)
   - Ranks solutions using Pareto dominance based on crowding distance

5. Precipitation Kinetics 
   - TC-PRISMA simulations for carbide precipitation
   - Calculates volume fraction, particle size, and number density
   - Time-series evolution tracking
   - Handles all types of precipitates and different matrices
   - Accounts for nucleation, growth, and coarsening

6. PRISMA Geometric Mean ranking 
   - Organizes PRISMA output files by GM rank
   - Prepares batch simulations for top-ranked alloys
   - Manages file structure for efficient post-processing
   - Extracts key metrics from PRISMA output files"""

        self.show_popup("Computational Modules", content)

    def show_workflow_description(self):
        """Show workflow description popup."""
        content = """The automated workflow executes the following sequence:

1. Define Composition Space
   - User specifies elements to include for the calculations
   - Define concentration ranges and step sizes for each element
   - User defines the temperature range and step size
   - User selects phases and their fractions and/or compositions for calculation
   - User defines the steel and Fe-alloys database (e.g. TCFE13)
   - User selects the suitable workers number (parallel calculation executed at the same time) based on CPU capacity
   - System generates all composition combinations within specified ranges

2. Predict Retained Austenite
   - This module imports the data from step 1 (austenite fraction and composition vs temperature) 
   - Calculate martensite start temperature (Ms) for each composition
   - Estimate retained austenite fraction after quenching

3. Calculate Stacking Fault Energy
   - This module imports the data from step 1 (austenite composition vs temperature) 
   - calculate SFE for austenite phase at specified temperature


4. Multi-Objective Optimization (alloying elements, annealing temperature)
   - This module imports the data from steps 1-3
   - It is used to optimize the alloying elements content and intrcritical annealing temperature
   - Constraints: Ms, RA fraction, SFE ranges, Cementite fraction, Martensite fraction
   - Objectives: maximize strength, ductility, and processing window width represented by Ms, RA, SFE, ΔT
   - Apply NSGA-II algorithm to identify Pareto-optimal solutions according to the designated objectives
   - Rank solutions by dominance and crowding distance

5. Precipitation Kinetics Simulation
   - Run TC-PRISMA for top-ranked alloys
   - Simulate precipitation during intercritical annealing
   - Track precipitate evolution (e.g. volume fraction, size, number density) vs time

6. Determine Optimal Annealing Time
   - This module imports the data from steps 5
   - Analyze precipitation kinetics results
   - Identify optimal annealing time based on objectives
   - Balance precipitate strengthening with austenite stability

8. Export Results
   - All results saved to Excel file
   - Comprehensive data including compositions, properties, and rankings
   - Separate sheets for each calculation module
   - Ready for further analysis and visualization

Each step can also be executed independently using the "Standalone Scripts" tab for custom analyses or troubleshooting."""

        self.show_popup("Workflow Description", content)
