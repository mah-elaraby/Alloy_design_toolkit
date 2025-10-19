# Alloy Design Toolkit

This toolkit provides an automatic computational workflow for designing advanced high-strength steel alloys, specifically focusing on medium-manganese steels. It integrates thermodynamic calculations (using TC-Python), kinetic modelling (TC-PRISMA), and multi-objective optimization (NSGA-II) to predict optimal alloy compositions and processing parameters (intrcritical annealing temperature and time).

## Features

- **Phase Calculations**: Advanced thermodynamic calculations for alloy phase prediction
- **Martensite/Austenite Predictions**: Automated prediction of phase transformations
- **Stacking Fault Energy Analysis**: Calculation and optimization of stacking fault energies
- **Multi-objective Optimization**: Sophisticated optimization algorithms for alloy composition
- **Precipitation Kinetics**: Modeling of precipitation behavior in medium-Mn steels
- **Annealing Time Optimization**: Automated optimization of heat treatment parameters
- **Graphical User Interface**: User-friendly Tkinter-based GUI for easy interaction
- **Thermo-Calc Integration**: Seamless integration with Thermo-Calc for thermodynamic calculations

## Computational Workflow

The Alloy Design Toolkit follows a systematic 8-step computational workflow:

### 1. Define Composition Space
- **User Input Configuration**:
  - Specify elements to include in calculations
  - Define concentration ranges and step sizes for each element
  - Set temperature range and step size for calculations
  - Select phases and their fractions/compositions for analysis
  - Choose steel and Fe-alloys database (e.g., TCFE13)
  - Configure parallel processing workers based on CPU capacity
- **System Processing**:
  - Generate all composition combinations within specified ranges

### 2. Predict Retained Austenite
- **Data Import**: Imports austenite fraction and composition vs temperature from Step 1
- **Calculations**:
  - Calculate martensite start temperature (Ms) for each composition
  - Estimate retained austenite fraction after quenching

### 3. Calculate Stacking Fault Energy
- **Data Import**: Imports austenite composition vs temperature from Step 1
- **SFE Calculation**:
  - Calculate stacking fault energy (SFE) for austenite phase at specified temperatures

### 4. Multi-Objective Optimization
- **Data Integration**: Imports results from Steps 1-3
- **Optimization Parameters**:
  - **Variables**: Alloying element content and intercritical annealing temperature
  - **Constraints**: Ms temperature, RA fraction, SFE ranges, cementite fraction, martensite fraction, processing window width (ΔT)
  - **Objectives**: Maximize strength, ductility, and processing window width (represented by Ms, RA, SFE, ΔT)
- **Algorithm**: NSGA-II (Non-dominated Sorting Genetic Algorithm II)
  - Identify Pareto-optimal solutions
  - Rank solutions by dominance and crowding distance

### 5. Precipitation Kinetics Simulation
- **TC-PRISMA Integration**: Run simulations for top-ranked alloys from Step 4
- **Kinetic Analysis**:
  - Simulate precipitation during intercritical annealing
  - Track precipitate evolution parameters:
    - Volume fraction, mean radius, nucleation rate, precipitate and matrix composition vs time

### 6. Determine Optimal Annealing Time
- **Data Analysis**: Process precipitation kinetics results from Step 5
- **Optimization Criteria**:
  - Balance precipitate strengthening with austenite stability
  - Identify optimal annealing time based on defined objectives

### 7. Export Results
- **Comprehensive Data Export**:
  - All results saved to a structured Excel file
  - Multiple worksheets for each calculation module:
    - Composition matrix and phase fractions
    - Retained austenite predictions
    - Stacking fault energy calculations
    - Optimization results and Pareto fronts
    - Precipitation kinetics data
    - Optimal processing parameters
  - Ready for further analysis and visualization

## Project Structure

```
Alloy_design_toolkit/
├── main.py                 # Main application entry point
├── run_app.py             # Alternative application launcher
├── config/                # Configuration files and settings
├── core/                  # Core computational modules
│   ├── workflow_runner.py            # Workflow orchestration
│   ├── phase_calculator.py           # Step 1: Composition space generation
│   ├── martensite_calculator.py      # Step 2: Retained austenite prediction
│   ├── sfe_calculator.py             # Step 3: Stacking fault energy
│   ├── moo_optimizer.py              # Step 4: Multi-objective optimization
│   ├── precipitation_calculator.py   # Step 5: Precipitation kinetics
│   └── annealing_optimizer.py        # Step 6: Annealing time optimization
├── gui/                   # Graphical user interface components
│   ├── main_window.py     # Main application window
│   ├── tabs/              # Individual workflow step tabs
│   └── dialogs/           # Configuration and input dialogs
├── standalone_scripts/    # Independent utility scripts
└── utils/                # Utility functions and helpers
```

## Requirements

- Python 3.7 or higher
- Tkinter (usually included with Python)
- **Thermo-Calc** (for thermodynamic calculations)
- **TC-Python** (Thermo-Calc Python API)
- **TC-PRISMA** (for precipitation kinetics)
- Required Python packages:
  ```bash
  pip install numpy scipy pandas openpyxl matplotlib
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mah-elaraby/Alloy_design_toolkit.git
   cd Alloy_design_toolkit
   ```

2. Ensure Thermo-Calc, TC-Python, and TC-PRISMA are properly installed and configured

3. Configure database paths in the `config/` directory

## Usage

### GUI Application

Launch the graphical interface:
```bash
python main.py
```

Or use the alternative launcher:
```bash
python run_app.py
```

### Workflow Navigation

1. **Setup**: Configure composition space and calculation parameters
2. **Execute**: Run sequential workflow steps or individual modules
3. **Monitor**: Track calculation progress and intermediate results
4. **Analyze**: Review optimization results and Pareto fronts
5. **Export**: Generate comprehensive Excel reports

### Standalone Scripts

Individual computational modules can be run independently from the `standalone_scripts/` directory for specific calculations or testing.

## Configuration

Configuration files in the `config/` directory allow customization of:
- Default alloy compositions and element ranges
- Thermodynamic database selection (TCFE13, etc.)
- Optimization parameters
- Parallel processing settings

## Performance Optimization

- **Parallel Processing**: Utilize multiple CPU cores for thermodynamic calculations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions, suggestions, or support, please contact the project maintainer or create an issue in the GitHub repository.


---

*This toolkit is designed for research and educational purposes in materials science and metallurgy, specifically focusing on advanced high-strength steel development.*
