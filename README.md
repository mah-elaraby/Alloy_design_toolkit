Alloy Design Toolkit - Automatic Workflow
https://img.shields.io/badge/Python-3.7%252B-blue
https://img.shields.io/badge/Thermo--Calc-Integrated-orange
https://img.shields.io/badge/GUI-Tkinter-green

A comprehensive computational workflow for automated alloy design and optimization, specifically designed for medium-Mn steels and multi-phase alloys.

🚀 Features
Phase Calculations: Thermodynamic equilibrium calculations using TC-Python

Martensite/Austenite Prediction: Ms temperature, martensite fraction, and retained austenite

Stacking Fault Energy: Advanced thermodynamic SFE models

Multi-Objective Optimization: Pareto front analysis with design constraints

Precipitation Kinetics: TC-PRISMA integration for precipitation simulation

Optimal Annealing: Geometric Mean ranking for time-temperature optimization

User-Friendly GUI: Tkinter-based interface with parallel processing

📋 Requirements
Python 3.7+

Thermo-Calc software (TC-Python, TC-PRISMA)

Required Python packages: pandas, numpy, scipy, tkinter

🛠 Installation
Clone the repository:

bash
git clone https://github.com/your-username/alloy-design-toolkit.git
cd alloy-design-toolkit
Install required dependencies:

bash
pip install pandas numpy scipy
Ensure Thermo-Calc software is installed and licensed

🎯 Usage
Automatic Workflow
Run the complete automated workflow through the main GUI:

bash
python main.py
Individual Models
Execute specific models independently from the "Standalone_scripts" tab:

Phase fraction and composition

Martensite-retained austenite model

Stacking fault energy calculator

Multi-objective optimization algorithm

Precipitation kinetics

PRISMA GM sorting

📁 Project Structure
text
alloy-design-toolkit/
├── main.py                 # Main application entry point
├── run_app.py             # Alternative entry point
├── gui/                   # GUI components
│   ├── main_window.py     # Main window
│   ├── tabs/              # Application tabs
│   └── dialogs/           # Configuration dialogs
├── core/                  # Core computational modules
│   ├── workflow_runner.py # Workflow orchestration
│   ├── phase_calculator.py
│   ├── martensite_calculator.py
│   ├── sfe_calculator.py
│   ├── moo_optimizer.py
│   ├── precipitation_calculator.py
│   └── gm_sorter.py
├── config/                # Configuration files
│   └── settings.py        # Default parameters
├── standalone_scripts/    # Individual model scripts
└── utils/                 # Utility functions
🔧 Configuration
The application provides comprehensive configuration dialogs for:

Element composition ranges and temperature settings

Phase selection and database configuration

SFE calculation parameters

MOO constraints and objectives

Precipitation kinetics settings

Annealing optimization criteria

📊 Output
The workflow generates comprehensive Excel files containing:

Phase fractions and compositions

Calculated material properties (Ms, Fm, RA, SFE)

Pareto-optimal alloy solutions

Precipitation kinetics data

Optimal annealing time recommendations

🎨 GUI Interface
The application features a modern, user-friendly interface with three main tabs:

Standalone Scripts: Individual model execution

Computational Workflow: Automated end-to-end workflow

About: Application information and system details

🔬 Scientific Background
This toolkit implements state-of-the-art models for:

Thermodynamic phase equilibrium

Martensite transformation kinetics

Stacking fault energy calculations

Multi-objective optimization for materials design

Precipitation kinetics using classical models

Geometric Mean ranking for multi-criteria decision making

📄 License
This project is licensed for academic and research use. Please contact the author for commercial licensing.

👨‍💻 Author
Mahmoud Elaraby
Materials Science Researcher
Specializing in computational materials design and alloy development

🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

📞 Support
For technical support or questions about the models, please open an issue on GitHub or contact the author directly.
