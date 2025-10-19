# Alloy Design Toolkit

This toolkit provides a computational workflow for designing advanced high-strength steel alloys, specifically focusing on medium-manganese (Med-Mn) steels. It integrates thermodynamic calculations (using TC-Python), kinetic modeling (TC-PRISMA), and multi-objective optimization (NSGA-II) to predict optimal alloy compositions and processing parameters.

---

## Project Structure

```text
alloy-design-toolkit/
├── main.py                           # Main application entry point
├── run_app.py                        # Alternative entry point
├── gui/                              # GUI components
│   ├── main_window.py                # Main window
│   ├── tabs/                         # Application tabs
│   └── dialogs/                      # Configuration dialogs
├── core/                             # Core computational modules
│   ├── workflow_runner.py            # Workflow orchestration
│   ├── phase_calculator.py
│   ├── martensite_calculator.py
│   ├── sfe_calculator.py
│   ├── moo_optimizer.py
│   ├── precipitation_calculator.py
│   └── gm_sorter.py
├── config/                           # Configuration files
│   └── settings.py                   # Default parameters
├── standalone_scripts/               # Individual model scripts
└── utils/                            # Utility functions
```
