"""
Configuration settings and default parameters
"""

# Phase calculation parameters
DEFAULT_ELEMENTS = ["Fe", "C", "Mn", "Si", "Al", "Mo", "Nb", "V"]
DEFAULT_PHASES = ["FCC_A1", "BCC_A2", "CEMENTITE_D011"]

# Element composition ranges
DEFAULT_ELEMENT_RANGES = {
    'C': {'start': 0.35, 'end': 0.45, 'step': 0.05},
    'Mn': {'start': 8.0, 'end': 10.1, 'step': 1.0},
    'Si': {'start': 1.0, 'end': 1.0, 'step': 1.0},
    'Al': {'start': 3.0, 'end': 4.1, 'step': 1.0},
    'Mo': {'start': 1.0, 'end': 1.1, 'step': 1.0},
    'Nb': {'start': 0.05, 'end': 0.1, 'step': 0.05},
    'V': {'start': 0.3, 'end': 0.3, 'step': 0.1}
}

# Temperature settings
DEFAULT_TEMP_START = 773
DEFAULT_TEMP_END = 1073
DEFAULT_TEMP_STEP = 1

# Database settings
DEFAULT_PHASE_DATABASE = "TCFE13"
DEFAULT_CACHE_FOLDER = "./cache/"
DEFAULT_WORKERS = 4
DEFAULT_AXIS_MAX_STEP = 1.0

# SFE parameters
DEFAULT_SFE_PARAMS = {
    'sigma': 10,
    'grain_size': 2,
    'lattice_param': 3.6e-10,
    'temperature': 298
}

# MOO constraints
DEFAULT_MOO_CONSTRAINTS = {
    'Ms_min': -100.0,
    'Ms_max': 100.0,
    'Ra_min': 0.1,
    'Ra_max': 0.7,
    'SFE_min': 5.0,
    'SFE_max': 40.0,
    'min_deltaT': 5.0,
    'Cementite_max': 0.0
}

# Precipitation parameters
DEFAULT_PRECIPITATION_PARAMS = {
    'sim_time': 600,
    'matrix_phase': 'FCC_A1',
    'precipitate_phase': 'FCC_A1#2',
    'growth_model': 'PE_AUTOMATIC',
    'nucleation_site': 'Dislocations',
    'tdb': 'TCFE13',
    'kdb': 'MOBFE8'
}

# GM sorting objectives
DEFAULT_GM_OBJECTIVES = {
    'Time [s]': {'goal': 'minimize', 'selected': True},
    'Mean radius (Nb-V)C': {'goal': 'minimize', 'selected': True},
    'Number density (Nb-V)C': {'goal': 'maximize', 'selected': True},
    'Volume fraction (Nb-V)C': {'goal': 'maximize', 'selected': True},
    'Matrix composition Mo': {'goal': 'maximize', 'selected': True},
    'Matrix composition C': {'goal': 'maximize', 'selected': True}
}

# Combine all default parameters
DEFAULT_PARAMETERS = {
    'selected_elements': DEFAULT_ELEMENTS,
    'selected_phases': DEFAULT_PHASES,
    'element_ranges': DEFAULT_ELEMENT_RANGES,
    'temp_start': DEFAULT_TEMP_START,
    'temp_end': DEFAULT_TEMP_END,
    'temp_step': DEFAULT_TEMP_STEP,
    'quenching_temp': 25,
    'phase_database': DEFAULT_PHASE_DATABASE,
    'cache_folder': DEFAULT_CACHE_FOLDER,
    'workers': DEFAULT_WORKERS,
    'axis_max_step': DEFAULT_AXIS_MAX_STEP,
    'calc_fraction': True,
    'calc_composition': True,
    'sfe_params': DEFAULT_SFE_PARAMS,
    'moo_constraints': DEFAULT_MOO_CONSTRAINTS,
    'precipitation_params': DEFAULT_PRECIPITATION_PARAMS,
    'gm_objectives': DEFAULT_GM_OBJECTIVES
}