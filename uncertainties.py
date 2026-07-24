#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat August 23 13:00:00 2025

Two-stage pH control for chain elongation in microalgae fermentation without external electron donor addition

References
----------
[1] BioSTEAM Documentation: 
    https://biosteam.readthedocs.io/en/latest/API/evaluation/Model.html
[2] Cortes-Peña et al., BioSTEAM: A Fast and Flexible Platform for the Design, 
    Simulation, and Techno-Economic Analysis of Biorefineries under Uncertainty. 
    ACS Sustainable Chem. Eng. 2020, 8 (8), 3302–3310.
[3] succinic biorefineries project:
    https://github.com/BioSTEAMDevelopmentGroup/Bioindustrial-Park/tree/master/biorefineries/succinic

@author: Xingdong Shi
@version: 0.0.1
"""

from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
import biosteam as bst
from ._chemicals import chems
from .tea import microalgae_tea as create_tea_for_system
from .lca import create_microalgae_lca
from importlib import import_module
from .model_utils import MicroalgaeModel
from .tea import get_unit_groups_for_system
from .yield_capacity_analysis import (
    generate_tea_breakdown_figures,
    generate_uncertainty_figures,
)
from biosteam.evaluation import Metric
from datetime import datetime
from biosteam.utils import Timer
import os

microalgae_filepath = os.path.dirname(__file__)
microalgae_results_filepath = os.path.join(microalgae_filepath, 'analyses', 'results')
microalgae_figures_filepath = os.path.join(microalgae_filepath, 'analyses', 'figures')

# Create results directory if it doesn't exist
if not os.path.exists(microalgae_results_filepath):
    os.makedirs(microalgae_results_filepath)
if not os.path.exists(microalgae_figures_filepath):
    os.makedirs(microalgae_figures_filepath)

def _get_system_creators():
    system_module = import_module('.system', __package__)
    ethanol_module = import_module('.system_ethanol', __package__)
    return [
        ('baseline', system_module.create_microalgae_MCCA_production_sys),
        ('control', system_module.create_microalgae_MCCA_control_sys),
        ('ethanol', ethanol_module.create_microalgae_MCCA_ethanol_sys),
    ]

def build_context_for_system(microalgae_mcca_sys, sys_label='baseline'):
    u = microalgae_mcca_sys.flowsheet.unit
    s = microalgae_mcca_sys.flowsheet.stream
    main_product = getattr(s, 'caproic_acid_product', None)
    if main_product is None:
        raise AttributeError('caproic_acid_product not found in flowsheet streams.')
    boiler = next((unit for unit in microalgae_mcca_sys.units if ('BT' in unit.ID)), None)
    lca = create_microalgae_lca(microalgae_mcca_sys, main_product, ['CaproicAcid'], boiler)
    tea_obj = create_tea_for_system(microalgae_mcca_sys)
    metrics = [
        Metric('MPSP', lambda: tea_obj.solve_price(main_product), '$/kg', element='TEA'),
        Metric('TCI', lambda: tea_obj.TCI/1e6, 'MM$', element='TEA'),
        Metric('VOC', lambda: tea_obj.VOC/1e6, 'MM$/y', element='TEA'),
        Metric('FOC', lambda: tea_obj.FOC/1e6, 'MM$/y', element='TEA'),
        Metric('GWP', lambda: lca.GWP, 'kg CO2-eq/kg', element='LCA'),
        Metric('FEC', lambda: lca.FEC, 'MJ/kg', element='LCA'),
    ]
    namespace_dict = {
        'microalgae_sys': microalgae_mcca_sys,
        'microalgae_tea': tea_obj,
        'u': u,
        's': s,
        'lca': lca,
        'bst': bst,
        'np': np,
        'PowerUtility': bst.PowerUtility,
    }
    # Provide system-specific alias names expected by parameter distribution 'Load statement'
    if sys_label == 'baseline':
        namespace_dict['microalgae_mcca_sys'] = microalgae_mcca_sys
        namespace_dict['microalgae_tea_baseline'] = tea_obj
    elif sys_label == 'control':
        namespace_dict['microalgae_mcca_control_sys'] = microalgae_mcca_sys
        namespace_dict['microalgae_tea_control'] = tea_obj
    elif sys_label == 'ethanol':
        namespace_dict['microalgae_mcca_ethanol_sys'] = microalgae_mcca_sys
        namespace_dict['microalgae_tea_ethanol'] = tea_obj
    # No universal aliases; only system-specific names are exposed to match each system's distribution file.
    model = MicroalgaeModel(microalgae_mcca_sys, metrics=metrics, namespace_dict=namespace_dict)
    return model, tea_obj, lca, namespace_dict


def restore_baseline_state(system, parameters):
    """Restore all parameters after model evaluation leaves a sampled state."""
    for parameter in parameters:
        parameter.setter(parameter.baseline)
    system.simulate()

 

# %% 

# Number of Monte Carlo samples used for the formal uncertainty analysis.
N_simulations_per_mode = 2000

percentiles = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1]

notification_interval = 10

results_dict = {'Baseline':{'MPSP':{}, 'GWP100a':{}, 'FEC':{}, 
                            'GWP Breakdown':{}, 'FEC Breakdown':{},},
                'Uncertainty':{'MPSP':{}, 'GWP100a':{}, 'FEC':{}},
                'Sensitivity':{'Spearman':{'MPSP':{}, 'GWP100a':{}, 'FEC':{}}},}

modes = []  # will be populated with system labels as they are processed
# Store context needed for plotting per mode
_mode_to_system = {}
_mode_to_tea = {}
_mode_to_unit_groups = {}
_mode_to_filebase = {}

# Each scenario has its own parameter distribution workbook.
parameter_distributions_files = {
    'baseline': 'parameter_distributions.xlsx',
    'control': 'parameter_distributions_control.xlsx',
    'ethanol': 'parameter_distributions_ethanol.xlsx',
}

#%%

timer = Timer('timer')
RUN_UNCERTAINTY_ANALYSIS = __name__ == '__main__'
if RUN_UNCERTAINTY_ANALYSIS:
    timer.start()

# Set seed to make sure each time the same set of random numbers will be used
np.random.seed(3221)

systems_to_run = _get_system_creators() if RUN_UNCERTAINTY_ANALYSIS else []
if RUN_UNCERTAINTY_ANALYSIS and not systems_to_run:
    raise RuntimeError('No microalgae systems available to analyze.')

for (sys_label, creator_fn) in systems_to_run:
    print(f"\n===== Running uncertainty analysis for system: {sys_label} =====")
    bst.settings.set_thermo(chems)
    scenario_flowsheet = bst.Flowsheet(f'MCCA_{sys_label}')
    bst.main_flowsheet.set_flowsheet(scenario_flowsheet)
    microalgae_mcca_sys = creator_fn()
    microalgae_mcca_sys.simulate()
    model, tea_obj, lca, namespace_dict = build_context_for_system(microalgae_mcca_sys, sys_label=sys_label)
    mode = sys_label
    if mode not in modes:
        modes.append(mode)
    # Choose distribution file for this system label
    fname = parameter_distributions_files[sys_label]
    parameter_distributions_filename = os.path.join(microalgae_filepath, fname)
    if not os.path.isfile(parameter_distributions_filename):
        raise FileNotFoundError(f"Parameter distributions file not found for '{sys_label}': {parameter_distributions_filename}")
    
    model.parameters = ()
    parameter_distributions = pd.read_excel(parameter_distributions_filename)

    # The baseline two-stage process intentionally has no external ethanol
    # feed.  Older distribution workbooks may still contain ``s.ethanol``
    # from the ethanol-fed comparison system; do not register that stale
    # parameter against the baseline flowsheet.
    if sys_label != 'ethanol':
        load_statements = parameter_distributions['Load statement'].fillna('').astype(str)
        stale_ethanol = load_statements.str.contains(r'\bs\.ethanol\b', regex=True)
        if stale_ethanol.any():
            skipped = ', '.join(parameter_distributions.loc[stale_ethanol, 'Parameter name'].astype(str))
            print(f"Skipping parameters not applicable to '{sys_label}': {skipped}")
            parameter_distributions = parameter_distributions.loc[~stale_ethanol].copy()

    model.load_parameter_distributions(parameter_distributions, namespace_dict)
    parameters = model.get_parameters()
    restore_baseline_state(microalgae_mcca_sys, parameters)
    
    samples = model.sample(N=N_simulations_per_mode, rule='L')
    model.load_samples(samples)
    
    
    model.exception_hook = 'warn'
    baseline_initial = model.metrics_at_baseline()
    baseline = pd.DataFrame(data=np.array([[i for i in baseline_initial.values],]), 
                            columns=baseline_initial.keys())
    
    results_dict['Baseline']['MPSP'][mode] = tea_obj.solve_price(microalgae_mcca_sys.flowsheet.stream.caproic_acid_product)
    print(f"MPSP: ${results_dict['Baseline']['MPSP'][mode]:.2f}/kg")
    results_dict['Baseline']['GWP100a'][mode] = tot_GWP = lca.GWP
    print(f"GWP: {results_dict['Baseline']['GWP100a'][mode]:.4f} kg CO2-eq/kg")
    results_dict['Baseline']['FEC'][mode] = tot_FEC = lca.FEC
    print(f"FEC: {results_dict['Baseline']['FEC'][mode]:.4f} MJ/kg")
          
    model.evaluate(notify=notification_interval, autoload=None, autosave=None, file=None)
    
        
    # Model evaluation leaves the system at the final random sample. Restore
    # the named baseline before saving deterministic results or reusing it.
    restore_baseline_state(microalgae_mcca_sys, parameters)
    baseline_end = model.metrics_at_baseline()
    dateTimeObj = datetime.now()
    minute = '0' + str(dateTimeObj.minute) if len(str(dateTimeObj.minute))==1 else str(dateTimeObj.minute)
    file_to_save = os.path.join(microalgae_results_filepath,
        f'_microalgae_{sys_label}_{dateTimeObj.year}.{dateTimeObj.month}.{dateTimeObj.day}-{dateTimeObj.hour}.{minute}'\
        + f'_{N_simulations_per_mode}sims')
    # Save context for later plotting
    _mode_to_system[mode] = microalgae_mcca_sys
    _mode_to_tea[mode] = tea_obj
    _mode_to_unit_groups[mode] = get_unit_groups_for_system(microalgae_mcca_sys)
    _mode_to_filebase[mode] = file_to_save
    
    baseline.index = ('initial', )
    baseline.to_excel(file_to_save+'_'+mode+'_0_baseline.xlsx')
    
    # Parameters
    parameters = model.get_parameters()
    index_parameters = len(model.get_baseline_sample())
    parameter_values = model.table.iloc[:, :index_parameters].copy()
    
    #%%
    
    # TEA results (split TEA vs LCA by Metric.element)
    for index_TEA, i in enumerate(model.metrics):
        if getattr(i, 'element', None) == 'LCA':
            break
    else:
        index_TEA = len(model.metrics) - 2  # Fallback: assume last 2 are LCA

    index_TEA = index_parameters + index_TEA
    TEA_results = model.table.iloc[:, index_parameters:index_TEA].copy()
    TEA_percentiles = TEA_results.quantile(q=percentiles)
    
    # LCA_results
    LCA_results = model.table.iloc[:, index_TEA::].copy()
    LCA_percentiles = LCA_results.quantile(q=percentiles)
    
    # # Spearman's rank correlation
    table = model.table
    model.table = model.table.dropna()
    
    spearman_results = model.spearman()
    spearman_results.columns = pd.Index([i.name_with_units for i in model.metrics])
    
    model.table = table
    
    # Calculate the cumulative probabilities of each parameter
    probabilities = {}
    for i in range(index_parameters):
        p = parameters[i]
        p_values = parameter_values.iloc[:, 2*i]
        probabilities[p.name] = p.distribution.cdf(p_values)
        parameter_values.insert(loc=2*i+1, 
                          column=(parameter_values.iloc[:, 2*i].name[0], 'Probability'), 
                          value=probabilities[p.name],
                          allow_duplicates=True)
    
    #%%
    with pd.ExcelWriter(file_to_save+'_'+mode+'_1_full_evaluation.xlsx') as writer:
        parameter_values.to_excel(writer, sheet_name='Parameters')
        TEA_results.to_excel(writer, sheet_name='TEA results')
        TEA_percentiles.to_excel(writer, sheet_name='TEA percentiles')
        LCA_results.to_excel(writer, sheet_name='LCA results')
        LCA_percentiles.to_excel(writer, sheet_name='LCA percentiles')
        spearman_results.to_excel(writer, sheet_name='Spearman')
        model.table.to_excel(writer, sheet_name='Raw data')
    
    # Extract results for plotting
    def _find_col_by_substring(columns, key):
        for col in columns:
            if key in str(col):
                return col
        return None

    mpsp_col = _find_col_by_substring(model.table.columns, 'MPSP')
    gwp_col = _find_col_by_substring(model.table.columns, 'GWP')
    fec_col = _find_col_by_substring(model.table.columns, 'FEC')
    
    if mpsp_col is not None:
        results_dict['Uncertainty']['MPSP'][mode] = pd.Series(model.table[mpsp_col])
    else:
        results_dict['Uncertainty']['MPSP'][mode] = pd.Series([results_dict['Baseline']['MPSP'][mode]] * len(model.table))

    if gwp_col is not None:
        results_dict['Uncertainty']['GWP100a'][mode] = pd.Series(model.table[gwp_col])
    else:
        results_dict['Uncertainty']['GWP100a'][mode] = pd.Series([results_dict['Baseline']['GWP100a'][mode]] * len(model.table))

    if fec_col is not None:
        results_dict['Uncertainty']['FEC'][mode] = pd.Series(model.table[fec_col])
    else:
        results_dict['Uncertainty']['FEC'][mode] = pd.Series([results_dict['Baseline']['FEC'][mode]] * len(model.table))
    
    # Spearman correlations for sensitivity analysis
    df_rho, df_p = model.spearman_r()

    mpsp_col_r = _find_col_by_substring(df_rho.columns, 'MPSP') if mpsp_col is None else (mpsp_col if mpsp_col in df_rho.columns else _find_col_by_substring(df_rho.columns, 'MPSP'))
    gwp_col_r = _find_col_by_substring(df_rho.columns, 'GWP') if gwp_col is None else (gwp_col if gwp_col in df_rho.columns else _find_col_by_substring(df_rho.columns, 'GWP'))
    fec_col_r = _find_col_by_substring(df_rho.columns, 'FEC') if fec_col is None else (fec_col if fec_col in df_rho.columns else _find_col_by_substring(df_rho.columns, 'FEC'))

    results_dict['Sensitivity']['Spearman']['MPSP'][mode] = df_rho[mpsp_col_r] if mpsp_col_r is not None else pd.Series()
    results_dict['Sensitivity']['Spearman']['GWP100a'][mode] = df_rho[gwp_col_r] if gwp_col_r is not None else pd.Series()
    results_dict['Sensitivity']['Spearman']['FEC'][mode] = df_rho[fec_col_r] if fec_col_r is not None else pd.Series()
            
# %% Figures
if RUN_UNCERTAINTY_ANALYSIS and modes:
    result_files = {
        mode: f'{_mode_to_filebase[mode]}_{mode}_1_full_evaluation.xlsx'
        for mode in modes
    }
    figure_outputs = generate_uncertainty_figures(
        result_files, microalgae_figures_filepath,
    )
    figure_outputs.extend(generate_tea_breakdown_figures(
        _mode_to_system, _mode_to_tea, _mode_to_unit_groups,
        microalgae_figures_filepath,
    ))
    for output in figure_outputs:
        print(f'Generated: {output}')
    print(f'\nAnalysis completed. Timer: {timer.measure():.2f} seconds')
