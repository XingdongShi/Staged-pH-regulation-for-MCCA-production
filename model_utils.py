#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat July 23 13:00:00 2025

Two-stage pH control for chain elongation in microalgae fermentation without external electron donor addition

References
----------
[1] BioSTEAM Documentation: 
    https://biosteam.readthedocs.io/en/latest/API/evaluation/Model.html
[2] Cortes-Peña et al., BioSTEAM: A Fast and Flexible Platform for the Design, 
    Simulation, and Techno-Economic Analysis of Biorefineries under Uncertainty. 
    ACS Sustainable Chem. Eng. 2020, 8 (8), 3302–3310.

@author: Xingdong Shi
@version: 0.0.1
"""

from pandas import DataFrame, read_excel
import chaospy as shape
import numpy as np
import biosteam as bst
from biosteam.evaluation import Model
from .system import microalgae_mcca_sys, microalgae_tea

class MicroalgaeModel(bst.Model):
    def __init__(self, system, metrics=None, specification=None, 
                 parameters=None, retry_evaluation=True, exception_hook='warn',
                 namespace_dict={}):
        Model.__init__(self, system=system, specification=specification, 
                     parameters=parameters, retry_evaluation=retry_evaluation, exception_hook=exception_hook)
        self.namespace_dict = namespace_dict
        # Set metrics after initialization
        if metrics is not None:
            self.metrics = metrics
    
    def load_parameter_distributions(self, distributions, namespace_dict=None):
        namespace_dict = namespace_dict or self.namespace_dict
            
        df = distributions
        if type(df) is not DataFrame:
            df = read_excel(distributions)
            
        create_function = self.create_function
        param = self.parameter
        
        for i, row in df.iterrows():
            name = row['Parameter name']
            element = row['Element']
            kind = row['Kind']
            units = row['Units']
            baseline = row['Baseline']
            shape_data = row['Shape']
            lower, midpoint, upper = row['Lower'], row['Midpoint'], row['Upper']
            load_statements = str(row['Load statement'])
            
            D = None
            if shape_data.lower() in ['triangular', 'triangle']:
                D = shape.Triangle(lower, midpoint, upper)
            elif shape_data.lower() in ['uniform']:
                D = shape.Uniform(lower, upper)
            
            if D is not None:
                param(name=name, 
                      setter=create_function(load_statements, namespace_dict), 
                      element=element, 
                      kind=kind, 
                      units=units,
                      baseline=baseline, 
                      distribution=D)
    
    def create_function(self, code, namespace_dict):
        def wrapper_fn(statement):
            def f(x):
                namespace_dict['x'] = x
                exec(statement, namespace_dict)
            return f
        return wrapper_fn(code) 

def create_unit_groups(system=None):
    """Create mutually exclusive unit groups for a specific system."""
    system = microalgae_mcca_sys if system is None else system
    units_dict = {unit.ID: unit for unit in system.units}
    unit_groups = []
    assigned = set()

    def add_group(name, unit_ids):
        units = [
            units_dict[ID] for ID in unit_ids
            if ID in units_dict and ID not in assigned
        ]
        if units:
            unit_groups.append(bst.UnitGroup(name, units=units))
            assigned.update(unit.ID for unit in units)

    # Area 1: Microalgae cultivation and harvesting
    add_group('Cultivation and harvesting', ['U101'])
    
    # Area 2: Pretreatment and hydrolysis
    pretreatment_unit_ids = ['T201', 'P201', 'M201', 'P202', 'H201', 'R201', 
                            'T202', 'P203', 'R202', 'P204', 'H202', 'T203', 
                            'P205', 'T204', 'P206', 'T205', 'P207', 'S201', 
                            'M202', 'R203', 'H203', 'M203', 'R204', 'S202', 'P208']
    add_group('Pretreatment and hydrolysis', pretreatment_unit_ids)
    
    # Area 3: Conversion
    add_group('Conversion', ['H301', 'T301', 'P301', 'R301', 'R302', 'T302', 'S301'])
    
    # Area 4: Separation
    add_group('Separation', ['M401', 'S402', 'D401', 'D402', 'D403', 'D404'])
    
    # Area 5: Waste treatment - Anaerobic digestion
    add_group('Waste treatment and biogas', [
        'M501', 'R501', 'M502', 'M503', 'M601', 'R601',
        'R602', 'R603', 'S601', 'S602', 'S603', 'M603', 'M602',
        'Upgrading', 'S604',
    ])
    
    # Area 6: Storage
    storage_unit_ids = ['T601', 'P601', 'T602', 'P602', 'T603', 'P603', 'T604', 'P604']
    add_group('Storage', storage_unit_ids)

    # Boiler & turbogenerator
    add_group('BT', ['BT601'])

    # Heat Exchanger Network
    hxn_units = [units_dict[uid] for uid in ['HXN601'] if uid in units_dict and uid not in assigned]
    if hxn_units:
        hxn_group = bst.UnitGroup('Heat exchange network', units=hxn_units)
        hxn_group.filter_savings = False
        unit_groups.append(hxn_group)
        assigned.update(unit.ID for unit in hxn_units)

    # Other facilities
    add_group('Other facilities', ['CT', 'PWC', 'ADP', 'CWP'])

    # Preserve visibility of new units without incorrectly assigning them to
    # wastewater treatment or duplicating a facility in several groups.
    add_group('Unassigned', [unit.ID for unit in system.units if unit.ID not in assigned])

    # Fixed Operating Costs
    foc_group = bst.UnitGroup('Fixed operating costs')
    unit_groups.append(foc_group)
    for ug in unit_groups:
        ug.autofill_metrics(shorthand=False, 
                           electricity_production=False, 
                           electricity_consumption=True,
                           material_cost=True)
    
    # Keep each group tied to its own units.  System-wide TEA quantities are
    # reported separately in tea.py and must not be assigned to BT or storage
    # through a module-level TEA object.
    return unit_groups

def get_unit_groups(system=None):
    return create_unit_groups(system)
