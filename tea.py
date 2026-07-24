# -*- coding: utf-8 -*-
"""
Created on Sat July 20 17:50:00 2025

Two-stage pH control for chain elongation in microalgae fermentation without external electron donor addition

References
----------
[1] BioSTEAM Documentation: 
    https://biosteam.readthedocs.io/en/latest/tutorial/Creating_a_System.html
[2] Cortes-Peña et al., BioSTEAM: A Fast and Flexible Platform for the Design, 
    Simulation, and Techno-Economic Analysis of Biorefineries under Uncertainty. 
    ACS Sustainable Chem. Eng. 2020, 8 (8), 3302–3310.
[3] 3-Hydroxypropionic acid biorefineries project:
    https://github.com/BioSTEAMDevelopmentGroup/Bioindustrial-Park/tree/master/biorefineries/HP
[4] Succinic projest
    https://github.com/BioSTEAMDevelopmentGroup/Bioindustrial-Park/tree/master/biorefineries/succinic

@author: Xingdong Shi
@version: 0.0.9
"""

import pandas as pd
import numpy as np
import biosteam as bst
from biorefineries.tea.cellulosic_ethanol_tea import CellulosicEthanolTEA as create_tea

def compute_labor_cost(dry_tpd: float,
                       base_tpd: float = 2205.0,
                       base_cost: float = 3212962.0) -> float:
    """Return continuously scaled annual labor cost for dry feed capacity."""
    if dry_tpd < 0:
        raise ValueError('dry_tpd must be nonnegative.')
    return base_cost * dry_tpd / base_tpd

# Generic TEA builder for microalgae systems
def microalgae_tea(system):
    u = system.flowsheet.unit
    dry_tpd = u.U101.ins[0].F_mass * 24 / 1000
    tea = create_tea(system=system, IRR=0.10, duration=(2024, 2045),
        depreciation='MACRS7', income_tax=0.21, 
        operating_days=330,
        lang_factor= None, construction_schedule=(0.08, 0.60, 0.32),
        startup_months=3, startup_FOCfrac=1, startup_salesfrac=0.5,
        startup_VOCfrac=0.75, WC_over_FCI=0.05,
        finance_interest=0.08, finance_years=10, finance_fraction=0.4,
        OSBL_units=(u.CT, u.CWP, u.ADP, u.PWC, u.BT601),
        warehouse=0.04, site_development=0.09, additional_piping=0.045,
        proratable_costs=0.10, field_expenses=0.10, construction=0.20,
        contingency=0.10, other_indirect_costs=0.10, 
        labor_cost=compute_labor_cost(dry_tpd),
        labor_burden=0.90, property_insurance=0.007, maintenance=0.03,
        boiler_turbogenerator=u.BT601,
        steam_power_depreciation='MACRS20')
    def update_labor_cost():
        current_dry_tpd = u.U101.ins[0].F_mass * 24 / 1000
        tea.labor_cost = compute_labor_cost(current_dry_tpd)

    u.U101.add_specification(update_labor_cost, run=True)
    return tea

class BTElectricityRevenueConfig:
    def __init__(self, 
                 electricity_price=None,  # None -> bst.PowerUtility.price
                 include_in_material_cost=True,
                 include_in_operating_cost=False):
        self.electricity_price = electricity_price
        self.include_in_material_cost = include_in_material_cost
        self.include_in_operating_cost = include_in_operating_cost


bt_revenue_config = BTElectricityRevenueConfig()


def get_system_power_breakdown(system):
    power_breakdown = {
        'total_consumption': 0,
        'total_production': 0,
        'net_electricity': 0,
        'unit_details': {}
    }
    for unit in system.units:
        if hasattr(unit, 'power_utility') and unit.power_utility:
            pu = unit.power_utility
            consumption = pu.consumption
            production = pu.production
            if consumption != 0 or production != 0:
                power_breakdown['unit_details'][unit.ID] = {
                    'consumption': consumption,
                    'production': production,
                    'net': consumption - production
                }
                power_breakdown['total_consumption'] += consumption
                power_breakdown['total_production'] += production
    power_breakdown['net_electricity'] = (
        power_breakdown['total_consumption'] - power_breakdown['total_production']
    )
    return power_breakdown


def calculate_system_electricity_revenue(system, tea, config: BTElectricityRevenueConfig | None = None):   
    config = config or bt_revenue_config
    power_breakdown = get_system_power_breakdown(system)
    net_power_demand = power_breakdown['net_electricity']  # kW
    operating_hours = tea.operating_hours
    electricity_price = (
        bst.PowerUtility.price
        if config.electricity_price is None else config.electricity_price
    )
    if net_power_demand < 0:
        surplus_electricity = abs(net_power_demand)  # kW
        annual_electricity_revenue = surplus_electricity * electricity_price * operating_hours
        return -annual_electricity_revenue 
    else:
        electricity_cost = net_power_demand * electricity_price * operating_hours
        return electricity_cost 


def _add_net_electricity_column(dataframe, system, tea, config):
    """Return a writable breakdown table with net electricity shown separately."""
    dataframe = dataframe.copy(deep=True)
    if not (config.include_in_material_cost or config.include_in_operating_cost):
        return dataframe
    column = 'Net electricity cost [USD/hr]'
    dataframe[column] = 0.0
    row = 'BT' if 'BT' in dataframe.index else 'System'
    if row == 'System' and row not in dataframe.index:
        dataframe.loc[row] = 0.0
    dataframe.loc[row, column] = calculate_system_electricity_revenue(system, tea, config) / tea.operating_hours
    return dataframe


def _add_fixed_operating_cost(dataframe, tea):
    """Show the system-wide TEA FOC in the material-cost breakdown column."""
    dataframe = dataframe.copy(deep=True)
    column = 'Material cost [USD/hr]'
    row = 'Fixed operating costs'
    if column not in dataframe.columns:
        dataframe[column] = 0.0
    if row not in dataframe.index:
        dataframe.loc[row] = 0.0
    dataframe.loc[row, column] = tea.FOC / tea.operating_hours
    return dataframe


def _to_fraction_dataframe(dataframe, scale_fractions_to_positive_values):
    """Convert a raw unit-group table to column-wise percentages on a copy."""
    values = dataframe.to_numpy(dtype=float, copy=True)
    for index in range(values.shape[1]):
        column = values[:, index]
        denominator = (
            np.abs(column).sum()
            if scale_fractions_to_positive_values else column.sum()
        )
        if abs(denominator) < 1e-12:
            denominator = np.abs(column).sum()
        values[:, index] = 0.0 if abs(denominator) < 1e-12 else column * 100.0 / denominator
    return pd.DataFrame(values, index=dataframe.index, columns=dataframe.columns)


def create_tea_breakdown_data(system, tea, unit_groups, 
                              print_output: bool = False, 
                              fractions: bool = False,
                              config: BTElectricityRevenueConfig | None = None):
    config = config or bt_revenue_config
    dataframe = create_tea_breakdown_dataframe(
        system, tea, unit_groups, fraction=fractions, config=config,
    )
    metric_breakdowns = {
        metric_name: dataframe[metric_name].to_dict()
        for metric_name in dataframe.columns
    }
    if print_output:
        for metric_name, metric_breakdown in metric_breakdowns.items():
            print(f"\n\n----- {metric_name} -----")
            for group_name, value in metric_breakdown.items():
                print(f"{group_name}: {value:.3f}")
    return metric_breakdowns


def create_tea_breakdown_dataframe(system, tea, unit_groups, fraction=True, 
                                   scale_fractions_to_positive_values=False,
                                   config: BTElectricityRevenueConfig | None = None):
    config = config or bt_revenue_config
    # BioSTEAM 2.52 returns a read-only array when asked to normalize groups.
    # Build a raw table first, then normalize a writable copy below.
    df = bst.UnitGroup.df_from_groups(
        unit_groups, 
        fraction=False,
        scale_fractions_to_positive_values=False,
    )
    df = _add_fixed_operating_cost(df, tea)
    df = _add_net_electricity_column(df, system, tea, config)
    if fraction:
        df = _to_fraction_dataframe(df, scale_fractions_to_positive_values)
    return df


def get_unit_groups_for_system(system):
    try:
        # Use the project area map so external-ethanol storage and wastewater
        # mixer units are assigned once, rather than shown as a duplicate
        # ``Unassigned`` material cost in the TEA table.
        from .model_utils import get_unit_groups
        return get_unit_groups(system)
    except Exception:
        try:
            return bst.UnitGroup.group_by_area(system)
        except Exception:
            return []


def get_cost_breakdown_by_category(tea):
    breakdown = {}
    breakdown['Installed equipment cost'] = tea.installed_equipment_cost / 1e6
    breakdown['ISBL installed equipment cost'] = getattr(tea, 'ISBL_installed_equipment_cost', 0) / 1e6
    breakdown['OSBL installed equipment cost'] = getattr(tea, 'OSBL_installed_equipment_cost', 0) / 1e6
    breakdown['Fixed operating cost'] = tea.FOC / 1e6
    breakdown['Variable operating cost'] = tea.VOC / 1e6
    breakdown['Material cost'] = tea.material_cost / 1e6
    breakdown['Utility cost'] = tea.utility_cost / 1e6
    breakdown['Total capital investment'] = tea.TCI / 1e6
    breakdown['Fixed capital investment'] = tea.FCI / 1e6
    breakdown['Total depreciable capital'] = tea.TDC / 1e6
    breakdown['Direct permanent investment'] = tea.DPI / 1e6
    return breakdown


def analyze_system_tea_breakdown(system, label='Two-stage pH'):
    """Print and return the TEA breakdown for one fully simulated scenario."""
    system.simulate()
    unit_groups = get_unit_groups_for_system(system)
    tea_obj = microalgae_tea(system)
    product = system.flowsheet.stream.caproic_acid_product
    print(f'\n\n========== {label} ==========' )
    print(f'MPSP: {tea_obj.solve_price(product):.3f} USD/kg C6')
    _ = create_tea_breakdown_data(
        system=system,
        tea=tea_obj,
        unit_groups=unit_groups,
        print_output=True
    )
    df_breakdown = create_tea_breakdown_dataframe(
        system=system,
        tea=tea_obj,
        unit_groups=unit_groups,
        fraction=True,
        scale_fractions_to_positive_values=False
    )
    cost_breakdown = get_cost_breakdown_by_category(tea_obj)
    results = {
        'breakdown_dataframe': df_breakdown,
        'cost_breakdown': cost_breakdown,
        'unit_groups': unit_groups,
        'tea_system': system,
        'system_electricity_revenue': calculate_system_electricity_revenue(system, tea_obj, bt_revenue_config),
    }
    return results


def analyze_microalgae_tea_breakdown():
    """Backward-compatible two-stage pH TEA breakdown entry point."""
    from .system import create_microalgae_MCCA_production_sys
    flowsheet = bst.Flowsheet('TEA_two_stage')
    bst.main_flowsheet.set_flowsheet(flowsheet)
    return analyze_system_tea_breakdown(
        create_microalgae_MCCA_production_sys(), label='Two-stage pH',
    )


def analyze_all_microalgae_tea_breakdowns():
    """Print TEA breakdowns for two-stage, control, and ethanol-fed systems."""
    from .system import (
        create_microalgae_MCCA_control_sys,
        create_microalgae_MCCA_production_sys,
    )
    from .system_ethanol import create_microalgae_MCCA_ethanol_sys

    scenarios = (
        ('Two-stage pH', create_microalgae_MCCA_production_sys),
        ('Control (pH 7)', create_microalgae_MCCA_control_sys),
        ('External ethanol', create_microalgae_MCCA_ethanol_sys),
    )
    results = {}
    for key, (label, creator) in enumerate(scenarios):
        flowsheet = bst.Flowsheet(f'TEA_breakdown_{key}')
        bst.main_flowsheet.set_flowsheet(flowsheet)
        results[label] = analyze_system_tea_breakdown(creator(), label=label)
    return results


if __name__ == "__main__":
    analyze_all_microalgae_tea_breakdowns()
