# -*- coding: utf-8 -*-
"""
Created on Sat July 20 10:50:00 2025

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

from thermosteam import Stream
import numpy as np
import pandas as pd
import biosteam as bst
from .system import create_microalgae_MCCA_production_sys
from ._chemicals import chems
from .utils import CFs

__all__ = [
    'LCA',
    'create_microalgae_lca',
    'analyze_system_lca',
    'analyze_all_microalgae_lca',
]

class LCA:
    GWP_key = 'GWP_100'
    FEC_key = 'FEC'

    def __init__(self, 
                 system, 
                 CFs, 
                 main_product, 
                 main_product_chemical_IDs, 
                 boiler, 
                 by_products=[], 
                 complex_feeds={}, 
                 cooling_tower=None, 
                 chilled_water_processing_units=[],
                 functional_unit=None, 
                 functional_quantity_per_h_fn=None, 
                 add_EOL_GWP=False, 
                 input_biogenic_carbon_streams=[]):
        
        self.system = system
        system._LCA = self
        self.chemicals = chemicals = main_product.chemicals
        self.units = self.system.units
        self.flowsheet = self.system.flowsheet
        self.streams = self.system.streams
        self.input_biogenic_carbon_streams = input_biogenic_carbon_streams
        self.add_EOL_GWP = add_EOL_GWP
        self.complex_feeds = complex_feeds
        self.boiler = boiler
        
        kg_product_per_h_fn = lambda: self.main_product.imass[self.main_product_chemical_IDs].sum()
        MJ_LHV_product_per_h_fn = lambda: self.main_product.LHV
        MJ_HHV_product_per_h_fn = lambda: self.main_product.HHV
        if not functional_unit: 
            self.functional_unit = functional_unit = '1 kg'
            self.functional_quantity_per_h_fn = kg_product_per_h_fn
            
        elif not functional_quantity_per_h_fn:
            self.functional_unit = functional_unit
            if functional_unit == '1 kg':
              self.functional_quantity_per_h_fn = kg_product_per_h_fn  
            elif functional_unit == '1 metric ton':
              self.functional_quantity_per_h_fn = lambda: kg_product_per_h_fn() * 1000.
            elif functional_unit == '1 MJ (by LHV)':
              self.functional_quantity_per_h_fn = MJ_LHV_product_per_h_fn  
            elif functional_unit == '1 MJ (by HHV)':
              self.functional_quantity_per_h_fn = MJ_HHV_product_per_h_fn  

        self.CFs = CFs
        self._chemical_IDs = [chem.ID for chem in chemicals]
        self._CF_streams = _CF_streams = {}
        for impact_category in CFs.keys():
            _CF_streams[impact_category] = ic_CF_stream = Stream(f'{system.ID}_{impact_category}_CF_stream')
            for k, v in CFs[impact_category].items():
                if k not in complex_feeds and k != 'Electricity':
                    try:
                        ic_CF_stream.imass[k] = v
                    except:
                        pass
                        
        self._LCA_stream = Stream(f'{system.ID}_LCA_stream')
        self.main_product_chemical_IDs = main_product_chemical_IDs
        self.main_product = main_product
        self.by_products = by_products
        self.chem_IDs = [i.ID for i in chemicals]
        self.CT = self.cooling_tower = cooling_tower
        self.CWP_units = self.chilled_water_processing_units = chilled_water_processing_units
        self._CO2_MW = self.chemicals.CO2.MW
    
    @property
    def products(self):
        return [self.main_product] + self.by_products
    
    @property
    def functional_quantity_per_h(self):
        return self.functional_quantity_per_h_fn()
    
    @property
    def emissions(self):
        emissions = list(self.system.products)
        for i in self.products: 
            if i in emissions:
                emissions.remove(i)
        return emissions
    
    @property
    def feeds(self): 
        return self.system.feeds
        
    @property
    def LCA_stream(self):
        _LCA_stream = self._LCA_stream
        to_mix = list(self.feeds)
        for s, m_k in self.complex_feeds.values(): 
            if s in to_mix:
                to_mix.remove(s)
        _LCA_stream.mix_from(to_mix)
        return _LCA_stream
    
    def get_material_impact_array(self, impact_category):
        return self.LCA_stream.mass*self._CF_streams[impact_category].mass
    
    def get_material_impact(self, impact_category):
        return self.get_material_impact_array(impact_category).sum() / self.functional_quantity_per_h

    @property
    def net_electricity(self):
        return self.system.power_utility.rate
        
    def get_net_electricity_impact(self, impact_category):
        if 'Electricity' in self.CFs[impact_category]:
            return self.net_electricity * self.CFs[impact_category]['Electricity'] / self.functional_quantity_per_h
        return 0
    
    @property
    def EOL_GWP(self): 
        return sum([i.get_atomic_flow('C') for i in [self.main_product] + self.by_products]) *\
            self._CO2_MW/self.functional_quantity_per_h
    
    @property
    def direct_emissions_GWP(self):
        """Direct fossil CO2 from supplementary natural-gas combustion.

        Biogenic methane and sludge supplied to the boiler are excluded because
        their combustion CO2 is biogenic under the adopted cradle-to-gate
        convention. Process outlets are not treated as emissions unless an
        explicit disposal or leakage fate has been modelled.
        """
        return self.boiler.natural_gas.get_atomic_flow('C') *\
            self._CO2_MW / self.functional_quantity_per_h
            
    @property
    def biogenic_emissions_GWP(self): 
        return sum([i.get_atomic_flow('C') for i in self.input_biogenic_carbon_streams]) *\
            self._CO2_MW/self.functional_quantity_per_h
    
    @property
    def direct_non_biogenic_emissions_GWP(self):
        return self.direct_emissions_GWP
    
    def get_complex_feeds_impact(self, impact_category):
        tot_cfs_impact = 0.
        impact_CFs = self.CFs[impact_category]
        for k, (s, mass_kind) in self.complex_feeds.items():
            if k in impact_CFs:
                mass = s.F_mass
                if mass_kind=='dry': mass -= s.imass['Water']
                tot_cfs_impact += impact_CFs[k] * mass
        return tot_cfs_impact/self.functional_quantity_per_h
    
    def get_total_impact(self, impact_category):
        tot_impact = self.get_complex_feeds_impact(impact_category) +\
                     self.get_material_impact(impact_category) +\
                     self.get_net_electricity_impact(impact_category)
        if impact_category in ('GWP', 'GWP_100', 'GWP100'):
            tot_impact += self.direct_non_biogenic_emissions_GWP
            if self.add_EOL_GWP:
                tot_impact += self.EOL_GWP
        return tot_impact

    
    def get_material_impact_breakdown(self, impact_category):
        chemical_impact_dict = {'H2SO4':0, 'NaOH':0, 'NH4OH':0, 'CH4':0, 'CO2':0, 'OleylAlcohol':0}
        LCA_stream = self.LCA_stream
        ic_CF_stream = self._CF_streams[impact_category]
        functional_quantity_per_h = self.functional_quantity_per_h
        chemical_impact_dict_additional = {ID: LCA_stream.imass[ID] * ic_CF_stream.imass[ID] / functional_quantity_per_h
                                for ID in self.chem_IDs}
        for k in list(chemical_impact_dict_additional.keys()):
            if chemical_impact_dict_additional[k] == 0.: del(chemical_impact_dict_additional[k])
        chemical_impact_dict.update(chemical_impact_dict_additional)
        return chemical_impact_dict

    @property
    def GWP(self): 
        return self.get_total_impact(self.GWP_key)
     
    @property
    def material_FEC(self): 
        return self.get_material_impact(self.FEC_key)
     
    @property
    def feedstock_FEC(self):
        return self.get_complex_feeds_impact(self.FEC_key)
     
    @property
    def net_electricity_FEC(self): 
        return self.get_net_electricity_impact(self.FEC_key)
     
    @property
    def material_FEC_breakdown(self):
        return self.get_material_impact_breakdown(self.FEC_key)
     
    @property
    def FEC(self): 
        return self.get_total_impact(self.FEC_key)

    def get_detailed_GWP_breakdown(self):
        breakdown = {}
        gwp_key = self.GWP_key
        gwp_cfs = self.CFs[gwp_key]
        functional_quantity = self.functional_quantity_per_h
        
        # 1. Feedstock impact
        feedstock_impact = self.get_complex_feeds_impact(gwp_key)
        breakdown['feedstock'] = feedstock_impact
        
        # 2. Material impacts from various chemicals
        chemical_impacts = self.get_material_impact_breakdown(gwp_key)
        
        # 3. Report each material separately; do not aggregate minor materials.
        for chem_id, impact in chemical_impacts.items():
            if abs(impact) > 1e-6:
                breakdown[chem_id.lower()] = impact

        # 4. Net electricity impact
        net_electricity_impact = self.get_net_electricity_impact(gwp_key)
        breakdown['net_electricity'] = net_electricity_impact
        
        # 6. Direct fossil emissions from supplementary boiler fuel
        breakdown['direct_non_biogenic_emissions'] = self.direct_non_biogenic_emissions_GWP

        if self.add_EOL_GWP:
            breakdown['end_of_life'] = self.EOL_GWP
        
        breakdown['total_actual'] = self.GWP
        
        return breakdown
    
    def print_GWP_breakdown(self, sort_by_magnitude=True, show_small_contributions=False):
        breakdown = self.get_detailed_GWP_breakdown()
        
        print(f"Total GWP: {breakdown['total_actual']:.4f} kg CO2e/{self.functional_unit}")
        print("\nComponent contributions:")
        print("-" * 80)
        print(f"{'Component Name':<30} {'Contribution':<15} {'Percentage':<10} {'Unit'}")
        print("-" * 80)
        
        display_items = []
        total_gwp = breakdown['total_actual']
        component_map = {
            'feedstock': 'Microalgae feedstock',
            'h2so4': 'Sulfuric acid (H2SO4)',
            'naoh': 'Sodium hydroxide (NaOH)',
            'nh4oh': 'Ammonium hydroxide (NH4OH)',
            'lime': 'Lime',
            'calciumdihydroxide': 'Calcium hydroxide',
            'glucoamylase': 'Glucoamylase',
            'alphaamylase': 'Alpha-amylase',
            'oleylalcohol': 'Oleyl alcohol',
            'ethanol': 'Ethanol',
            'ch4': 'Natural gas (CH4)',
            'co2': 'Carbon dioxide (CO2)',
            'net_electricity': 'Net electricity',
            'direct_non_biogenic_emissions': 'Direct fossil CO2 emissions',
            'end_of_life': 'End-of-life emissions',
        }
        
        for key, value in breakdown.items():
            if key == 'total_actual':
                continue
                
            if not show_small_contributions and abs(value) < 1e-4:
                continue
                
            display_name = component_map.get(key, key)
            percentage = (value / total_gwp * 100) if total_gwp != 0 else 0
            display_items.append((display_name, value, percentage))
        
        if sort_by_magnitude:
            display_items.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for name, value, percentage in display_items:
            unit = f"kg CO2e/{self.functional_unit}"
            print(f"{name:<30} {value:>14.4f} {percentage:>9.1f}% {unit}")
       
        return breakdown
    
    def get_GWP_breakdown_dataframe(self):
        breakdown = self.get_detailed_GWP_breakdown()
        
        component_map = {
            'feedstock': 'Microalgae feedstock',
            'h2so4': 'Sulfuric acid (H2SO4)',
            'naoh': 'Sodium hydroxide (NaOH)',
            'nh4oh': 'Ammonium hydroxide (NH4OH)',
            'lime': 'Lime',
            'calciumdihydroxide': 'Calcium hydroxide',
            'glucoamylase': 'Glucoamylase',
            'alphaamylase': 'Alpha-amylase',
            'oleylalcohol': 'Oleyl alcohol',
            'ethanol': 'Ethanol',
            'ch4': 'Natural gas (CH4)',
            'co2': 'Carbon dioxide (CO2)',
            'net_electricity': 'Net electricity',
            'direct_non_biogenic_emissions': 'Direct fossil CO2 emissions',
            'end_of_life': 'End-of-life emissions',
        }
        
        data = []
        total_gwp = breakdown['total_actual']
        
        for key, value in breakdown.items():
            if key == 'total_actual':
                continue
                
            display_name = component_map.get(key, key)
            percentage = (value / total_gwp * 100) if total_gwp != 0 else 0
            
            data.append({
                'Component': display_name,
                'Original_key': key,
                f'GWP_contribution (kg CO2e/{self.functional_unit})': value,
                'Percentage (%)': percentage,
                'Absolute_value': abs(value)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Absolute_value', ascending=False)
        df = df.drop('Absolute_value', axis=1)
        
        return df

    def get_direct_emissions_breakdown(self):
        return {
            'fossil_natural_gas_combustion_GWP': self.direct_emissions_GWP,
            'EOL_GWP': int(self.add_EOL_GWP) * self.EOL_GWP,
            'direct_non_biogenic_emissions_GWP': self.direct_non_biogenic_emissions_GWP,
        }

    def get_direct_emissions_stream_breakdown(self):
        return [('supplementary_natural_gas_combustion', self.direct_emissions_GWP)]


def create_microalgae_lca(system, main_product, main_product_chemical_IDs, boiler):
        
    cooling_tower = next((unit for unit in system.units 
                         if unit.__class__.__name__ == 'CoolingTower'), None)
    chilled_water_processing_units = [unit for unit in system.units 
                                     if unit.__class__.__name__ == 'ChilledWaterPackage']
    boiler = system.flowsheet.unit.BT601
    
    microalgae_feed = system.flowsheet.stream.microalgae

    complex_feeds = {
        'Microalgae': (microalgae_feed, 'dry'),
    }
    
    input_biogenic_carbon_streams = [microalgae_feed] if microalgae_feed else []
    microalgae_lca = LCA(
        system=system,
        CFs=CFs,
        main_product=main_product,
        main_product_chemical_IDs=main_product_chemical_IDs,
        boiler=boiler,
        by_products=[system.flowsheet.stream.butyric_acid_product,
                     system.flowsheet.stream.acetic_acid_product,
                     system.flowsheet.stream.propionic_acid_product],
        complex_feeds=complex_feeds,
        cooling_tower=cooling_tower,
        chilled_water_processing_units=chilled_water_processing_units,
        functional_unit='1 kg',  
        add_EOL_GWP=False,
        input_biogenic_carbon_streams=input_biogenic_carbon_streams
    )
    
    return microalgae_lca


def analyze_system_lca(system, label='Two-stage pH'):
    """Print and return a cradle-to-gate LCA breakdown for one scenario."""
    system.simulate()
    main_product = system.flowsheet.stream.caproic_acid_product
    lca = create_microalgae_lca(
        system=system,
        main_product=main_product,
        main_product_chemical_IDs=['CaproicAcid'],
        boiler=system.flowsheet.unit.BT601,
    )
    print(f'\n\n========== {label} ==========')
    breakdown_data = lca.print_GWP_breakdown(
        sort_by_magnitude=True, show_small_contributions=False,
    )
    print('\n--- Direct non-biogenic emissions composition ---')
    for key, value in lca.get_direct_emissions_breakdown().items():
        print(f'{key}: {value:.4f} kg CO2e/1 kg C6')
    return {'lca': lca, 'GWP_breakdown': breakdown_data}


def analyze_all_microalgae_lca():
    """Print LCA results for two-stage, control, and ethanol-fed scenarios."""
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
        flowsheet = bst.Flowsheet(f'LCA_{key}')
        bst.main_flowsheet.set_flowsheet(flowsheet)
        results[label] = analyze_system_lca(creator(), label=label)
    return results


if __name__ == "__main__":
    bst.settings.set_thermo(chems)
    analyze_all_microalgae_lca()

