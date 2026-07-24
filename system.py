# -*- coding: utf-8 -*-
"""
Created on Sat June 08 13:00:00 2026

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
[5] Microalgae project:
    https://github.com/XingdongShi/Microalgae-to-MCCA-without-external-electron-donor

@author: Xingdong Shi
@version: 0.0.8
"""

import biosteam as bst
import numpy as np
import thermosteam as tmo
from biosteam import Stream, SystemFactory
from biosteam.units import Pump, StorageTank, HXutility, Mixer, Splitter, MultiStageMixerSettlers
from biosteam.facilities import AirDistributionPackage, ProcessWaterCenter, CoolingTower, ChilledWaterPackage, HeatExchangerNetwork
from biosteam import main_flowsheet
import os
from datetime import datetime
from .units import (
    FeedstockPreprocessing, AcidPretreatmentReactor, Saccharification, SolidLiquidSeparation,
    MCCAFermentation, MCCAFermentation_acidic, MCCAFermentation_control,
    NeutralizationTank, AnaerobicDigestion
)
from .utils import price
from ._chemicals import chems
from .tea import microalgae_tea
from .streams import microalgae_feed
import warnings
# Filter out specific warnings
warnings.filterwarnings("ignore", message="phase equilibrium solution results in negative flow rates")
warnings.filterwarnings("ignore", message=".*has no defined Dortmund groups.*")
warnings.filterwarnings("ignore", message=".*has been replaced in registry")
warnings.filterwarnings("ignore", category=bst.exceptions.CostWarning)
warnings.filterwarnings("ignore", message=".*moisture.*is smaller than the desired.*")
warnings.filterwarnings("ignore", message=".*moisture of influent.*is smaller than the desired.*")

# Set up the main flowsheet and thermodynamic environment
bst.settings.set_thermo(chems)
main_flowsheet.clear()
flowsheet = bst.Flowsheet('MCCA')
bst.main_flowsheet.set_flowsheet(flowsheet)

# System settings
bst.System.default_converge_method = 'wegstein'
bst.System.default_maxiter = 2000
bst.System.default_molar_tolerance = 1e-4
bst.System.default_relative_molar_tolerance = 1e-4 # supersedes absolute tolerance
bst.System.strict_convergence = True # True => throw exception if system does not converge; False => continue with unconverged system

@SystemFactory(
    ID='Microalgae_MCCA_production',
    ins=[dict(microalgae_feed, thermo=chems)],
    outs=[
          dict(ID='butyric_acid_product', thermo=chems), 
          dict(ID='caproic_acid_product', thermo=chems),
          dict(ID='acetic_acid_product', thermo=chems),
          dict(ID='propionic_acid_product', thermo=chems)]
    )
def create_microalgae_MCCA_production_sys(ins, outs, fermentation_mode='two_stage'):
    # Set the thermodynamic package explicitly
    tmo.settings.set_thermo(chems)
    
    # Main feed and product
    microalgae_feed, = ins
    (   butyric_acid_product,
        caproic_acid_product,
        acetic_acid_product,
        propionic_acid_product) = outs
    
    # Calculate all required stream properties based on feed
    microalgae_mass = microalgae_feed.F_mass
    # Added water required for a 4 wt% dry-microalgae slurry.
    microalgae_water_mass = microalgae_mass * (1 - 0.04) / 0.04
    microalgae_water = Stream('microalgae_water', Water=microalgae_water_mass, units='kg/hr')
    # H2SO4 for microalgae biomass hydrolysis
    acid_loading = 1.47  # g H2SO4 / g microalgae
    acid_purity = 0.93 
    water_mass = microalgae_mass * (1 - 0.04) / 0.04
    pure_H2SO4 = microalgae_mass * acid_loading
    acid_solution_mass = pure_H2SO4 / acid_purity
    water_mass_acid = acid_solution_mass * (1 - acid_purity)
    SulfuricAcid = Stream('sulfuricacid', H2SO4=pure_H2SO4, Water=water_mass_acid, units='kg/hr', price=price['SulfuricAcid'])
    
    # Store reference for later specification
    _sulfuric_acid_stream = SulfuricAcid
    # Ammonium Hydroxide for neutralization
    h2so4_mol = pure_H2SO4 * 1000 / 98 # mol mass
    nh4oh_mol = h2so4_mol * 0.08 # preadjustment
    nh4oh_mass = nh4oh_mol * 35 / 1000 # mol mass to mass
    ammonium_hydroxide = Stream('ammonium_hydroxide', NH4OH=nh4oh_mass, units='kg/hr', price=price['AmmoniumHydroxide'])
    # Enzyme dosages
    gluco_coeff = 0.0011  # kg enzyme per kg microalgae
    alpha_coeff = 0.0082  # kg enzyme per kg microalgae
    glucoamylase_mass = float(microalgae_mass * gluco_coeff)
    alpha_amylase_mass = float(microalgae_mass * alpha_coeff)
    glucoamylase = Stream('glucoamylase', GlucoAmylase=glucoamylase_mass, units='kg/hr', price=price['GlucoAmylase'])
    alpha_amylase = Stream('alpha_amylase', AlphaAmylase=alpha_amylase_mass, units='kg/hr', price=price['AlphaAmylase'])
    
    # Store references for later specification
    _glucoamylase_stream = glucoamylase
    _alpha_amylase_stream = alpha_amylase
    # NaOH is retained only for the two upstream saccharification pH settings;
    naoh1_mass = microalgae_mass * 0.015 # first saccharification setting
    naoh2_mass = microalgae_mass * 0.015 # second saccharification setting
    total_naoh_mass = naoh1_mass + naoh2_mass
    naoh = Stream('naoh', NaOH=total_naoh_mass, units='kg/hr', price=price['NaOH'])
    # OleylAlcohol for extraction
    fresh_oleylalcohol = Stream('fresh_oleylalcohol', OleylAlcohol=200, units='kg/hr', price=price['OleylAlcohol'])
    oleylalcohol_recycle = Stream('oleylalcohol_recycle', OleylAlcohol=50, units='kg/hr')
    oleylalcohol_recycle_placeholder = bst.Stream()
    oleylalcohol_feed = bst.Stream()
    # Assign prices to product streams
    butyric_acid_product.price = price['ButyricAcid']
    caproic_acid_product.price = price['CaproicAcid']
    acetic_acid_product.price = price['AceticAcid']
    propionic_acid_product.price = price['PropionicAcid']

    # =====================
    # Area 1: Microalgae process
    # =====================
    U101 = FeedstockPreprocessing('U101', microalgae_feed, thermo=chems)

    # =====================
    # Area 2: Hydrolysis and saccharification
    # =====================
    T201 = StorageTank('T201', SulfuricAcid)
    P201 = Pump('P201', T201-0, P=5e5, pump_type='Default')
    M201 = Mixer('M201', [P201-0, microalgae_water, U101-0])
    P202 = Pump('P202', M201-0)
    H201 = HXutility('H201', P202-0, T=121+273.15)
    R201 = AcidPretreatmentReactor('R201', H201-0)
    T202 = StorageTank('T202', ammonium_hydroxide)
    P203 = Pump('P203', T202-0)
    R202 = NeutralizationTank('R202', [R201-0, P203-0])
    P204 = Pump('P204', R202-0)
    H202 = HXutility('H202', P204-0, T=55+273.15)
    T203 = StorageTank('T203', glucoamylase)
    P205 = Pump('P205', T203-0, P=25e5, pump_type='Default', dP_design=24e5, ignore_NPSH=True)
    T204 = StorageTank('T204', alpha_amylase)
    P206 = Pump('P206', T204-0, P=25e5, pump_type='Default', dP_design=24e5, ignore_NPSH=True)
    T205 = StorageTank('T205', naoh)
    P207 = Pump('P207', T205-0, P=25e5, pump_type='Default', dP_design=24e5, ignore_NPSH=True)
    S201 = Splitter('S201', P207-0, split=naoh1_mass/total_naoh_mass)
    M202 = Mixer('M202', [H202-0, P205-0, S201-0])
    R203 = Saccharification('R203', M202-0)
    H203 = HXutility('H203', R203-0, T=90+273.15)
    M203 = Mixer('M203', [H203-0, P206-0, S201-1])
    R204 = Saccharification('R204', M203-0)
    S202 = SolidLiquidSeparation('S202', R204-0)
    P208 = Pump('P208', S202-0)

    # =====================
    # Area 3: Fermentation for MCCA production
    # =====================
    H301 = HXutility('H301', P208-0, T=37+273.15)
    
    if fermentation_mode == 'two_stage':
        R301 = MCCAFermentation_acidic(
            'R301', H301-0, microalgae_mass_flow=microalgae_mass,
            tau=7.5 * 24, pH=4.5,
        )
        R302 = MCCAFermentation(
            'R302', R301-0, microalgae_mass_flow=microalgae_mass,
            tau=7.5 * 24, pH=7.0,
        )
        fermentation_reactors = (R301, R302)
        fermentation_gases = (R301-1, R302-1)
        fermentation_wastes = (R301-2, R302-2)

        def set_total_HRT(days):
            stage_tau = float(days) * 24 / 2
            R301.tau = stage_tau
            R302.tau = stage_tau
    elif fermentation_mode == 'control':
        # Single-stage experimental control at pH 7.0.
        R302 = MCCAFermentation_control(
            'R302', H301-0, microalgae_mass_flow=microalgae_mass,
            tau=15 * 24, pH=7.0,
        )
        fermentation_reactors = (R302,)
        fermentation_gases = (R302-1,)
        fermentation_wastes = (R302-2,)

        def set_total_HRT(days):
            R302.tau = float(days) * 24
    else:
        raise ValueError("fermentation_mode must be 'two_stage' or 'control'.")

    # Add C6 yield factor specification
    @R302.add_specification(run=True)
    def set_C6_yield_factor(factor=1.0):
        R302.caproic_acid_yield_factor = factor
    
    # Bind specification functions as methods for parameter loading
    R302.set_C6_yield_factor = lambda x: setattr(R302, 'caproic_acid_yield_factor', x)

    R302.set_total_HRT = set_total_HRT

    # Gas yields are defined inside the fermentation-unit classes.
    T302 = Mixer('T302', fermentation_gases)
    S301 = SolidLiquidSeparation('S301', R302-0)

    # =====================
    # Area 4: Product extraction
    # =====================
    M401 = Mixer('M401', [fresh_oleylalcohol, oleylalcohol_recycle_placeholder], oleylalcohol_feed)
    M401.oleylalcohol_per_dry_microalgae = 200 / 5000
    @M401.add_specification(run=True)
    def adjust_fresh_oleylalcohol():
        total_oleylalcohol = (
            microalgae_feed.F_mass
            * M401.oleylalcohol_per_dry_microalgae
        )
        recycle = oleylalcohol_recycle.imass['OleylAlcohol']
        if recycle > total_oleylalcohol:
            oleylalcohol_recycle.imass['OleylAlcohol'] = total_oleylalcohol
            recycle = total_oleylalcohol
        fresh = max(total_oleylalcohol - recycle, 0.0)
        fresh_oleylalcohol.imass['OleylAlcohol'] = fresh
    IDs = ['Water', 'AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid', 'CaproicAcid', 'OleylAlcohol']
    K = np.array([1/5000, 2000/1, 3000/1, 5000/1, 5000/1, 5000/1, 100000/1])
    S402 = MultiStageMixerSettlers(
        'S402',
        partition_data={'K': K, 'IDs': IDs},
        N_stages=5,
        ins=[S301-0, M401-0]
    )

    # Keep extraction efficiency persistent across system simulations.
    original_K = K.copy()
    S402.extraction_efficiency = 1.0

    @S402.add_specification(run=True)
    def apply_extraction_efficiency():
        efficiency = S402.extraction_efficiency
        S402.partition_data = {'K': original_K * efficiency, 'IDs': IDs}
    
    def set_extraction_efficiency(efficiency):
        efficiency = float(efficiency)
        if efficiency <= 0:
            raise ValueError('Extraction efficiency factor must be positive.')
        S402.extraction_efficiency = efficiency
        apply_extraction_efficiency()

    S402.set_extraction_efficiency = set_extraction_efficiency

    D401 = bst.BinaryDistillation('D401', S402-0, LHK=('AceticAcid', 'PropionicAcid'),
            Lr=0.99, Hr=0.99, k=1.2,
            partial_condenser=False,
            is_divided=True)
    D401.check_LHK = False

    D402 = bst.BinaryDistillation('D402', D401-1,
            LHK=('PropionicAcid', 'ButyricAcid'),
            Lr=0.99, Hr=0.99, k=1.2,
            partial_condenser=False,
            is_divided=True)
    D402.check_LHK = False

    D403 = bst.BinaryDistillation('D403', D402-1,
            LHK=('ButyricAcid', 'CaproicAcid'),
            Lr=0.99, Hr=0.99, k=1.2,
            partial_condenser=False,
            is_divided=True)
    D403.check_LHK = False

    D404 = bst.BinaryDistillation('D404', D403-1, ['', oleylalcohol_recycle], LHK=('CaproicAcid', 'OleylAlcohol'),
            Lr=0.99, Hr=0.99, k=1.2,
            partial_condenser=False,
            is_divided=True
        )

    M401.ins[1] = oleylalcohol_recycle
    if oleylalcohol_recycle_placeholder in M401.ins:
        M401.ins[M401.ins.index(oleylalcohol_recycle_placeholder)] = oleylalcohol_recycle
    D404.check_LHK = False

    # Keep the shared distillation-recovery factor persistent across simulations.
    distillation_units = [D401, D402, D403, D404]
    D404.distillation_efficiency = 1.0
    
    @D404.add_specification(run=True)
    def apply_distillation_efficiency():
        efficiency = D404.distillation_efficiency
        for unit in distillation_units:
            unit.Lr = 0.99 * efficiency
            unit.Hr = 0.99 * efficiency
    
    def set_distillation_efficiency(efficiency):
        efficiency = float(efficiency)
        if not 0 < 0.99 * efficiency <= 1:
            raise ValueError(
                'Distillation efficiency factor must give recoveries in (0, 1].'
            )
        D404.distillation_efficiency = efficiency
        apply_distillation_efficiency()

    D404.set_distillation_efficiency = set_distillation_efficiency
    
    # Set sulfuric acid by loading x (g H2SO4 / g microalgae): function(x)
    @R201.add_specification(run=True)
    def set_acid_loading(x=None):
        nonlocal acid_loading
        if x is not None:
            acid_loading = float(x)
        current_microalgae_mass = microalgae_feed.F_mass
        pure_h2so4 = current_microalgae_mass * acid_loading
        acid_solution_mass_new = pure_h2so4 / acid_purity
        water_mass_acid_new = acid_solution_mass_new * (1 - acid_purity)
        _sulfuric_acid_stream.imass['H2SO4'] = pure_h2so4
        _sulfuric_acid_stream.imass['Water'] = water_mass_acid_new
    # Bind as method for parameter loading (function(x))
    R201.set_acid_loading = set_acid_loading
    
    # Enzyme loading functions: function(x) directly sets coefficients (kg/kg)
    @R203.add_specification(run=True)
    def set_glucoamylase_loading(x=None):
        nonlocal gluco_coeff
        if x is not None:
            gluco_coeff = float(x)
        _glucoamylase_stream.imass['GlucoAmylase'] = microalgae_feed.F_mass * gluco_coeff
    @R204.add_specification(run=True)
    def set_alpha_amylase_loading(x=None):
        nonlocal alpha_coeff
        if x is not None:
            alpha_coeff = float(x)
        _alpha_amylase_stream.imass['AlphaAmylase'] = microalgae_feed.F_mass * alpha_coeff
    # Bind as methods for parameter loading (function(x))
    R203.set_glucoamylase_loading = set_glucoamylase_loading
    R204.set_alpha_amylase_loading = set_alpha_amylase_loading

    # =====================
    # Area 5: Waste reuse for biogas production
    # =====================
    # Collect sludge/residual streams from both fermentation stages and
    # downstream solids separation for the waste-treatment section.
    M501 = Mixer('M501', [S202-1, S402-1, *fermentation_wastes, S301-1])
    R501 = AnaerobicDigestion(
        'R501', M501-0,
        design_microalgae_mass=microalgae_mass,
    )
    M503 = Mixer('M503', R501-1)
    wastewater_biogas_placeholder = bst.Stream()
    M502 = Mixer('M502', [R501-0, T302-0, wastewater_biogas_placeholder])

    @T201.add_specification(run=True)
    def synchronize_feedstock_capacity():
        current_microalgae_mass = microalgae_feed.F_mass
        microalgae_water.imass['Water'] = (
            current_microalgae_mass * (1 - 0.04) / 0.04
        )
        set_acid_loading()

        pure_h2so4_current = current_microalgae_mass * acid_loading
        h2so4_mol_current = pure_h2so4_current * 1000 / 98
        ammonium_hydroxide.imass['NH4OH'] = h2so4_mol_current * 0.08 * 35 / 1000

        set_glucoamylase_loading()
        set_alpha_amylase_loading()
        naoh.imass['NaOH'] = current_microalgae_mass * 0.03

        for reactor in fermentation_reactors:
            reactor.microalgae_mass_flow = current_microalgae_mass
        R501.design_microalgae_mass = current_microalgae_mass

    # =====================
    # Area 6: Facilities requirements
    # =====================

    T601 = StorageTank('T601', D401-0, tau=30.*24., V_wf=0.9, vessel_type='Floating roof', vessel_material='Stainless steel')
    P601 = Pump('P601', T601-0, acetic_acid_product)
    T602 = StorageTank('T602', D402-0, tau=30.*24., V_wf=0.9, vessel_type='Floating roof', vessel_material='Stainless steel')
    P602 = Pump('P602', T602-0, propionic_acid_product)
    T603 = StorageTank('T603', D403-0, tau=30.*24., V_wf=0.9, vessel_type='Floating roof', vessel_material='Stainless steel')
    P603 = Pump('P603', T603-0, butyric_acid_product)
    T604 = StorageTank('T604', D404-0, tau=30.*24., V_wf=0.9, vessel_type='Floating roof', vessel_material='Stainless steel')
    P604 = Pump('P604', T604-0, caproic_acid_product)
    CT = CoolingTower('CT')
    HXN601 = HeatExchangerNetwork('HXN601', 
                                  #T_min_app=10, 
                                  #min_heat_util=2e6
                                  ) 
    PWC = ProcessWaterCenter('PWC')
    ADP = AirDistributionPackage('ADP')
    CWP = ChilledWaterPackage('CWP')
    BT601 = bst.facilities.BoilerTurbogenerator('BT601', 
                                                ins=(R501-2, M502-0, '', '', '', ''),
                                                satisfy_system_electricity_demand=True,  
                                                boiler_efficiency=0.7,
                                                turbogenerator_efficiency=0.7,
                                                natural_gas_price=price['NaturalGas'])

    WastewaterT = bst.create_high_rate_wastewater_treatment_system('WastewaterT',
        M503-0,
        skip_IC=True,  # Skip internal circulation to avoid division by zero
        process_ID='6'  # Use process ID 6 for unit numbering
    )
    M502.ins[2] = WastewaterT.flowsheet.stream.biogas

def create_microalgae_MCCA_control_sys(**kwargs):
    """Create the independent single-stage control system."""
    kwargs.setdefault('ID', 'Microalgae_MCCA_control')
    return create_microalgae_MCCA_production_sys(
        fermentation_mode='control',
        **kwargs,
    )


# ==========================================
# TEA Analysis
# ==========================================
# Create system and TEA objects at module level for import
u = flowsheet.unit
s = flowsheet.stream
microalgae_mcca_sys = create_microalgae_MCCA_production_sys()
microalgae_mcca_sys.simulate()

# TEA analysis
# Dry biomass feed rate in ton per day (t/d)
dry_tpd = u.U101.ins[0].F_mass * 24 / 1000  # kg/h -> t/d
microalgae_tea = microalgae_tea(microalgae_mcca_sys)

if __name__ == '__main__':
    microalgae_mcca_sys.diagram('cluster', format='png')
    microalgae_mcca_sys.print()
    # print("\n===== Techno-Economic Analysis (TEA) Main Results =====")
    # # # Use the system's main product stream directly for price calculation
    caproic_acid_product = s.caproic_acid_product
    price = microalgae_tea.solve_price(caproic_acid_product)
    print(f"Caproic Acid Minimum Selling Price: {price:.2f} $/kg")
    if caproic_acid_product.F_mass > 0 and caproic_acid_product.price > 0:
        print("Caproic Acid Unit Production Cost:", microalgae_tea.production_costs([caproic_acid_product]))
    print("NPV:", microalgae_tea.NPV)
    print("TCI:", microalgae_tea.TCI)
    print("FCI:", microalgae_tea.FCI)
    print("DPI:", microalgae_tea.DPI)
    print("TDC:", microalgae_tea.TDC)
    print("FOC:", microalgae_tea.FOC)
    print("VOC:", microalgae_tea.VOC)
    print("AOC:", microalgae_tea.AOC)
    print("ROI:", microalgae_tea.ROI)
    print("PBP:", microalgae_tea.PBP)
    print("Annual Depreciation:", microalgae_tea.annual_depreciation)
    print("Sales:", microalgae_tea.sales)
    print("Material Cost:", microalgae_tea.material_cost)
    print("Utility Cost:", microalgae_tea.utility_cost)
    print("CAPEX Table:\n", microalgae_tea.CAPEX_table())
    print("FOC Table:\n", microalgae_tea.FOC_table())
    cashflow_df = microalgae_tea.get_cashflow_table()
    print("Cashflow Table:\n", cashflow_df.to_string(index=True))
    #Save cashflow as CSV to analyses/results with timestamp
    results_dir = os.path.join(os.path.dirname(__file__), 'analyses', 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    csv_path = os.path.join(results_dir, f"cashflow_{timestamp}.csv")
    cashflow_df.to_csv(csv_path, index=True)
    print(f"Cashflow CSV saved to: {csv_path}")
    
    # Quick check: product flows in each units
    # print("\n===== Stream Mass Flows for Each Unit (kg/hr) =====")
    # for u_ in microalgae_mcca_sys.units:
    #     print(f"\n[{u_.ID} - {u_.__class__.__name__}]")
    #     for i, stream in enumerate(u_.ins):
    #         if stream:
    #             print(f"  Inlet {i+1} ({stream.ID}):")
    #             for chem, flow in zip(stream.chemicals.IDs, stream.mass):
    #                 if abs(flow) > 1e-6:
    #                     print(f"    {chem}: {flow:.2f} kg/hr")
    #     for i, stream in enumerate(u_.outs):
    #         if stream:
    #             print(f"  Outlet {i+1} ({stream.ID}):")
    #             for chem, flow in zip(stream.chemicals.IDs, stream.mass):
    #                 if abs(flow) > 1e-6:
    #                     print(f"    {chem}: {flow:.2f} kg/hr")

    # # Quick check: product flows and prices
    # for p in (#s.butanol_product, 
    #           s.caproic_acid_product, s.butyric_acid_product):
    #    print(f"{p.ID}: {p.F_mass:.2f} kg/h @ {p.price} $/kg")
