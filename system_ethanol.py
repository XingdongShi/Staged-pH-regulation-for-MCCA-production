"""Standalone external-ethanol microalgae chain-elongation system."""

import warnings

import biosteam as bst
import numpy as np
import thermosteam as tmo
from biosteam import Stream, SystemFactory, main_flowsheet
from biosteam.facilities import (AirDistributionPackage, ChilledWaterPackage,
                                 CoolingTower, HeatExchangerNetwork,
                                 ProcessWaterCenter)
from biosteam.units import (HXutility, Mixer, MultiStageMixerSettlers, Pump,
                            Splitter, StorageTank)

from ._chemicals import chems
from .streams import microalgae_feed
from .tea import microalgae_tea
from .units import (AcidPretreatmentReactor, AnaerobicDigestion,
                    FeedstockPreprocessing, MCCAFermentation_ethanol,
                    NeutralizationTank, Saccharification,
                    SolidLiquidSeparation)
from .utils import price

__all__ = ('create_microalgae_MCCA_ethanol_sys',
           'microalgae_mcca_ethanol_sys', 'microalgae_ethanol_tea')

warnings.filterwarnings('ignore', category=bst.exceptions.CostWarning)
bst.settings.set_thermo(chems)
ethanol_flowsheet = bst.Flowsheet('MCCA_ethanol')
main_flowsheet.set_flowsheet(ethanol_flowsheet)


@SystemFactory(
    ID='Microalgae_MCCA_ethanol',
    ins=[dict(microalgae_feed, thermo=chems)],
    outs=[dict(ID='butyric_acid_product', thermo=chems),
          dict(ID='caproic_acid_product', thermo=chems),
          dict(ID='acetic_acid_product', thermo=chems),
          dict(ID='propionic_acid_product', thermo=chems)],
)
def create_microalgae_MCCA_ethanol_sys(ins, outs):
    """Build the independent external-ethanol reference scenario.

    Ethanol is supplied at 1.38 kg per kg dry microalgae. Fermentation product
    yields are defined by ``MCCAFermentation_ethanol`` on the same basis.
    """
    tmo.settings.set_thermo(chems)
    microalgae, = ins
    butyric_product, caproic_product, acetic_product, propionic_product = outs
    microalgae_mass = microalgae.F_mass
    water = Stream('microalgae_water', Water=microalgae_mass * .96 / .04,
                   units='kg/hr')
    acid_loading, acid_purity = 1.47, .93
    acid = Stream('sulfuricacid', H2SO4=microalgae_mass * acid_loading,
                  Water=microalgae_mass * acid_loading / acid_purity * (1-acid_purity),
                  units='kg/hr', price=price['SulfuricAcid'])
    ammonium = Stream('ammonium_hydroxide',
        NH4OH=microalgae_mass * acid_loading * 1000 / 98 * .08 * 35 / 1000,
        units='kg/hr', price=price['AmmoniumHydroxide'])
    gluco_coeff, alpha_coeff = .0011, .0082
    gluco = Stream('glucoamylase', GlucoAmylase=microalgae_mass*gluco_coeff,
                   units='kg/hr', price=price['GlucoAmylase'])
    alpha = Stream('alpha_amylase', AlphaAmylase=microalgae_mass*alpha_coeff,
                   units='kg/hr', price=price['AlphaAmylase'])
    naoh = Stream('naoh', NaOH=microalgae_mass*.03, units='kg/hr', price=price['NaOH'])
    # Bioindustrial-Park microalgae ethanol-fed reference dosage.
    ethanol_loading = 1.38
    ethanol = Stream('ethanol', Ethanol=microalgae_mass*ethanol_loading,
                     units='kg/hr', price=price['Ethanol'])
    fresh_oleyl = Stream('fresh_oleylalcohol', OleylAlcohol=200, units='kg/hr',
                         price=price['OleylAlcohol'])
    recycle_oleyl = Stream('oleylalcohol_recycle', OleylAlcohol=50, units='kg/hr')
    oleyl_feed = Stream('oleylalcohol_feed')
    for stream, key in ((butyric_product, 'ButyricAcid'),
                        (caproic_product, 'CaproicAcid'),
                        (acetic_product, 'AceticAcid'),
                        (propionic_product, 'PropionicAcid')):
        stream.price = price[key]

    U101 = FeedstockPreprocessing('U101', microalgae, thermo=chems)
    T201 = StorageTank('T201', acid); P201 = Pump('P201', T201-0, P=5e5)
    M201 = Mixer('M201', [P201-0, water, U101-0]); P202 = Pump('P202', M201-0)
    H201 = HXutility('H201', P202-0, T=394.15); R201 = AcidPretreatmentReactor('R201', H201-0)
    T202 = StorageTank('T202', ammonium); P203 = Pump('P203', T202-0)
    R202 = NeutralizationTank('R202', [R201-0, P203-0]); P204 = Pump('P204', R202-0)
    H202 = HXutility('H202', P204-0, T=328.15)
    T203 = StorageTank('T203', gluco); P205 = Pump('P205', T203-0, P=25e5, dP_design=24e5, ignore_NPSH=True)
    T204 = StorageTank('T204', alpha); P206 = Pump('P206', T204-0, P=25e5, dP_design=24e5, ignore_NPSH=True)
    T205 = StorageTank('T205', naoh); P207 = Pump('P207', T205-0, P=25e5, dP_design=24e5, ignore_NPSH=True)
    S201 = Splitter('S201', P207-0, split=.5)
    M202 = Mixer('M202', [H202-0, P205-0, S201-0]); R203 = Saccharification('R203', M202-0)
    H203 = HXutility('H203', R203-0, T=363.15)
    M203 = Mixer('M203', [H203-0, P206-0, S201-1]); R204 = Saccharification('R204', M203-0)
    S202 = SolidLiquidSeparation('S202', R204-0); P208 = Pump('P208', S202-0)

    H301 = HXutility('H301', P208-0, T=310.15)
    T301 = StorageTank('T301', ethanol); P301 = Pump('P301', T301-0)
    R301 = MCCAFermentation_ethanol('R301', [H301-0, P301-0],
                                    microalgae_mass_flow=microalgae_mass,
                                    tau=15*24, pH=7.0)
    R301.set_C6_yield_factor = lambda x: setattr(R301, 'caproic_acid_yield_factor', x)
    T302 = Mixer('T302', R301-1); S301 = SolidLiquidSeparation('S301', R301-0)

    M401 = Mixer('M401', [fresh_oleyl, recycle_oleyl], oleyl_feed)
    @M401.add_specification(run=True)
    def set_fresh_oleyl():
        fresh_oleyl.imass['OleylAlcohol'] = max(200-recycle_oleyl.imass['OleylAlcohol'], 0)
    IDs = ['Water', 'AceticAcid', 'PropionicAcid', 'ButyricAcid', 'ValericAcid', 'CaproicAcid', 'OleylAlcohol']
    S402 = MultiStageMixerSettlers('S402', ins=[S301-0, M401-0], N_stages=5,
        partition_data={'IDs': IDs, 'K': np.array([1/5000, 2000, 3000, 5000, 5000, 5000, 100000])})
    original_K = np.array(S402.partition_data['K'], dtype=float)

    def set_extraction_efficiency(efficiency):
        efficiency = float(efficiency)
        S402.partition_data = {'IDs': IDs, 'K': original_K * efficiency}

    S402.set_extraction_efficiency = set_extraction_efficiency
    D401 = bst.BinaryDistillation('D401', S402-0, LHK=('AceticAcid','PropionicAcid'), Lr=.99, Hr=.99, k=1.2, partial_condenser=False, is_divided=True)
    D402 = bst.BinaryDistillation('D402', D401-1, LHK=('PropionicAcid','ButyricAcid'), Lr=.99, Hr=.99, k=1.2, partial_condenser=False, is_divided=True)
    D403 = bst.BinaryDistillation('D403', D402-1, LHK=('ButyricAcid','CaproicAcid'), Lr=.99, Hr=.99, k=1.2, partial_condenser=False, is_divided=True)
    D404 = bst.BinaryDistillation('D404', D403-1, outs=('', recycle_oleyl), LHK=('CaproicAcid','OleylAlcohol'), Lr=.99, Hr=.99, k=1.2, partial_condenser=False, is_divided=True)
    for unit in (D401, D402, D403, D404): unit.check_LHK = False

    M501 = Mixer('M501', [S202-1, S402-1, R301-2, S301-1])
    R501 = AnaerobicDigestion('R501', M501-0, design_microalgae_mass=microalgae_mass)
    M503 = Mixer('M503', R501-1); gas_placeholder = Stream()
    M502 = Mixer('M502', [R501-0, T302-0, gas_placeholder])
    T601 = StorageTank('T601', D401-0, tau=30*24); P601 = Pump('P601', T601-0, acetic_product)
    T602 = StorageTank('T602', D402-0, tau=30*24); P602 = Pump('P602', T602-0, propionic_product)
    T603 = StorageTank('T603', D403-0, tau=30*24); P603 = Pump('P603', T603-0, butyric_product)
    T604 = StorageTank('T604', D404-0, tau=30*24); P604 = Pump('P604', T604-0, caproic_product)
    CT = CoolingTower('CT'); HXN601 = HeatExchangerNetwork('HXN601'); PWC = ProcessWaterCenter('PWC')
    ADP = AirDistributionPackage('ADP'); CWP = ChilledWaterPackage('CWP')
    BT601 = bst.facilities.BoilerTurbogenerator('BT601', ins=(R501-2, M502-0, '', '', '', ''),
        satisfy_system_electricity_demand=True, boiler_efficiency=.7,
        turbogenerator_efficiency=.7, natural_gas_price=price['NaturalGas'])
    WWT = bst.create_high_rate_wastewater_treatment_system('WastewaterT', M503-0, skip_IC=True, process_ID='6')
    M502.ins[2] = WWT.flowsheet.stream.biogas

    @R201.add_specification(run=True)
    def set_acid_loading(x=None):
        nonlocal acid_loading
        if x is not None:
            acid_loading = float(x)
        mass = microalgae.F_mass
        acid.imass['H2SO4'] = mass * acid_loading
        acid.imass['Water'] = mass * acid_loading / acid_purity * (1 - acid_purity)

    R201.set_acid_loading = set_acid_loading

    @R203.add_specification(run=True)
    def set_glucoamylase_loading(x=None):
        nonlocal gluco_coeff
        if x is not None:
            gluco_coeff = float(x)
        gluco.imass['GlucoAmylase'] = microalgae.F_mass * gluco_coeff

    @R204.add_specification(run=True)
    def set_alpha_amylase_loading(x=None):
        nonlocal alpha_coeff
        if x is not None:
            alpha_coeff = float(x)
        alpha.imass['AlphaAmylase'] = microalgae.F_mass * alpha_coeff

    R203.set_glucoamylase_loading = set_glucoamylase_loading
    R204.set_alpha_amylase_loading = set_alpha_amylase_loading

    @T201.add_specification(run=True)
    def synchronize_capacity():
        mass = microalgae.F_mass
        water.imass['Water'] = mass*.96/.04
        set_acid_loading()
        ammonium.imass['NH4OH'] = mass*acid_loading*1000/98*.08*35/1000
        set_glucoamylase_loading(); set_alpha_amylase_loading()
        naoh.imass['NaOH'] = mass*.03; ethanol.imass['Ethanol'] = mass*ethanol_loading
        R301.microalgae_mass_flow = mass; R501.design_microalgae_mass = mass


microalgae_mcca_ethanol_sys = create_microalgae_MCCA_ethanol_sys()
microalgae_mcca_ethanol_sys.simulate()
microalgae_ethanol_tea = microalgae_tea(microalgae_mcca_ethanol_sys)
