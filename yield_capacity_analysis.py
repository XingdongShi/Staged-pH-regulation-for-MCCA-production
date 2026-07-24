#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-stage pH control for chain elongation in microalgae fermentation without external electron donor addition

Generate an MPSP contour comparison across C6 yield and plant capacity for
the two-stage process and the single-stage pH 7 control.  The two scenarios
retain their own yield domains; this is not a matched-yield analysis.

References
----------
[1] BioSTEAM Documentation: 
[2] Cortes-Peña et al., BioSTEAM: A Fast and Flexible Platform for the Design, 
    Simulation, and Techno-Economic Analysis of Biorefineries under Uncertainty. 
    ACS Sustainable Chem. Eng. 2020, 8 (8), 3302–3310.
[3] succinic biorefineries project:
    https://github.com/BioSTEAMDevelopmentGroup/Bioindustrial-Park/tree/master/biorefineries/succinic

@author: Xingdong Shi
@version: 0.0.1
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import biosteam as bst
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from ._chemicals import chems
from .system import (
    create_microalgae_MCCA_control_sys,
    create_microalgae_MCCA_production_sys,
)
from .tea import microalgae_tea
from .utils import price


BASELINE_CAPACITY = 5000.0  # kg dry microalgae/h
TOTAL_HRT = 15.0  # d

SCENARIOS = {
    'two_stage': {
        'label': 'Two-stage pH',
        'creator': create_microalgae_MCCA_production_sys,
        'baseline_yield': 0.05,
        'yield_range': (0.03, 0.10),
        'reactor_ids': ('R301', 'R302'),
    },
    'control': {
        'label': 'Control (pH 7)',
        'creator': create_microalgae_MCCA_control_sys,
        'baseline_yield': 0.005,
        'yield_range': (0.004, 0.010),
        'reactor_ids': ('R302',),
    },
}

METRICS = {
    'MPSP': {
        'label': 'MPSP [2016 USD kg$^{-1}$ recovered product]',
        'title': 'MPSP across C6 yield and plant capacity',
        'cmap': 'viridis_r',
    },
}

AUXILIARY_RESULTS = (
    'C6_production',
    'C6_mass_fraction',
)


def _axis_with_baseline(lower, upper, steps, baseline):
    """Return an ordered grid with the experimental baseline represented exactly."""
    values = np.linspace(float(lower), float(upper), int(steps))
    values[np.argmin(np.abs(values - baseline))] = float(baseline)
    values.sort()
    if np.any(np.diff(values) <= 0):
        raise ValueError('Grid resolution is too low to include the baseline uniquely.')
    return values


def _create_context(scenario):
    scenario_data = SCENARIOS[scenario]
    bst.settings.set_thermo(chems)
    bst.PowerUtility.price = price['Electricity']
    flowsheet = bst.Flowsheet(f'yield_capacity_{scenario}')
    bst.main_flowsheet.set_flowsheet(flowsheet)

    system = scenario_data['creator']()
    system.set_tolerance(
        mol=1e-3, rmol=1e-4, maxiter=1000, subsystems=True,
    )
    system.simulate()

    u = system.flowsheet.unit
    s = system.flowsheet.stream
    u.R302.set_total_HRT(TOTAL_HRT)
    tea = microalgae_tea(system)
    # Run again after TEA adds its continuously scaled labor specification.
    system.simulate()
    return system, u, s, tea


def _validate_grid_point(scenario, capacity, c6_yield, u, s):
    scenario_data = SCENARIOS[scenario]
    if not np.isclose(s.microalgae.F_mass, capacity, rtol=0.0, atol=1e-8):
        raise RuntimeError(f'{scenario}: feed capacity was not applied.')
    for reactor_id in scenario_data['reactor_ids']:
        reactor = getattr(u, reactor_id)
        if not np.isclose(
            reactor.microalgae_mass_flow, capacity, rtol=0.0, atol=1e-8,
        ):
            raise RuntimeError(
                f'{scenario}: {reactor_id}.microalgae_mass_flow did not '
                'synchronize with plant capacity.'
            )
    expected_c6 = capacity * c6_yield
    actual_c6 = u.R302.outs[0].imass['CaproicAcid']
    if not np.isclose(actual_c6, expected_c6, rtol=1e-8, atol=1e-8):
        raise RuntimeError(
            f'{scenario}: R302 C6 target mismatch at capacity={capacity:g} '
            f'kg/h and yield={c6_yield:g}.'
        )


def _evaluate_scenario(scenario, yields, capacities):
    system, u, s, tea = _create_context(scenario)
    product = s.caproic_acid_product
    grids = {
        name: np.empty((len(capacities), len(yields)), dtype=float)
        for name in (*METRICS, *AUXILIARY_RESULTS)
    }

    for row, capacity in enumerate(capacities):
        system.empty_recycles()
        for column, c6_yield in enumerate(yields):
            bst.main_flowsheet.set_flowsheet(system.flowsheet)
            bst.PowerUtility.price = price['Electricity']
            s.microalgae.F_mass = float(capacity)
            u.R302.set_C6_yield(float(c6_yield))
            system.simulate()
            _validate_grid_point(scenario, capacity, c6_yield, u, s)

            # Match the uncertainty analysis by reporting the selling price of
            # the recovered C6-rich product stream. The C6 mass fraction is
            # exported separately and is not used to renormalize MPSP.
            mixed_product_mpsp = float(tea.solve_price(product))
            c6_mass = float(product.imass['CaproicAcid'])
            product_mass = float(product.F_mass)
            if c6_mass <= 0 or product_mass <= 0:
                raise RuntimeError(
                    f'{scenario}: nonpositive C6 product at capacity={capacity:g} '
                    f'kg/h and yield={c6_yield:g}.'
                )
            c6_mass_fraction = c6_mass / product_mass
            values = {
                'MPSP': mixed_product_mpsp,
                'C6_production': c6_mass,
                'C6_mass_fraction': c6_mass_fraction,
            }
            for name, value in values.items():
                if not np.isfinite(value):
                    raise RuntimeError(
                        f'{scenario}: nonfinite {name} at capacity={capacity:g} '
                        f'kg/h and yield={c6_yield:g}.'
                    )
                grids[name][row, column] = value

        print(
            f'{scenario}: completed capacity row {row + 1}/{len(capacities)} '
            f'({capacity:.1f} kg/h)'
        )

    # Restore the experimental baseline state before releasing the context.
    s.microalgae.F_mass = BASELINE_CAPACITY
    u.R302.set_C6_yield(SCENARIOS[scenario]['baseline_yield'])
    system.simulate()
    return grids


def _local_linear_levels(values, number=13):
    """Return evenly spaced, scenario-specific contour levels."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise RuntimeError('No finite contour results were calculated.')
    lower = float(finite.min())
    upper = float(finite.max())
    if lower == upper:
        upper = lower + 1e-6
    return np.linspace(lower, upper, number)


def _set_plot_style():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.edgecolor': '#111111',
        'axes.linewidth': 1.0,
    })


def _plot_metric(metric, results, yields_by_scenario, capacities, figures_dir, tag):
    _set_plot_style()
    scenario_names = tuple(SCENARIOS)
    arrays = [results[scenario][metric] for scenario in scenario_names]
    figure, axes = plt.subplots(
        1, len(scenario_names), figsize=(15.0, 7.0), sharey=True,
    )
    axes = np.atleast_1d(axes)

    for axis, scenario, values in zip(axes, scenario_names, arrays):
        scenario_data = SCENARIOS[scenario]
        yields = yields_by_scenario[scenario]
        X, Y = np.meshgrid(yields, capacities)
        levels = _local_linear_levels(values)
        axis.contourf(
            X, Y, values, levels=levels,
            cmap=METRICS[metric]['cmap'], extend='both',
        )
        line_color = '#252525'
        label_levels = levels[1:-1:2]
        contour_lines = axis.contour(
            X, Y, values, levels=label_levels,
            colors=line_color, linewidths=0.55, alpha=0.88,
        )
        axis.clabel(
            contour_lines, levels=label_levels,
            inline=True, fontsize=9, fmt='%.2f',
        )
        annotation_color = '#F7FAFC' if scenario == 'control' else line_color
        baseline_yield = scenario_data['baseline_yield']
        baseline_row = int(np.flatnonzero(np.isclose(
            capacities, BASELINE_CAPACITY,
        ))[0])
        baseline_column = int(np.flatnonzero(np.isclose(
            yields, baseline_yield,
        ))[0])
        baseline_mpsp = float(values[baseline_row, baseline_column])
        axis.scatter(
            baseline_yield, BASELINE_CAPACITY,
            marker='*', s=190, facecolor='white', edgecolor='#111111',
            linewidth=1.0, zorder=5,
        )
        axis.annotate(
            f'{baseline_mpsp:.2f}',
            (baseline_yield, BASELINE_CAPACITY),
            xytext=(-8, 18), textcoords='offset points', fontsize=11,
            color=annotation_color, ha='right',
        )
        axis.set_title(
            scenario_data['label'], loc='center', pad=12,
            fontsize=15, fontweight='normal',
        )
        axis.set_xlabel('C6 yield [kg C6 kg$^{-1}$ dry microalgae]')
        axis.grid(False)

    axes[0].set_ylabel('Plant capacity [kg dry microalgae h$^{-1}$]')
    figure.suptitle(
        METRICS[metric]['title'], x=0.5, y=0.975,
        ha='center', fontsize=18, fontweight='normal',
    )
    figure.subplots_adjust(
        left=0.075, right=0.97, top=0.84, bottom=0.15, wspace=0.14,
    )
    paths = []
    for extension in ('png', 'svg', 'pdf'):
        path = figures_dir / f'{metric}_yield_capacity_contour-{tag}.{extension}'
        save_kwargs = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if extension == 'png':
            save_kwargs['dpi'] = 600
        figure.savefig(path, **save_kwargs)
        paths.append(path)
    plt.close(figure)
    return paths


def _write_results(results, yields_by_scenario, capacities, results_dir, tag):
    archive = {'plant_capacity_kgph': capacities}
    summary_rows = []
    output_paths = []

    for scenario, scenario_results in results.items():
        yields = yields_by_scenario[scenario]
        archive[f'{scenario}_C6_yield'] = yields
        baseline_row = int(np.flatnonzero(np.isclose(capacities, BASELINE_CAPACITY))[0])
        baseline_column = int(np.flatnonzero(np.isclose(
            yields, SCENARIOS[scenario]['baseline_yield'],
        ))[0])
        for metric, values in scenario_results.items():
            archive[f'{scenario}_{metric}'] = values
            dataframe = pd.DataFrame(
                values,
                index=np.round(capacities, 6),
                columns=np.round(yields, 8),
            )
            dataframe.index.name = 'Plant capacity [kg dry microalgae/h]'
            dataframe.columns.name = 'Specified C6 yield [kg C6/kg dry microalgae]'
            csv_path = results_dir / f'{scenario}_{metric}_yield_capacity-{tag}.csv'
            dataframe.to_csv(csv_path)
            output_paths.append(csv_path)
            summary_rows.append({
                'Scenario': scenario,
                'Metric': metric,
                'Minimum': float(np.min(values)),
                'Experimental_baseline': float(values[baseline_row, baseline_column]),
                'Maximum': float(np.max(values)),
            })

    npz_path = results_dir / f'{tag}.npz'
    np.savez_compressed(npz_path, **archive)
    output_paths.append(npz_path)
    summary_path = results_dir / f'yield_capacity_summary-{tag}.csv'
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    output_paths.append(summary_path)
    return output_paths


def run_yield_capacity_analysis(
        steps: int = 60,
        capacity_range=(4000.0, 6000.0),
        plot: bool = True,
    ):
    """Run scenario-specific C6-yield-by-capacity analyses and export results."""
    if steps < 5:
        raise ValueError('steps must be at least 5 for stable contour maps.')

    capacities = _axis_with_baseline(
        *capacity_range, steps, BASELINE_CAPACITY,
    )
    yields_by_scenario = {
        scenario: _axis_with_baseline(
            *scenario_data['yield_range'], steps,
            scenario_data['baseline_yield'],
        )
        for scenario, scenario_data in SCENARIOS.items()
    }
    results = {
        scenario: _evaluate_scenario(
            scenario, yields_by_scenario[scenario], capacities,
        )
        for scenario in SCENARIOS
    }

    base = Path(__file__).resolve().parent
    results_dir = base / 'analyses' / 'results'
    figures_dir = base / 'analyses' / 'figures'
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    tag = (
        f'microalgae_baseline_control_yield_capacity_{steps}x{steps}_'
        f'{now.year}.{now.month}.{now.day}-{now.hour}.{now.minute:02d}.{now.second:02d}'
    )

    result_paths = _write_results(
        results, yields_by_scenario, capacities, results_dir, tag,
    )
    figure_paths = {}
    if plot:
        figure_paths = {
            metric: _plot_metric(
                metric, results, yields_by_scenario, capacities,
                figures_dir, tag,
            )
            for metric in METRICS
        }

    print(f'Yield-capacity data saved to: {results_dir}')
    if plot:
        print(f'Yield-capacity figures saved to: {figures_dir}')
    print(f'Tag: {tag}')
    return {
        'capacities': capacities,
        'yields': yields_by_scenario,
        'results': results,
        'result_paths': result_paths,
        'figure_paths': figure_paths,
        'tag': tag,
    }


# Uncertainty-result plotting is kept here so all project plotting utilities
# share one module with the yield-capacity analysis.
_PLOT_LABELS = {
    'baseline': 'Two-stage pH',
    'control': 'Control (pH 7)',
    'ethanol': 'External ethanol',
}
_PLOT_COLORS = {
    'baseline': '#2F6B9A',
    'control': '#B7791F',
    'ethanol': '#5B8C5A',
}
_PLOT_METRICS = {
    'MPSP': ('TEA results', 0, 'MPSP [$/kg]'),
    'GWP': ('LCA results', 0, 'GWP [kg CO2-eq/kg]'),
    'FEC': ('LCA results', 1, 'FEC [MJ/kg]'),
}


def _save_plot(fig, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f'{stem}.{suffix}' for suffix in ('png', 'svg')]
    fig.savefig(paths[0], dpi=600, bbox_inches='tight')
    fig.savefig(paths[1], bbox_inches='tight')
    plt.close(fig)
    return paths


def _read_uncertainty_workbook(path):
    path = Path(path)
    baseline_path = Path(str(path).replace('_1_full_evaluation.xlsx', '_0_baseline.xlsx'))
    if not baseline_path.is_file():
        raise FileNotFoundError(f'Baseline workbook not found: {baseline_path}')
    return (
        pd.read_excel(path, sheet_name='TEA results', header=[0, 1], index_col=0),
        pd.read_excel(path, sheet_name='LCA results', header=[0, 1], index_col=0),
        pd.read_excel(path, sheet_name='Spearman'),
        pd.read_excel(baseline_path, header=[0, 1], index_col=0),
    )


def generate_uncertainty_figures(result_files, output_dir):
    """Create uncertainty and sensitivity figures from completed workbooks."""
    data = {mode: _read_uncertainty_workbook(path)
            for mode, path in result_files.items()}
    modes = [mode for mode in ('baseline', 'control', 'ethanol') if mode in data]
    if not modes:
        raise ValueError('No uncertainty-result workbooks were provided.')

    fig, axes = plt.subplots(len(modes), 3, figsize=(12, 3.4 * len(modes)), squeeze=False)
    summary = []
    for row, mode in enumerate(modes):
        tea, lca, _, baseline = data[mode]
        for col, (metric, (sheet, position, label)) in enumerate(_PLOT_METRICS.items()):
            frame = tea if sheet == 'TEA results' else lca
            values = pd.to_numeric(frame.iloc[:, position], errors='coerce').dropna()
            if values.empty:
                raise ValueError(f'{mode} {metric} has no valid results.')
            baseline_value = float(baseline.iloc[0][frame.columns[position]])
            ax = axes[row, col]
            box = ax.boxplot([values], vert=False, widths=.48, showfliers=False,
                             patch_artist=True, medianprops={'color': '#1A202C'})
            box['boxes'][0].set(facecolor=_PLOT_COLORS[mode], alpha=.30,
                                edgecolor=_PLOT_COLORS[mode])
            ax.scatter(baseline_value, 1, marker='D', s=34, facecolor='white', edgecolor='#1A202C')
            q05, q50, q95 = values.quantile([.05, .5, .95])
            ax.set(title=f'{_PLOT_LABELS[mode]} — {metric}', xlabel=label, yticks=[])
            ax.grid(axis='x', color='#E2E8F0', linewidth=.7)
            ax.text(.99, .91, f'P5–P95: {q05:.2f}–{q95:.2f}\nMedian: {q50:.2f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8)
            summary.append({'Scenario': mode, 'Metric': metric, 'N': len(values),
                            'P05': q05, 'Median': q50, 'P95': q95,
                            'Deterministic_baseline': baseline_value})
    fig.suptitle('Monte Carlo uncertainty distributions', x=.07, ha='left', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    outputs = _save_plot(fig, output_dir, 'uncertainty_distributions_2000sims_baseline_control_ethanol')

    fig, axes = plt.subplots(len(modes), 3, figsize=(13.5, 4.8 * len(modes)), squeeze=False)
    for row, mode in enumerate(modes):
        _, _, spearman, _ = data[mode]
        for col, (metric, (_, _, label)) in enumerate(_PLOT_METRICS.items()):
            column = label.replace(' [$/kg]', ' (USD kg$^{-1}$)').replace(' [', ' (').replace(']', ')')
            if column not in spearman.columns:
                column = next((x for x in spearman.columns if metric in str(x)), None)
            subset = spearman[['Parameter', column]].dropna()
            subset = subset[~subset.Parameter.astype(str).str.contains('Blank', case=False)]
            subset = subset.assign(abs_rho=subset[column].abs()).nlargest(8, 'abs_rho').sort_values('abs_rho')
            values = subset[column].to_numpy(float)
            labels = [re.sub(r'\s*\[[^\]]*\]$', '', str(x)) for x in subset.Parameter]
            ax = axes[row, col]
            ax.barh(range(len(values)), values,
                    color=np.where(values >= 0, _PLOT_COLORS['control'], _PLOT_COLORS['baseline']))
            ax.set(yticks=range(len(values)), yticklabels=labels, xlim=(-1, 1),
                   xlabel=r'Spearman $\rho$', title=f'{_PLOT_LABELS[mode]} — {metric}')
            ax.axvline(0, color='#4A5568', linewidth=.8)
    fig.legend(handles=[Patch(facecolor=_PLOT_COLORS['control'], label='Positive'),
                        Patch(facecolor=_PLOT_COLORS['baseline'], label='Negative')],
               loc='upper right', frameon=False)
    fig.suptitle('Global sensitivity of TEA and LCA outcomes', x=.06, ha='left', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    outputs.extend(_save_plot(fig, output_dir, 'spearman_sensitivity_2000sims_baseline_control_ethanol'))
    summary_path = Path(next(iter(result_files.values()))).parent / 'monte_carlo_summary_2000sims_baseline_control_ethanol.csv'
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    return [*outputs, summary_path]


def generate_tea_breakdown_figures(systems, teas, unit_groups, output_dir):
    """Create baseline TEA and utility contribution figures by scenario."""
    from .tea import create_tea_breakdown_dataframe
    outputs = []
    palette = ['#2F6B9A', '#B7791F', '#667C3E', '#C56A3A', '#8A5A83', '#4D7C78']
    for mode, system in systems.items():
        frame = create_tea_breakdown_dataframe(system, teas[mode], unit_groups[mode],
                                                fraction=True, scale_fractions_to_positive_values=True)
        frame = frame.loc[:, frame.abs().sum(axis=0) > 1e-12]
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        positive = np.zeros(len(frame.columns)); negative = np.zeros(len(frame.columns))
        for i, (name, row) in enumerate(frame.iterrows()):
            values = row.to_numpy(float); bottom = np.where(values >= 0, positive, negative)
            ax.bar(range(len(values)), values, bottom=bottom, label=name,
                   color=palette[i % len(palette)], edgecolor='white', linewidth=.35)
            positive += np.maximum(values, 0); negative += np.minimum(values, 0)
        ax.set(xticks=range(len(frame.columns)),
               xticklabels=[str(x).replace(' [', '\n[') for x in frame.columns], ylabel='Contribution (%)')
        ax.tick_params(axis='x', rotation=20)
        ax.axhline(0, color='#4A5568', linewidth=.8)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
        fig.suptitle(f'TEA and utility breakdown — {_PLOT_LABELS.get(mode, mode)}', x=.08, ha='left')
        fig.tight_layout()
        outputs.extend(_save_plot(fig, output_dir, f'tea_breakdown_{mode}'))
    return outputs


if __name__ == '__main__':
    run_yield_capacity_analysis()
