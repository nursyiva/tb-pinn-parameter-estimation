"""Exploratory LHS-PRCC sensitivity for revised eight-compartment DS/MDR TB model.
Python 3.10+: pip install numpy scipy pandas matplotlib
Run: python LHS_PRCC_TB_8kompartemen.py
Source: manuscript Tables 2-4; eta=0.51 is the current Table 3 value.
Sensitivity is conditional, not uncertainty quantification or identifiability.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import qmc, rankdata

SEED = 2026
N_SAMPLES = 300
T_END = 156.0
SHOW_PLOTS = False
OUT = Path(__file__).resolve().parent / 'hasil_LHS_PRCC'
OUT.mkdir(exist_ok=True)

# Table 3 fixed and Table 4 PINN-estimated reference parameters (per week)
BASE = dict(beta_s=.07647, beta_r=.07171, sigma_s=.009319,
            sigma_r=.008460, tau_s=.05909, tau_r=.01414,
            rho=.01523, kappa=.9863, mu=.000269,
            gamma_s=1/26, gamma_r=.0075, delta_s=.000439,
            delta_r=.00365, omega=.00278, eta=.51)
# If the manuscript's eta=0.51 is a typo and should be 0.05,
# change BASE['eta'] above, and adjust eta's sampling range below.
N0 = 288315899.0
# Consistent recruitment: Lambda=mu*N0 for each LHS sample
Y0 = np.array([288016688.,100000.,75690.,70796.,50000.,1000.,941.,784.])
assert np.isclose(Y0.sum(), N0)

# Exploratory ranges (not confidence intervals). Ten jointly varied parameters:
# four estimated (sigma_s,sigma_r,rho,kappa) and six fixed
# (gamma_s,gamma_r,delta_s,delta_r,omega,eta).
# Other parameters beta_s,beta_r,tau_s,tau_r,mu remain at Table 3-4 values.
RANGES = {
    'sigma_s': (.005, .015),
    'sigma_r': (.005, .015),
    'gamma_s': (1/35, 1/17),
    'gamma_r': (.00375, .015),
    'delta_s': (.0002195, .000878),
    'delta_r': (.001825, .00730),
    'omega': (.00139, .00556),
    'eta': (.255, .765),  # 50%-150% of manuscript eta=0.51
    'rho': (0.0, .05),
    'kappa': (.95, .999),
}
PARAMS = list(RANGES)


def rhs(t, y, p):
    S, Es, Is, Ts, R, Er, Ir, Tr, Cs, Cr = y
    N = max(S + Es + Is + Ts + R + Er + Ir + Tr, 1.0)
    ls = p['beta_s'] * Is / N
    lr = p['beta_r'] * Ir / N
    gs, gr = p['gamma_s'], p['gamma_r']
    rho, kap = p['rho'], p['kappa']
    return [
        p['Lambda'] - (ls+lr)*S - p['mu']*S + p['omega']*R,
        ls*(S+p['eta']*R) - (p['sigma_s']+p['mu'])*Es,
        p['sigma_s']*Es + gs*(1-rho)*(1-kap)*Ts - (p['tau_s']+p['mu']+p['delta_s'])*Is,
        p['tau_s']*Is - (gs+p['mu'])*Ts,
        gs*kap*Ts + gr*Tr - p['eta']*(ls+lr)*R - (p['omega']+p['mu'])*R,
        lr*(S+p['eta']*R) - (p['sigma_r']+p['mu'])*Er,
        p['sigma_r']*Er + gs*rho*(1-kap)*Ts - (p['tau_r']+p['mu']+p['delta_r'])*Ir,
        p['tau_r']*Ir - (gr+p['mu'])*Tr,
        p['tau_s']*Is,  # cumulative DS treatment initiations
        p['tau_r']*Ir,  # cumulative MDR treatment initiations
    ]


def integrate(p):
    p = dict(p)
    p['Lambda'] = p['mu'] * N0
    initial = np.r_[Y0, 0., 0.]
    sol = solve_ivp(rhs, (0, T_END), initial, args=(p,), method='LSODA',
                    rtol=1e-7, atol=1e-5)
    if not sol.success or not np.all(np.isfinite(sol.y)):
        raise RuntimeError(sol.message)
    if sol.y[:8].min() < -1e-3:
        raise RuntimeError('Negative compartment detected')
    return sol.y[-2, -1], sol.y[-1, -1], sol.y[2, -1], sol.y[6, -1]


def prcc(X, y):
    """PRCC via rank residualization on remaining inputs; no p-values assumed."""
    ranks = np.column_stack([rankdata(X[:, k]) for k in range(X.shape[1])])
    yrank = rankdata(y)
    result = []
    for k in range(X.shape[1]):
        others = np.delete(ranks, k, axis=1)
        A = np.column_stack((np.ones(len(y)), others))
        rx = ranks[:, k] - A @ np.linalg.lstsq(A, ranks[:, k], rcond=None)[0]
        ry = yrank - A @ np.linalg.lstsq(A, yrank, rcond=None)[0]
        result.append(float(np.corrcoef(rx, ry)[0, 1]))
    return np.array(result)


# Mathematical symbols for publication-quality plots
SYMBOLS = {
    'sigma_s': r'$\sigma_s$', 'sigma_r': r'$\sigma_r$',
    'gamma_s': r'$\gamma_s$', 'gamma_r': r'$\gamma_r$',
    'delta_s': r'$\delta_s$', 'delta_r': r'$\delta_r$',
    'omega': r'$\omega$', 'eta': r'$\eta$',
    'rho': r'$\rho$', 'kappa': r'$\kappa$',
}


def plot_prcc(table, column, filename, title):
    """Horizontal signed PRCC bars with mathematical labels and |PRCC| values."""
    vals = table[column].to_numpy(dtype=float)
    order = np.argsort(np.abs(vals))[::-1]  # largest absolute values at top
    ordered_vals = vals[order]
    ordered_names = [PARAMS[i] for i in order]
    positions = np.arange(len(ordered_vals))

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    colors = ['red' if v < 0 else 'blue' for v in ordered_vals]
    ax.barh(positions, ordered_vals, color=colors, height=0.78)
    ax.set_yticks(positions)
    ax.set_yticklabels([SYMBOLS.get(k, k) for k in ordered_names], fontsize=15)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.9)
    ax.set_xlim(-1.08, 1.08)
    ax.set_xlabel('Partial rank correlation coefficient (PRCC)', fontsize=12)
    ax.set_title(title, fontsize=12, pad=12)
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=11)

    # Second y-axis is aligned to the bars; show absolute values on the right.
    right_ax = ax.twinx()
    right_ax.set_ylim(ax.get_ylim())
    right_ax.set_yticks(positions)
    right_ax.set_yticklabels([f'{abs(v):.4f}' for v in ordered_vals], fontsize=11)
    right_ax.set_ylabel('PRCC Absolute Value', fontsize=12, labelpad=10)
    right_ax.tick_params(axis='y', length=0, pad=8)

    fig.subplots_adjust(left=0.15, right=0.83, top=0.89, bottom=0.13)
    fig.savefig(OUT / filename, dpi=300, bbox_inches='tight')
    fig.savefig(OUT / filename.replace('.png', '.pdf'), bbox_inches='tight')
    return fig



def plot_prcc_combined(table):
    """One publication figure, two panels, shared parameter order and scale."""
    cols = ['DS_cumulative_enrolment', 'MDR_cumulative_enrolment']
    # Order by maximum absolute PRCC across both outcomes, consistently.
    score = np.maximum(np.abs(table[cols[0]].to_numpy()),
                       np.abs(table[cols[1]].to_numpy()))
    order = np.argsort(score)[::-1]
    labels = [SYMBOLS.get(PARAMS[i], PARAMS[i]) for i in order]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6), sharey=True)
    for ax, col, heading in zip(axes, cols, ['(a) DS-TB', '(b) MDR-TB']):
        vals = table[col].to_numpy(dtype=float)[order]
        ax.barh(y, vals, color=['#e03131' if v < 0 else '#2455d6' for v in vals], height=.72)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=13)
        ax.tick_params(axis='y', labelleft=True, length=0, pad=9)
        ax.invert_yaxis()
        ax.axvline(0, color='#222222', lw=1)
        ax.set_xlim(-1.07, 1.07)
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.set_title(heading, fontsize=14, weight='bold', pad=10)
        ax.grid(axis='x', linestyle='--', alpha=.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=10)
        # right aligned numeric column, preserving a common plotting scale
        right = ax.twinx()
        right.set_ylim(ax.get_ylim())
        right.set_yticks(y)
        right.set_yticklabels([f'{abs(v):.4f}' for v in vals], fontsize=10)
        right.tick_params(axis='y', length=0, pad=6)
        right.set_ylabel('|PRCC|', fontsize=11)
    fig.supxlabel('Partial rank correlation coefficient (PRCC)', fontsize=12, y=.035)
    fig.subplots_adjust(left=.095, right=.92, wspace=.52, top=.88, bottom=.13)
    for ext in ('png','pdf'):
        fig.savefig(OUT / f'PRCC_DS_MDR_combined.{ext}', dpi=300, bbox_inches='tight')
    return fig

def main():
    print('Reference eta =', BASE['eta'], '(Table 3; verify against original simulation)')
    print('LHS:', N_SAMPLES, 'draws, 10 varied parameters; 156 weeks')
    names = PARAMS
    bounds = np.array([RANGES[k] for k in names], dtype=float)
    sample = qmc.LatinHypercube(d=len(names), seed=SEED).random(n=N_SAMPLES)
    X = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    output_names = ['DS_cumulative_enrolment', 'MDR_cumulative_enrolment',
                    'Is_week156', 'Ir_week156']
    Y = np.empty((N_SAMPLES, len(output_names)))
    for i, x in enumerate(X):
        p = dict(BASE)
        p.update(dict(zip(names, x)))
        Y[i] = integrate(p)
        if (i+1) % 50 == 0:
            print('Simulations completed:', i+1, '/', N_SAMPLES)
    sample_df = pd.DataFrame(X, columns=names)
    for j, key in enumerate(output_names):
        sample_df[key] = Y[:, j]
    sample_df.to_csv(OUT / 'LHS_samples_and_outputs.csv', index=False)

    prcc_table = pd.DataFrame({'Parameter': names})
    for j, key in enumerate(output_names):
        prcc_table[key] = prcc(X, Y[:, j])
    prcc_table.to_csv(OUT / 'PRCC_results.csv', index=False)
    print('\nPRCC results:')
    print(prcc_table.to_string(index=False, float_format=lambda z: f'{z:+.4f}'))
    print('\nBaseline outputs (cumulative enrolment DS, MDR; final Is, Ir):')
    print(integrate(BASE))
    plot_prcc(prcc_table, output_names[0], 'PRCC_DS_cumulative.png',
              'DS-TB: cumulative treatment initiations (156 weeks)')
    plot_prcc(prcc_table, output_names[1], 'PRCC_MDR_cumulative.png',
              'MDR-TB: cumulative treatment initiations (156 weeks)')
    plot_prcc(prcc_table, output_names[2], 'PRCC_DS_week156.png',
              'DS-TB: infectious population at week 156')
    plot_prcc(prcc_table, output_names[3], 'PRCC_MDR_week156.png',
              'MDR-TB: infectious population at week 156')
    plot_prcc_combined(prcc_table)
    print('\nFiles saved in:', OUT)
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
