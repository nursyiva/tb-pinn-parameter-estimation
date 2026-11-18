


    
  
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

# Eight-compartment model: S, Es, Is, Ts, R, Er, Ir, Tr
P = 288315899.0
mu = 0.000269
p = dict(beta_s=0.07647, beta_r=0.07171,
         sigma_s=0.009319, sigma_r=0.00846,
         tau_s=0.05909, tau_r=0.01414,
         kappa=0.9863, rho=0.01523,
         gamma_s=1.0/26.0, gamma_r=0.0075,
         delta_s=0.000439, delta_r=0.00365,
         eta=0.51, omega=0.00278,
         mu=mu, Lambda=mu*P)

Es0, Is0, Ts0, R0 = 100000.0, 75690.0, 70796.0, 50000.0
Er0, Ir0, Tr0 = 1000.0, 941.0, 784.0
S0 = P - sum([Es0, Is0, Ts0, R0, Er0, Ir0, Tr0])
y0 = np.array([S0, Es0, Is0, Ts0, R0, Er0, Ir0, Tr0])
assert np.isclose(y0.sum(), P)

def rhs(t, y):
    S, Es, Is, Ts, R, Er, Ir, Tr = y
    N = sum(y)
    if N <= 0:
        raise ValueError('Nonpositive total population')
    ls = p['beta_s']*Is/N
    lr = p['beta_r']*Ir/N
    total_l = ls+lr
    return np.array([
        p['Lambda']-total_l*S-p['mu']*S+p['omega']*R,
        ls*(S+p['eta']*R)-(p['sigma_s']+p['mu'])*Es,
        p['sigma_s']*Es+p['gamma_s']*(1-p['rho'])*(1-p['kappa'])*Ts-(p['tau_s']+p['mu']+p['delta_s'])*Is,
        p['tau_s']*Is-(p['gamma_s']+p['mu'])*Ts,
        p['gamma_s']*p['kappa']*Ts+p['gamma_r']*Tr-p['eta']*total_l*R-(p['omega']+p['mu'])*R,
        lr*(S+p['eta']*R)-(p['sigma_r']+p['mu'])*Er,
        p['sigma_r']*Er+p['gamma_s']*p['rho']*(1-p['kappa'])*Ts-(p['tau_r']+p['mu']+p['delta_r'])*Ir,
        p['tau_r']*Ir-(p['gamma_r']+p['mu'])*Tr
    ])

a_s=p['sigma_s']+mu
b_s=p['tau_s']+mu+p['delta_s']
c_s=p['gamma_s']+mu
alpha=p['gamma_s']*(1-p['rho'])*(1-p['kappa'])
a_r=p['sigma_r']+mu
b_r=p['tau_r']+mu+p['delta_r']
Rs=p['beta_s']*p['sigma_s']*c_s/(a_s*(b_s*c_s-alpha*p['tau_s']))
Rr=p['beta_r']*p['sigma_r']/(a_r*b_r)
print('Initial state:',dict(zip(['S','Es','Is','Ts','R','Er','Ir','Tr'],y0)))
print(f'Rs={Rs:.8f}, Rr={Rr:.8f}, R0={max(Rs,Rr):.8f}')
t = np.linspace(0,1500,3001)
sol=solve_ivp(rhs,(0,1500),y0,method='LSODA',t_eval=t,rtol=1e-9,atol=1e-5)
if not sol.success: raise RuntimeError(sol.message)
if sol.y.min() < -1e-3: raise RuntimeError('Negative populations')
labels=['S','Es','Is','Ts','R','Er','Ir','Tr']
# Plotting options: display figures after the simulation; saving is optional.
SAVE_FIGURES = False
SHOW_LONG_TERM = True
folder = Path(__file__).resolve().parent

# Long-term dynamics: each compartment has its own Y scale.
if SHOW_LONG_TERM:
    fig_long, axs = plt.subplots(4, 2, figsize=(13, 13), constrained_layout=True)
    for ax, k in zip(axs.flat, range(8)):
        values = sol.y[k] / 1e6 if k in (0, 4) else sol.y[k]
        ax.plot(sol.t, values, linewidth=1.8)
        ax.set_title(labels[k])
        ax.set_xlabel('Time (weeks)')
        ax.set_ylabel('Millions of individuals' if k in (0, 4) else 'Individuals')
        ax.grid(alpha=0.25)
        ax.margins(y=0.08)
   # fig_long.suptitle('Baseline TB dynamics: 0–1500 weeks', fontsize=14)
    if SAVE_FIGURES:
        fig_long.savefig(folder / 'Figure_Baseline_Dynamics_Revised.png', dpi=180)

# Short-term plot: slice the DATA, not just the X axis.
# This prevents the large 1500-week values from flattening MDR-TB curves.
short = sol.t <= 160
short_t = sol.t[short]
fig_short, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
for ax, k in zip(axs.flat, [2, 3, 6, 7]):
    short_y = sol.y[k, short]
    ax.plot(short_t, short_y, linewidth=2, label=labels[k])
    ax.set_xlim(0, 160)
    ax.set_title(labels[k])
    ax.set_xlabel('Time (weeks)')
    ax.set_ylabel('Individuals')
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.margins(y=0.12)
    ax.grid(alpha=0.25)
    ax.legend()
#fig_short.suptitle('Baseline infectious and treated TB: first 160 weeks', fontsize=14)
if SAVE_FIGURES:
    fig_short.savefig(folder / 'Figure_Baseline_First160Weeks_Revised.png', dpi=180)

idx160 = int(np.argmin(np.abs(sol.t - 160)))
print('At 160 weeks:', dict(zip(labels, np.round(sol.y[:, idx160]).astype(int))))
print('At 1500 weeks:', dict(zip(labels, np.round(sol.y[:, -1]).astype(int))))
print('End residual max abs:', np.max(np.abs(rhs(1500, sol.y[:, -1]))))
print('Close the figure windows to finish the program.')
plt.show()  # Works as a window when run from Python IDLE with a GUI backend.
