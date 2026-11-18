
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from pathlib import Path

# ==========================================
# 1. PARAMETER MODEL
# ==========================================

P = 288315899.0

X0 = np.array([
    288016688, 100000, 75690, 70796,
    50000, 1000, 941, 784
], dtype=float)

p = {
    "beta_s": 0.07647,
    "beta_r": 0.07171,
    "sigma_s": 0.009319,
    "sigma_r": 0.00846,
    "tau_s": 0.05909,
    "tau_r": 0.01414,
    "kappa": 0.9863,
    "rho": 0.01523,
    "mu": 0.000269,
    "delta_s": 0.000439,
    "delta_r": 0.00365,
    "gamma_s": 1.0/26.0,
    "gamma_r": 0.0075,
    "omega": 0.00278,
    "eta": 0.51
}

p["Lambda"] = p["mu"] * P

T = 156.0
tgrid = np.linspace(0, T, 157)

A1, A2 = 1.0, 20.0
B1, B2 = 1e5, 3e3

U1MAX = 0.8
U2MAX = 1.0

OUT = Path("hasil_FBSM")
OUT.mkdir(exist_ok=True)

# ==========================================
# 2. MODEL DELAPAN KOMPARTEMEN
# ==========================================

def rhs(y, u1, u2):

    S, Es, Is, Ts, R, Er, Ir, Tr = y

    N = np.sum(y)

    ls = (1-u1)*p["beta_s"]*Is/N
    lr = (1-u1)*p["beta_r"]*Ir/N

    kap = p["kappa"]+(1-p["kappa"])*u2

    mu = p["mu"]
    gs = p["gamma_s"]
    gr = p["gamma_r"]
    rho = p["rho"]
    eta = p["eta"]

    dS = (
        p["Lambda"]-(ls+lr)*S
        -mu*S+p["omega"]*R
    )

    dEs = ls*(S+eta*R)-(p["sigma_s"]+mu)*Es

    dIs = (
        p["sigma_s"]*Es
        +gs*(1-rho)*(1-kap)*Ts
        -(p["tau_s"]+mu+p["delta_s"])*Is
    )

    dTs = p["tau_s"]*Is-(gs+mu)*Ts

    dR = (
        gs*kap*Ts+gr*Tr
        -eta*(ls+lr)*R
        -(p["omega"]+mu)*R
    )

    dEr = lr*(S+eta*R)-(p["sigma_r"]+mu)*Er

    dIr = (
        p["sigma_r"]*Er
        +gs*rho*(1-kap)*Ts
        -(p["tau_r"]+mu+p["delta_r"])*Ir
    )

    dTr = p["tau_r"]*Ir-(gr+mu)*Tr

    return np.array([
        dS,dEs,dIs,dTs,dR,dEr,dIr,dTr
    ])

# ==========================================
# 3. JACOBIAN DENGAN COMPLEX-STEP
# ==========================================

def jacobian(y, u1, u2):

    h = 1e-20
    J = np.zeros((8,8))

    for j in range(8):
        yc = np.asarray(y, dtype=complex).copy()
        yc[j] += 1j*h

        J[:,j] = np.imag(rhs(yc,u1,u2))/h

    return J

# ==========================================
# 4. SOLUSI STATE MAJU
# ==========================================

def forward(u1, u2):

    def fun(t,y):
        a = np.interp(t,tgrid,u1)
        b = np.interp(t,tgrid,u2)
        return rhs(y,a,b)

    sol = solve_ivp(
        fun,(0,T),X0,
        method="LSODA",
        t_eval=tgrid,
        dense_output=True,
        rtol=1e-7,
        atol=1e-5
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol

# ==========================================
# 5. SOLUSI ADJOINT MUNDUR
# ==========================================

def backward(state_sol, u1, u2):

    gradL = np.array([
        0,0,A1,0,0,0,A2,0
    ],dtype=float)

    def adjoint(t,lam):

        y = state_sol.sol(t)

        a = np.interp(t,tgrid,u1)
        b = np.interp(t,tgrid,u2)

        J = jacobian(y,a,b)

        return -gradL-J.T@lam

    sol = solve_ivp(
        adjoint,
        (T,0),
        np.zeros(8),
        method="LSODA",
        t_eval=tgrid[::-1],
        dense_output=True,
        rtol=1e-6,
        atol=1e-5
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol

# ==========================================
# 6. KARAKTERISASI KONTROL OPTIMAL
# ==========================================


def update_controls(states, adjoints):

    u1new = np.zeros(len(tgrid))
    u2new = np.zeros(len(tgrid))

    for i in range(len(tgrid)):

        y = states[:, i]
        lam = adjoints[:, i]

        S, Es, Is, Ts, R, Er, Ir, Tr = y
        N = np.sum(y)

        bs = p["beta_s"] * Is / N
        br = p["beta_r"] * Ir / N

        eta = p["eta"]
        gs = p["gamma_s"]
        rho = p["rho"]
        k0 = p["kappa"]

        # Turunan state terhadap u1
        df_du1 = np.zeros(8)

        df_du1[0] = (bs + br) * S
        df_du1[1] = -bs * (S + eta * R)
        df_du1[4] = eta * (bs + br) * R
        df_du1[5] = -br * (S + eta * R)

        # Turunan state terhadap u2
        df_du2 = np.zeros(8)

        df_du2[2] = (
            -gs * (1-rho) * (1-k0) * Ts
        )

        df_du2[4] = (
            gs * (1-k0) * Ts
        )

        df_du2[6] = (
            -gs * rho * (1-k0) * Ts
        )

        # Kondisi stasioner Hamiltonian
        u1_star = -np.dot(lam, df_du1) / B1
        u2_star = -np.dot(lam, df_du2) / B2

        # Proyeksi ke batas kontrol
        u1new[i] = np.clip(u1_star, 0, U1MAX)
        u2new[i] = np.clip(u2_star, 0, U2MAX)
        
    return u1new, u2new


# ==========================================
# 7. FORWARD-BACKWARD SWEEP
# ==========================================

def solve_fbsm(strategy):

    u1 = np.zeros(len(tgrid))
    u2 = np.zeros(len(tgrid))

    if strategy in ["B","D"]:
        u1[:] = 0.4

    if strategy in ["C","D"]:
        u2[:] = 0.9
    if strategy == "A":
        return forward(u1,u2),u1,u2,True

    maxiter = 300
    tol = 1e-5
    relaxation = 0.3

    converged = False

    for iteration in range(maxiter):

        state_sol = forward(u1,u2)

        adj_sol = backward(
            state_sol,u1,u2
        )

        states = state_sol.y
        adjoints = adj_sol.y[:,::-1]

        v1,v2 = update_controls(
            states,adjoints
        )

        if strategy == "B":
            v2[:] = 0.0

        elif strategy == "C":
            v1[:] = 0.0

        new_u1 = (
            relaxation*v1
            +(1-relaxation)*u1
        )

        new_u2 = (
            relaxation*v2
            +(1-relaxation)*u2
        )

        error = max(
            np.max(np.abs(new_u1-u1)),
            np.max(np.abs(new_u2-u2))
        )

        u1,u2 = new_u1,new_u2

        if iteration%20 == 0:
            print(
                strategy,
                "iteration",iteration,
                "error",error
            )

        if error < tol:
            converged = True
            print(
                strategy,
                "converged at iteration",
                iteration+1
            )
            break

    if not converged:
        print(
            "WARNING:",
            strategy,
            "did not converge"
        )

    # Hitung ulang state dengan kontrol terakhir
    state_sol = forward(u1,u2)

    return state_sol,u1,u2,converged

# ==========================================
# 8. JALANKAN EMPAT STRATEGI
# ==========================================

results = {}

for strategy in ["A","B","C","D"]:

    print("\nStrategy",strategy)

    sol,u1,u2,converged = solve_fbsm(strategy)

    results[strategy] = {
        "sol":sol,
        "u1":u1,
        "u2":u2,
        "converged":converged
    }
    
print("\nNILAI KONTROL OPTIMAL PADA t=0")

for strategy in ["B", "C", "D"]:
    print(
        f"Strategy {strategy}: "
        f"u1(0) = {results[strategy]['u1'][0]:.6f}, "
        f"u2(0) = {results[strategy]['u2'][0]:.6f}"
    )
# ==========================================
# 9. GRAFIK PROFIL KONTROL
# ==========================================

fig,axs = plt.subplots(
    2,1,figsize=(9,7),
    sharex=True
)

styles = {
    "B":("blue","--"),
    "C":("orange","-"),
    "D":("green","-.")
}

for strategy in ["B","C","D"]:

    r = results[strategy]
    color,ls = styles[strategy]

    axs[0].plot(
        tgrid,r["u1"],
        color=color,
        linestyle=ls,
        linewidth=2,
        label="Strategy "+strategy
    )

    axs[1].plot(
        tgrid,r["u2"],
        color=color,
        linestyle=ls,
        linewidth=2,
        label="Strategy "+strategy
    )

axs[0].set_ylabel(r"$u_1(t)$")
axs[1].set_ylabel(r"$u_2(t)$")
axs[1].set_xlabel("Time (weeks)")

axs[0].set_title("Transmission reduction")
axs[1].set_title("Treatment adherence enhancement")

for ax in axs:
    ax.set_ylim(-0.02,1.05)
    ax.grid(alpha=0.25)
    ax.legend()

plt.tight_layout()

plt.savefig(
    OUT/"Figure_FBSM_Control.pdf",
    bbox_inches="tight"
)

plt.savefig(
    OUT/"Figure_FBSM_Control.png",
    dpi=300,bbox_inches="tight"
)

plt.show()

# ==========================================
# 10. GRAFIK DINAMIKA INFEKSI
# ==========================================

fig,axs = plt.subplots(
    1,2,figsize=(12,5)
)

styles2 = {
    "A":("blue","-"),
    "B":("orange","--"),
    "C":("green",":"),
    "D":("red","-.")
}

for strategy in ["A","B","C","D"]:

    sol = results[strategy]["sol"]
    color,ls = styles2[strategy]

    axs[0].plot(
        sol.t,sol.y[2],
        color=color,
        linestyle=ls,
        linewidth=2,
        label="Strategy "+strategy
    )

    axs[1].plot(
        sol.t,sol.y[6],
        color=color,
        linestyle=ls,
        linewidth=2,
        label="Strategy "+strategy
    )

axs[0].set_title("DS-TB")
axs[1].set_title("MDR-TB")

for ax in axs:
    ax.set_xlabel("Time (weeks)")
    ax.set_ylabel("Infectious population")
    ax.grid(alpha=0.25)
    ax.legend()

plt.tight_layout()

plt.savefig(
    OUT/"Figure_FBSM_Dynamics.pdf",
    bbox_inches="tight"
)

plt.savefig(
    OUT/"Figure_FBSM_Dynamics.png",
    dpi=300,bbox_inches="tight"
)

plt.show()

# ==========================================
# 11. TABEL HASIL
# ==========================================

summary = []

for strategy,r in results.items():

    sol = r["sol"]
    u1 = r["u1"]
    u2 = r["u2"]

    Is = sol.y[2]
    Ir = sol.y[6]

    burden = np.trapz(
        A1*Is+A2*Ir,
        tgrid
    )

    effort = np.trapz(
        0.5*B1*u1**2
        +0.5*B2*u2**2,
        tgrid
    )

    summary.append({
        "Strategy":strategy,
        "Converged":r["converged"],
        "Is_156":Is[-1],
        "Ir_156":Ir[-1],
        "Burden":burden,
        "Control_effort":effort,
        "Objective_J":burden+effort
    })

df = pd.DataFrame(summary)

print("\nFBSM RESULTS")
print(df.to_string(index=False))

df.to_csv(
    OUT/"FBSM_summary.csv",
    index=False
)

print("\nSELESAI")
print("Folder hasil:",OUT.resolve())
