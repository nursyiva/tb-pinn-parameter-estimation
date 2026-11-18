
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pathlib import Path

# ==============================================
# 1. PARAMETER MODEL TB
# ==============================================

P = 288315899.0

mu = 0.000269

# Parameter DS-TB
beta_s = 0.05       # Skenario bifurkasi
beta_s_baseline = 0.07647
sigma_s = 0.009319
tau_s = 0.05909
delta_s = 0.000439
gamma_s = 1.0 / 26.0

rho = 0.01523
kappa = 0.9863

# Parameter MDR-TB
sigma_r = 0.00846
tau_r = 0.01414
delta_r = 0.00365
gamma_r = 0.0075

omega = 0.00278
eta = 0.51

beta_r_baseline = 0.07171

Lambda = mu * P

# ==============================================
# 2. BILANGAN REPRODUKSI
# ==============================================

A_s = sigma_s + mu
B_s = tau_s + mu + delta_s
C_s = gamma_s + mu

denom_s = (
    A_s *
    (
        B_s * C_s
        - tau_s * gamma_s *
        (1.0 - rho) * (1.0 - kappa)
    )
)

Rs = (
    beta_s * sigma_s * C_s
    / denom_s
)

Rs_baseline = (
    beta_s_baseline * sigma_s * C_s
    / denom_s
)

# Ambang MDR-TB
beta_critical = (
    (sigma_r + mu) *
    (tau_r + mu + delta_r)
    / sigma_r
)

Rr_baseline = (
    beta_r_baseline / beta_critical
)

print("=" * 55)
print("ANALISIS BIFURKASI MDR-TB")
print("=" * 55)

print("Beta_s skenario =", beta_s)
print("Rs skenario     =", Rs)
print("Rs baseline     =", Rs_baseline)
print("Beta_r kritis   =", beta_critical)
print("Rr baseline     =", Rr_baseline)

if Rs >= 1.0:
    raise ValueError(
        "Rs harus kurang dari 1 "
        "untuk bifurkasi DFE sistem penuh."
    )

# ==============================================
# 3. KESETIMBANGAN MDR-ONLY
# ==============================================

a = (
    (tau_r + mu + delta_r)
    / sigma_r
)

c = tau_r / (gamma_r + mu)


def endemic_ir(Rr):

    if Rr <= 1.0:
        return 0.0

    beta_r = Rr * beta_critical

    def equilibrium_state(x):

        # x = Ir / N
        N = Lambda / (
            mu + delta_r * x
        )

        Ir = x * N
        Tr = c * Ir
        Er = a * Ir

        R = (
            gamma_r * Tr
            / (
                omega + mu
                + eta * beta_r * x
            )
        )

        S = N - R - Er - Ir - Tr

        return S, R, Er, Ir, Tr, N

    def equilibrium_equation(x):

        S, R, Er, Ir, Tr, N = (
            equilibrium_state(x)
        )

        return (
            beta_r * (S + eta * R) / N
            - beta_critical
        )

    # Batas atas proporsi infeksi
    xmax = 1.0 / (1.0 + a + c)

    f0 = equilibrium_equation(0.0)
    fmax = equilibrium_equation(xmax)

    if f0 * fmax > 0:
        return np.nan

    xstar = brentq(
        equilibrium_equation,
        0.0,
        xmax,
        xtol=1e-13
    )

    return equilibrium_state(xstar)[3]


# ==============================================
# 4. HITUNG CABANG KESETIMBANGAN
# ==============================================

Rr_values = np.linspace(
    1.001, 5.0, 350
)

Ir_values = np.array([
    endemic_ir(Rr)
    for Rr in Rr_values
])

Ir_baseline = endemic_ir(
    Rr_baseline
)

print()
print("Ir* baseline =", Ir_baseline)

# ==============================================
# 5. GRAFIK BIFURKASI
# ==============================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

# DFE stabil untuk Rr < 1
# karena Rs skenario < 1
ax.plot(
    [0, 1],
    [0, 0],
    color="blue",
    linestyle="-",
    linewidth=3,
    label="DFE Stable"
)

# DFE tidak stabil untuk Rr > 1
ax.plot(
    [1, 5],
    [0, 0],
    color="blue",
    linestyle="--",
    linewidth=3,
    label="DFE Unstable"
)

# Cabang MDR-only
ax.plot(
    Rr_values,
    Ir_values / 1e6,
    color="red",
    linewidth=2.8,
    label="MDR-only equilibrium Stable"
)

# Titik bifurkasi
ax.scatter(
    [1],
    [0],
    color="black",
    s=50,
    zorder=5
)

# Garis ambang
ax.axvline(
    x=1.0,
    color="gray",
    linestyle=":",
    linewidth=1.2
)

ax.set_xlabel(
    r"$\mathcal{R}_r$",
    fontsize=14
)

ax.set_ylabel(
    r"$I_r^*$ (millions)",
    fontsize=14
)

ax.set_xlim(0, 5)
ax.set_ylim(bottom=0)

ax.grid(alpha=0.2)
ax.legend(fontsize=11)

plt.tight_layout()

# ==============================================
# 6. SIMPAN GAMBAR
# ==============================================

# Simpan di folder yang sama dengan kode Python
try:
    folder_output = (
        Path(__file__).resolve().parent
    )
except NameError:
    folder_output = Path.cwd()

nama_file = "Figure_Bifurcation_MDR"

fig.savefig(
    folder_output / (nama_file + ".pdf"),
    bbox_inches="tight"
)

fig.savefig(
    folder_output / (nama_file + ".png"),
    dpi=300,
    bbox_inches="tight"
)

print()
print("Grafik disimpan di:")
print(folder_output)
print()
print("File:")
print(nama_file + ".pdf")
print(nama_file + ".png")

plt.show()

print("Selesai.")
