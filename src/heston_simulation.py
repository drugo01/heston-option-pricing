import numpy as np
import matplotlib.pyplot as plt

# Parametri del modello
S0 = 100   # Prezzo iniziale
V0 = 0.04  # Volatilità iniziale
mu = 0.05  # Rendimento atteso
kappa = 2  # Velocità di mean reversion
theta = 0.04  # Valore di lungo periodo della volatilità
sigma = 0.2   # Volatility of volatility
rho = -0.7    # Correlazione tra S_t e V_t
T = 1.0       # Orizzonte temporale (1 anno)
N = 252       # Passi nel tempo
dt = T / N    # Delta t
M = 10000     # Numero di simulazioni

# Generazione dei moti browniani correlati
dW_S = np.random.normal(0, np.sqrt(dt), (M, N))
dW_V = rho * dW_S + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), (M, N))

# Simulazione dei processi S_t e V_t
S = np.full((M, N+1), S0)
V = np.full((M, N+1), V0)

for t in range(N):
    V[:, t+1] = np.maximum(V[:, t] + kappa * (theta - V[:, t]) * dt + sigma * np.sqrt(V[:, t]) * dW_V[:, t], 0)
    S[:, t+1] = S[:, t] * np.exp((mu - 0.5 * V[:, t]) * dt + np.sqrt(V[:, t]) * dW_S[:, t])

# Plot di alcune simulazioni
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.plot(S[i, :], lw=1)
plt.title("Simulazione del prezzo di un'azione con il modello di Heston")
plt.xlabel("Giorni")
plt.ylabel("Prezzo")
plt.show()
