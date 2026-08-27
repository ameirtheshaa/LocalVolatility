import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm, gaussian_kde

# -----------------------------
# Parameters
# -----------------------------
start_date = "2001-08-07"
n_days = 316                 # number of trading days
annual_rate = 0.045          # realistic EUR short rate (~4.5% in 2001-2002)
trading_days = 252

# -----------------------------
# Download DAX
# -----------------------------
dax = yf.download("^GDAXI",
                  start=start_date,
                  progress=False,
                  auto_adjust=True)

dax = dax.dropna()

prices = dax["Close"].iloc[:n_days]

# -----------------------------
# Discount prices
# -----------------------------
times = np.arange(len(prices)) / trading_days
discount = np.exp(-annual_rate * times)

discounted = prices.values.flatten() * discount

# -----------------------------
# Standardize
# -----------------------------
z = (discounted - np.mean(discounted)) / np.std(discounted, ddof=1)

# -----------------------------
# Symmetric plotting interval
# -----------------------------
L = max(abs(z.min()), abs(z.max()))
L = np.ceil(L * 10) / 10

x = np.linspace(-L, L, 800)

# -----------------------------
# Kernel density estimate
# -----------------------------
kde = gaussian_kde(z)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(9,6))

plt.hist(
    z,
    bins=50,
    density=True,
    alpha=0.40,
    edgecolor="black",
    linewidth=0.4,
    label="Standardized discounted DAX"
)

plt.plot(
    x,
    kde(x),
    lw=2,
    label="Kernel density estimate"
)

plt.plot(
    x,
    norm.pdf(x),
    "r--",
    lw=2,
    label="Standard Gaussian"
)

plt.xlim([-L, L])

plt.xlabel("Standardized discounted index value")
plt.ylabel("Density")
plt.title("Discounted DAX values vs Standard Gaussian")
plt.legend()

plt.tight_layout()
plt.show()
