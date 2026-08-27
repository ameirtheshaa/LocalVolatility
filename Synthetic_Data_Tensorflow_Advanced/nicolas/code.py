import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde

# ----------------------------------------------------
# Download DAX index
# ----------------------------------------------------
dax = yf.download("^GDAXI",
                  start="2001-08-01",
                  progress=False,
                  auto_adjust=True)

# Closing prices
close = dax["Close"].dropna()

# Keep data from 7 August 2001 onward
close = close.loc["2001-08-07":]

# First 316 trading days
close = close.iloc[:316]

# Log transformation
x = np.log(close.to_numpy().flatten())

# ----------------------------------------------------
# Fit Gaussian
# ----------------------------------------------------
mu = np.mean(x)
sigma = np.std(x, ddof=1)

# Kernel density estimate
kde = gaussian_kde(x)

# ----------------------------------------------------
# Symmetric plotting interval about the mean
# ----------------------------------------------------
r = max(mu - np.min(x), np.max(x) - mu)

xmin = mu - r
xmax = mu + r

xx = np.linspace(xmin, xmax, 1000)

# ----------------------------------------------------
# Plot
# ----------------------------------------------------
plt.figure(figsize=(9,6))

# Histogram
plt.hist(x,
         bins=60,
         range=(xmin, xmax),
         density=True,
         color="lightblue",
         edgecolor="black",
         linewidth=0.4,
         label="Histogram")

# Kernel density estimate
plt.plot(xx,
         kde(xx),
         color="blue",
         linewidth=2.5,
         label="Kernel density")

# Gaussian density
plt.plot(xx,
         norm.pdf(xx, mu, sigma),
         color="red",
         linestyle="--",
         linewidth=2.5,
         label="Gaussian fit")

plt.xlabel(r"$\log(\mathrm{DAX})$", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.title("Log DAX values (first 316 trading days after 7 Aug 2001)",
          fontsize=15)

plt.xlim(xmin, xmax)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
