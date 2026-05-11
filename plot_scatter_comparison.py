import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# LOAD COMPARISON FILE
# ============================================
df = pd.read_csv("runup_comparison.csv")

# ============================================
# SCATTER PLOT
# ============================================
plt.figure(figsize=(7,7))

sc = plt.scatter(
    df["Runup_A"],
    df["Runup_B"],
    c=df["Difference"],
    s=40,
    cmap="RdBu_r"
)

# 1:1 line
mn = min(df["Runup_A"].min(), df["Runup_B"].min())
mx = max(df["Runup_A"].max(), df["Runup_B"].max())

plt.plot(
    [mn, mx],
    [mn, mx],
    'k--',
    linewidth=1
)

# Labels
plt.xlabel("Run-up A (m)")
plt.ylabel("Run-up B (m)")

plt.title("SWASH Run-up Comparison")

# Equal axes
plt.axis("equal")

# Colorbar
cbar = plt.colorbar(sc)

cbar.set_label("Difference (B - A) [m]")

# Grid
plt.grid(True)

plt.tight_layout()

plt.show()
