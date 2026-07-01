import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/log_gcp_a.csv")

plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["train_loss"], label="Train loss", color="#2255dd")
plt.plot(df["epoch"], df["val_loss"],   label="Val loss",   color="#dd4422")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Curva de entrenamiento — Modelo A")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("experimentos/curva_entrenamiento.png", dpi=150)
plt.show()