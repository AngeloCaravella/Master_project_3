import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ---- Modelli EV2Gym semplificati ----
def dcal(t_days, SoC_mean, theta, eps0, eps1, eps2):
    return 0.75 * (eps0 * SoC_mean - eps1) * np.exp(-eps2/theta) * t_days / (t_days+1)**0.25

def dcyc(E_exchanged, SoC_mean, z0, z1, Qacc):
    return (z0 + z1 * abs(SoC_mean-0.5)) * E_exchanged / np.sqrt(Qacc)

def Qlost_model(inputs, eps0, eps1, eps2, z0, z1, Qacc):
    t_days, SoC_mean, theta, E_exchanged = inputs
    theta_K = np.array(theta) + 273.15
    return dcal(t_days, SoC_mean, theta_K, eps0, eps1, eps2) + dcyc(E_exchanged, SoC_mean, z0, z1, Qacc)

# ---- Caricamento e Preparazione dei Dati ----
NOME_FILE = "cell_eocv2_P016_1_S15_C11.csv"

try:
    df = pd.read_csv(NOME_FILE, sep=';')
    print(f"File '{NOME_FILE}' caricato con successo.")
except Exception as e:
    print(f"ERRORE durante il caricamento del file: {e}")
    exit()

if df.empty:
    print("ERRORE: Il file è stato letto ma risulta vuoto.")
    exit()

# ---- PULIZIA DEI DATI ----
df.columns = df.columns.str.strip()
colonne_essenziali = [
    'cap_aged_est_Ah', 'timestamp_s', 'age_soc',
    't_start_degC', 'total_e_chg_sum_Wh', 'total_e_dischg_sum_Wh'
]
df.dropna(subset=colonne_essenziali, inplace=True)

if df.empty:
    print("ERRORE: Dopo aver rimosso i valori mancanti, non ci sono più dati.")
    exit()

# ---- Estrazione delle variabili dal DataFrame pulito ----
try:
    Q0 = df["cap_aged_est_Ah"].iloc[0]
    Qlost_measured = Q0 - df["cap_aged_est_Ah"]
    t_days = df["timestamp_s"].values / 86400
    theta = df["t_start_degC"].values
    E_exchanged = (df["total_e_chg_sum_Wh"].values + df["total_e_dischg_sum_Wh"].values) / 1000

    # ---- SOLUZIONE AL VALUEERROR ----
    # 'SoC_mean' deve essere un array della stessa lunghezza degli altri input.
    # Prendiamo il singolo valore di SoC e creiamo un array pieno di quel valore.
    soc_value = df["age_soc"].iloc[0] / 100.0
    SoC_mean = np.full_like(t_days, soc_value)
    # ---------------------------------

except KeyError as e:
    print(f"\nERRORE: Impossibile trovare la colonna {e}.")
    exit()

# ---- Fit dei Parametri del Modello ----
p0 = [6e6, 1.3e6, 7000, 4e-4, 2e-3, 12000]
print("\nInizio del processo di fitting...")
popt, pcov = curve_fit(
    Qlost_model,
    (t_days, SoC_mean, theta, E_exchanged),
    Qlost_measured,
    p0=p0,
    maxfev=20000
)
print("Fitting completato.")
print("\nParametri stimati (eps0, eps1, eps2, zeta0, zeta1, Qacc):")
print(popt)

# ---- Visualizzazione dei Risultati ----
plt.figure(figsize=(10, 6))
plt.scatter(t_days, Qlost_measured, label="Dati reali (puliti)", color="tab:blue", s=15)
plt.plot(t_days, Qlost_model((t_days, SoC_mean, theta, E_exchanged), *popt),
         label="Fit del modello", color="tab:red", linewidth=2)
plt.xlabel("Tempo [giorni]")
plt.ylabel("Capacità persa [Ah]")
plt.title("Confronto tra Dati Reali e Modello di Invecchiamento Batteria")
plt.legend()
plt.grid(True)
plt.show()
