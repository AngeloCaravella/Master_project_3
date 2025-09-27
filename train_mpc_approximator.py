# --- START OF FILE train_mpc_approximator.py ---

import numpy as np
import joblib
import os
import glob
import random
import yaml
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.pulp_mpc import OnlineMPC_Solver

# =====================================================================================
# --- PARAMETRI DI ADDESTRAMENTO ---
# =====================================================================================
CONFIG_PATH = 'ev2gym/example_config_files/'
AVAILABLE_SCENARIOS = glob.glob(os.path.join(CONFIG_PATH, "*.yaml"))

if not AVAILABLE_SCENARIOS:
    raise FileNotFoundError(f"Nessun file di scenario .yaml trovato in '{CONFIG_PATH}'")

# --- MODIFICA CHIAVE: Trova il numero massimo di stazioni di ricarica tra tutti gli scenari ---
MAX_CS = 25
#for scenario_file in AVAILABLE_SCENARIOS:
 #   with open(scenario_file, 'r') as f:
   #     config = yaml.safe_load(f)
  #      if 'number_of_charging_stations' in config:
    #        MAX_CS = max(MAX_CS, config['number_of_charging_stations'])

#if MAX_CS == 0:
    #raise ValueError("Impossibile determinare il numero massimo di stazioni di ricarica dai file di configurazione.")

print(f"Trovati {len(AVAILABLE_SCENARIOS)} scenari. Il numero massimo di stazioni di ricarica è: {MAX_CS}")

NUM_SAMPLES = 100  # Impostare a 5000+ per il modello finale
CONTROL_HORIZON = 5
MODEL_SAVE_PATH = 'ev2gym/baselines/mpc_approximator.joblib'

# =====================================================================================
# --- FUNZIONE PER LA CREAZIONE DEL VETTORE DI STATO ---
# =====================================================================================
def build_state_vector(env, max_cs, horizon):
    """Crea il vettore di stato con padding per garantire una dimensione fissa."""
    current_step = env.current_step
    num_cs_in_env = env.cs
    
    ev_socs = np.zeros(max_cs)
    ev_time_to_departure = np.zeros(max_cs)
    
    for i in range(num_cs_in_env):
        ev = next((ev for ev in env.charging_stations[i].evs_connected if ev is not None), None)
        if ev:
            ev_socs[i] = ev.get_soc()
            ev_time_to_departure[i] = max(0, ev.time_of_departure - current_step)

    h = min(horizon, env.simulation_length - current_step)
    prices_charge = env.charge_prices[0, current_step : current_step + h]
    prices_discharge = env.discharge_prices[0, current_step : current_step + h]
    
    padded_prices_ch = np.pad(prices_charge, (0, horizon - len(prices_charge)), 'edge')
    padded_prices_dis = np.pad(prices_discharge, (0, horizon - len(prices_discharge)), 'edge')

    return np.concatenate([ev_socs, ev_time_to_departure, padded_prices_ch, padded_prices_dis])

# =====================================================================================
# --- SCRIPT PRINCIPALE DI ADDESTRAMENTO ---
# =====================================================================================
if __name__ == "__main__":
    print("\n--- Avvio generazione dataset multi-scenario per MPC Esplicito Approssimato ---")
    
    X_data, y_data = [], []
    with tqdm(total=NUM_SAMPLES, desc="Campionamento Stati Multi-Scenario") as pbar:
        while len(X_data) < NUM_SAMPLES:
            env = None
            try:
                selected_config = random.choice(AVAILABLE_SCENARIOS)
                env = EV2Gym(config_file=selected_config, generate_rnd_game=True)
                
                if env.simulation_length <= CONTROL_HORIZON + 1:
                    continue

                random_step = np.random.randint(0, env.simulation_length - CONTROL_HORIZON - 1)
                
                # --- MODIFICA CHIAVE: Avanza l'ambiente fino allo step casuale ---
                env.reset() # Resetta l'ambiente all'inizio
                for _ in range(random_step):
                    # Esegui uno step con un'azione nulla per avanzare
                    env.step(np.zeros(env.cs))

                # Ora l'ambiente è allo step corretto, controlliamo gli EV
                if not any(ev is not None for cs in env.charging_stations for ev in cs.evs_connected):
                    continue

                mpc_oracle = OnlineMPC_Solver(env, control_horizon=CONTROL_HORIZON)
                state_vector = build_state_vector(env, MAX_CS, CONTROL_HORIZON)
                action_normalized = mpc_oracle.get_action(env)
                
                if np.any(action_normalized):
                    optimal_powers = np.zeros(MAX_CS)
                    for i in range(env.cs):
                        max_power = env.charging_stations[i].get_max_power()
                        optimal_powers[i] = action_normalized[i] * max_power
                        
                    X_data.append(state_vector)
                    y_data.append(optimal_powers)
                    pbar.update(1)
            finally:
                if env:
                    env.close()

    if not X_data:
        raise ValueError("Nessun dato di addestramento generato. Controlla i file di configurazione.")

    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    print(f"\nDataset generato. Shape X: {X_data.shape}, Shape y: {y_data.shape}")

    print("--- Avvio addestramento del modello Random Forest ---")
    model = RandomForestRegressor(
        n_estimators=100, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=42
    )
    
    model.fit(X_data, y_data)
    print("Addestramento completato.")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"--- Modello salvato con successo in '{MODEL_SAVE_PATH}' ---")
