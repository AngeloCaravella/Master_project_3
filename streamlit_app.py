import streamlit as st
import os
import sys
import glob
import yaml
import inspect
import subprocess
from typing import List, Dict, Any, Tuple, Callable, Optional

# Aggiungi la directory del progetto al PYTHONPATH per importare ev2gym e run_experiments
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importa le funzioni refactorizzate da run_experiments.py
# Assicurati che run_experiments.py sia stato refactorizzato come previsto
from run_experiments import (
    calculate_max_cs,
    get_algorithms,
    get_scenarios_to_test as get_scenarios_cli,
    get_selected_reward_function as get_rewards_cli,
    get_selected_price_file as get_price_file_cli,
    train_rl_models_if_requested,
    run_benchmark,
    run_fit_battery_if_requested
)

# Per Streamlit, dobbiamo adattare le funzioni CLI o ricreare la logica

def get_available_scenarios(config_path: str) -> List[str]:
    return sorted(glob.glob(os.path.join(config_path, "*.yaml")))

def get_available_reward_functions() -> List[Tuple[str, Callable]]:
    # Importa reward_module da ev2gym.rl_agent
    from ev2gym.rl_agent import reward as reward_module
    return [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]

def get_available_price_files() -> List[str]:
    price_data_dir = os.path.join(project_root, 'ev2gym', 'data')
    return sorted([f for f in os.listdir(price_data_dir) if f.endswith('.csv')])


st.set_page_config(layout="wide", page_title="EV2Gym Simulation Orchestrator")

st.title("EV2Gym Simulation Orchestrator")
st.markdown("Configura ed esegui le simulazioni per il progetto EV2Gym.")

# Inizializza session_state per le checkbox
if 'run_fit_battery_checked' not in st.session_state:
    st.session_state.run_fit_battery_checked = False

# --- Configurazione Percorsi --- 
config_path = os.path.join(project_root, "ev2gym", "example_config_files")

# --- Esegui Fit_battery.py --- 
st.session_state.run_fit_battery_checked = st.sidebar.checkbox("Esegui Fit_battery.py per calibrare il modello di degradazione?")

# --- Selezione Modalità Grafici ---
plot_mode_options = {"1": "Tesi (SAC, DDPG+PER, TQC + Baselines)", "2": "Completa (Tutti gli algoritmi)"}
plot_mode_choice = st.sidebar.radio(
    "Scegli modalità grafici:",
    options=list(plot_mode_options.keys()),
    format_func=lambda x: plot_mode_options[x],
    index=0 # Default to Thesis mode
)
is_thesis_mode = (plot_mode_choice == '1')

# --- Calcolo MAX_CS ---
MAX_CS = calculate_max_cs(config_path)
st.sidebar.info(f"Rilevato un massimo di {MAX_CS} stazioni di ricarica tra tutti gli scenari.")

# --- Selezione Algoritmi ---
algorithms_to_run = get_algorithms(MAX_CS, is_thesis_mode)
algorithm_names = list(algorithms_to_run.keys())
selected_algorithms = st.multiselect(
    "Seleziona gli algoritmi da eseguire:",
    options=algorithm_names,
    default=algorithm_names # Default to all
)

# Filtra algorithms_to_run in base alla selezione dell'utente
algorithms_to_run_filtered = {k: v for k, v in algorithms_to_run.items() if k in selected_algorithms}

# --- Selezione Scenari ---
available_scenarios_full_paths = get_available_scenarios(config_path)
available_scenario_names = [os.path.basename(s).replace(".yaml", "") for s in available_scenarios_full_paths]

selected_scenario_names = st.multiselect(
    "Seleziona gli scenari da testare:",
    options=available_scenario_names,
    default=available_scenario_names # Default to all
)
scenarios_to_test = [s for s in available_scenarios_full_paths if os.path.basename(s).replace(".yaml", "") in selected_scenario_names]

# --- Selezione Funzione di Reward ---
available_rewards = get_available_reward_functions()
reward_options_names = [name for name, _ in available_rewards]
selected_reward_name = st.selectbox(
    "Scegli la funzione di reward:",
    options=reward_options_names,
    index=2 # Default to the 3rd reward function (index 2)
)
selected_reward_func = next(func for name, func in available_rewards if name == selected_reward_name)

# --- Selezione File CSV Prezzi Energia ---
available_price_files = get_available_price_files()
default_price_file_name = "Netherlands_day-ahead-2015-2024.csv"
selected_price_file_name = st.selectbox(
    "Seleziona il file CSV per i prezzi dell'energia:",
    options=available_price_files,
    index=available_price_files.index(default_price_file_name) if default_price_file_name in available_price_files else 0
)

if selected_price_file_name == default_price_file_name:
    selected_price_file_abs_path = "default"
else:
    selected_price_file_abs_path = os.path.join(project_root, 'ev2gym', 'data', selected_price_file_name)

# --- Opzioni di Simulazione ---
num_sims = st.number_input("Quante simulazioni di valutazione per scenario?", min_value=1, value=1)

# --- Addestramento Modelli RL ---
is_multi_scenario = len(scenarios_to_test) > 1
scenario_name_for_path = 'multi_scenario' if is_multi_scenario else os.path.basename(scenarios_to_test[0]).replace(".yaml", "")
model_dir = os.path.join(project_root, f'./saved_models/{scenario_name_for_path}/')

train_rl_models = st.checkbox(f"Vuoi addestrare i modelli RL in modalità {'Multi-Scenario' if is_multi_scenario else 'Single-Domain'}?")
steps_for_training = 0
if train_rl_models:
    steps_for_training = st.number_input("Per quanti passi di training?", min_value=1, value=100000)


# --- Pulsante Esegui Simulazione ---
if st.button("Esegui Simulazione"):
    st.subheader("Output Simulazione")
    output_container = st.empty()
    
    # Costruisci il comando per run_experiments.py
    command = [sys.executable, os.path.join(project_root, "run_experiments.py")]
    
    if st.session_state.run_fit_battery_checked:
        command.append("--run_fit_battery")
    
    # Mappa la scelta del plot_mode da '1'/'2' a 'thesis'/'complete'
    mapped_plot_mode_choice = 'thesis' if plot_mode_choice == '1' else 'complete'
    command.extend(["--plot_mode", mapped_plot_mode_choice])
    
    if selected_scenario_names:
        command.append("--scenarios")
        command.extend(selected_scenario_names)
    
    command.extend(["--reward_func", selected_reward_name])
    
    if selected_price_file_abs_path:
        command.extend(["--price_file", selected_price_file_abs_path])
    
    if train_rl_models:
        command.append("--train_rl_models")
        command.extend(["--steps_for_training", str(steps_for_training)])
    
    command.extend(["--num_sims", str(num_sims)])

    full_output = []
    process = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        with st.spinner("Esecuzione simulazione..."):
            for line in process.stdout:
                full_output.append(line)
                output_container.code("".join(full_output))
            
        process.wait()
        
        if process.returncode != 0:
            st.error(f"Simulazione terminata con errore (codice {process.returncode}).")
        else:
            st.success("Simulazione completata con successo!")

    except Exception as e:
        st.error(f"Errore durante l'esecuzione della simulazione: {e}")
    finally:
        if process and process.poll() is None: # Se il processo è ancora in esecuzione
            process.terminate()
        output_container.code("".join(full_output))

st.markdown("--- \nPer avviare l'applicazione Streamlit, salva questo file come `streamlit_app.py` e esegui `streamlit run streamlit_app.py` nel terminale.")
