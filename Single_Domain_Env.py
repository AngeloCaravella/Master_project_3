import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import datetime
import gymnasium as gym
from gymnasium.envs.registration import registry
import time
import torch
import traceback
import inspect  # Per leggere le funzioni dal modulo reward
from collections import defaultdict

# --- NOTA IMPORTANTE ---
# Questo script richiede che le seguenti librerie siano installate nel tuo ambiente Python:
# pip install stable-baselines3[extra] sb3-contrib gymnasium pyyaml matplotlib torch pulp
# Richiede inoltre che la libreria 'ev2gym' sia installata o accessibile nel path di Python.

# --- Importazioni dalla libreria custom ev2gym ---
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible, ChargeAsLateAsPossible, RoundRobin
from ev2gym.baselines.pulp_mpc import eMPC_V2G_PuLP
from ev2gym.rl_agent.custom_algorithms import CustomDDPG
from ev2gym.utilities.per_buffer import PrioritizedReplayBuffer
from ev2gym.rl_agent import reward as reward_module
from ev2gym.rl_agent.state import V2G_profit_max_loads, V2G_profit_max

# --- Importazioni da librerie standard e di RL ---
from stable_baselines3 import SAC, PPO, A2C, TD3, DDPG
from sb3_contrib import TQC, TRPO, ARS
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --- DEFINIZIONE CLASSE CUSTOM ---
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, check_freq: int = 100, verbose: int = 1):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps, self.check_freq = total_timesteps, check_freq
        self.episode_count, self.start_time = 0, time.time()

    def _on_step(self) -> bool:
        if 'dones' in self.locals and any(self.locals['dones']):
            self.episode_count += 1
            if self.episode_count % self.check_freq == 0:
                progress = self.num_timesteps / self.total_timesteps
                elapsed_time = time.time() - self.start_time
                eta = elapsed_time * (1 - progress) / progress if progress > 0 else 0
                print(f"Episodio: {self.episode_count} | Timesteps: {self.num_timesteps}/{self.total_timesteps}({progress:.2%}) | ETA: {time.strftime('%H:%M:%S', time.gmtime(eta))}")
        return True

# --- FUNZIONI DI PLOTTING ---
def get_color_map_and_legend():
    algo_categories = {
        "AFAP": "heuristic", "ALAP": "heuristic", "RR": "heuristic", "MPC_PuLP": "mpc",
        "PPO": "on-policy", "A2C": "on-policy", "TRPO": "on-policy", "ARS": "on-policy",
        "SAC": "off-policy", "TD3": "off-policy", "DDPG": "off-policy", "DDPG+PER": "off-policy", "TQC": "off-policy"
    }
    category_colors = {
        "heuristic": "#4C72B0", "mpc": "#55A868", "on-policy": "#C44E52", "off-policy": "#8172B2", "default": "#B2B2B2"
    }
    legend_elements = [
        Patch(facecolor=category_colors['heuristic'], edgecolor='black', label='Heuristics'),
        Patch(facecolor=category_colors['mpc'], edgecolor='black', label='MPC'),
        Patch(facecolor=category_colors['on-policy'], edgecolor='black', label='On-Policy RL'),
        Patch(facecolor=category_colors['off-policy'], edgecolor='black', label='Off-Policy RL')
    ]
    return algo_categories, category_colors, legend_elements

def plot_performance_metrics_bars(stats_collection, save_path, scenario_name):
    if not stats_collection: return
    metrics_to_plot = {
        'total_profits': 'Profitto Totale (€)', 'average_user_satisfaction': 'Soddisfazione Utente Media (%)',
        'peak_transformer_loading_pct': 'Carico di Picco sul Trasformatore (%)', 'average_transformer_loading_pct': 'Carico Medio sul Trasformatore (%)'
    }
    model_names = list(stats_collection.keys())
    algo_categories, category_colors, _ = get_color_map_and_legend()
    fig, axes = plt.subplots(2, 2, figsize=(20, 12)); axes = axes.flatten()
    fig.suptitle(f'Metriche di Performance Finali - Scenario: {scenario_name}', fontsize=22)
    for i, (metric_key, title) in enumerate(metrics_to_plot.items()):
        ax = axes[i]
        valid_models = [name for name in model_names if name in stats_collection and metric_key in stats_collection[name]]
        if not valid_models:
            ax.text(0.5, 0.5, 'Metrica non disponibile', ha='center', va='center', transform=ax.transAxes); ax.set_title(title, fontsize=14)
            continue
        values = [stats_collection[name][metric_key] for name in valid_models]
        if 'satisfaction' in metric_key: values = [v * 100 for v in values]
        ax.bar(valid_models, values, color=[category_colors.get(algo_categories.get(name, 'default')) for name in valid_models])
        ax.set_title(title, fontsize=14); ax.tick_params(axis='x', rotation=45); ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylabel('Percentuale (%)' if '(%)' in title else 'Valore', fontsize=12)
        if '(%)' in title: ax.set_ylim(0, max(105, (max(values) if values else 0) * 1.1))
    _, _, legend_elements = get_color_map_and_legend()
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(save_path, f"performance_metrics_{scenario_name}.png")); print(f"Grafico metriche di performance salvato."); plt.close(fig)

def plot_temporal_graphs(all_sim_results, config, save_path, scenario_name):
    if not any(all_sim_results) or not all_sim_results[0]: return
    model_names = list(all_sim_results[0].keys())
    if not model_names: return
    algo_categories, category_colors, legend_elements = get_color_map_and_legend()
    simulation_length = len(all_sim_results[0][model_names[0]]['power'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 16), sharex=True)
    fig.suptitle(f'Benchmark Temporale (Aggregato) - Scenario: {scenario_name}', fontsize=22)
    time_steps = np.arange(simulation_length) * config['timescale'] / 3600
    avg_prices_charge = np.mean([sim['prices_charge'] for sim in all_sim_results[0].values()], axis=0)
    avg_prices_discharge = np.mean([sim['prices_discharge'] for sim in all_sim_results[0].values()], axis=0)
    ax1.plot(time_steps, avg_prices_charge, 'r-o', label='Prezzo Carica Medio (€/kWh)', markersize=4)
    ax1.plot(time_steps, avg_prices_discharge, 'b-o', label='Prezzo Scarica Medio (€/kWh)', markersize=4)
    ax1.set_ylabel('Prezzo (€/kWh)'); ax1.set_title('Profilo Prezzi Medio'); ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.6)
    for name in model_names:
        power_data = [sim_res[name]['power'] for sim_res in all_sim_results if name in sim_res]
        avg_power = np.mean(power_data, axis=0); std_power = np.std(power_data, axis=0)
        color = category_colors.get(algo_categories.get(name, 'default'))
        ax2.plot(time_steps, avg_power, label=f'{name} (Media)', alpha=0.9, linewidth=2, color=color)
        ax2.fill_between(time_steps, avg_power - std_power, avg_power + std_power, alpha=0.2, color=color)
    tr_limit = config['transformer']['max_power']
    ax2.axhline(y=tr_limit, color='k', linestyle='--', linewidth=2, label=f'Limite Trasformatore ({tr_limit} kW)')
    avg_demand = np.mean([sim_res[model_names[0]]['demand'] for sim_res in all_sim_results], axis=0)
    ax2.plot(time_steps, avg_demand, 'k:', label='Carico Base / Setpoint', linewidth=2)
    ax2.set_title('Azioni di Potenza Medie'); ax2.set_ylabel('Potenza Totale (kW)'); ax2.set_xlabel('Tempo (ore)')
    handles, labels = ax2.get_legend_handles_labels(); ax2.legend(handles, labels, loc='upper right')
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(save_path, f"temporal_comparison_{scenario_name}.png")); print(f"Grafico temporale salvato."); plt.close(fig)

def plot_soc_graphs(all_sim_results, config, save_path, scenario_name):
    if not any(all_sim_results) or not all_sim_results[0]: return
    model_names = list(all_sim_results[0].keys())
    if not model_names or 'soc' not in all_sim_results[0][model_names[0]]: print("Dati SoC non trovati, grafico saltato."); return
    algo_categories, category_colors, legend_elements = get_color_map_and_legend()
    simulation_length = len(all_sim_results[0][model_names[0]]['soc'])
    fig, ax = plt.subplots(1, 1, figsize=(22, 10))
    fig.suptitle(f'Evoluzione Stato di Carica (SoC) Medio - Scenario: {scenario_name}', fontsize=22)
    time_steps = np.arange(simulation_length) * config['timescale'] / 3600
    for name in model_names:
        soc_data = [sim_res[name]['soc'] for sim_res in all_sim_results if name in sim_res and 'soc' in sim_res[name]]
        if not soc_data: continue
        avg_soc = np.mean(soc_data, axis=0) * 100; std_soc = np.std(soc_data, axis=0) * 100
        color = category_colors.get(algo_categories.get(name, 'default'))
        ax.plot(time_steps, avg_soc, label=f'{name} (Media)', alpha=0.9, linewidth=2, color=color)
        ax.fill_between(time_steps, avg_soc - std_soc, avg_soc + std_soc, alpha=0.2, color=color)
    desired_soc = config.get('ev_config', {}).get('final_soc', 1.0) * 100
    ax.axhline(y=desired_soc, color='r', linestyle='--', linewidth=2, label=f'SoC Desiderato ({desired_soc:.0f}%)')
    ax.set_title('Stato di Carica Medio dei Veicoli Connessi'); ax.set_ylabel('Stato di Carica (SoC) (%)'); ax.set_xlabel('Tempo (ore)'); ax.set_ylim(0, 110)
    handles, labels = ax.get_legend_handles_labels(); ax.legend(handles, labels, loc='lower right')
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(save_path, f"soc_evolution_{scenario_name}.png")); print(f"Grafico evoluzione SoC salvato."); plt.close(fig)

def calculate_and_plot_final_benchmark(all_scenario_stats, save_path):
    print(f"\n{'='*30} CALCOLO BENCHMARK FINALE AGGREGATO {'='*30}")
    if not all_scenario_stats:
        print("Nessun dato raccolto per il benchmark finale. Saltato."); return

    final_data = defaultdict(lambda: defaultdict(list))
    all_model_names = set()
    for _, stats in all_scenario_stats.items():
        for model_name, metrics in stats.items():
            all_model_names.add(model_name)
            for metric_key, value in metrics.items():
                final_data[model_name][metric_key].append(value)

    summary_stats = defaultdict(dict)
    for model_name in all_model_names:
        for metric_key, values in final_data[model_name].items():
            if values:
                summary_stats[model_name][f'{metric_key}_mean'] = np.mean(values)
                summary_stats[model_name][f'{metric_key}_std'] = np.std(values)

    metrics_to_plot = {
        'total_profits': 'Profitto Totale Medio (€)',
        'average_user_satisfaction': 'Soddisfazione Utente Media (%)',
        'peak_transformer_loading_pct': 'Carico di Picco Medio sul Trasformatore (%)'
    }
    
    sorted_model_names = sorted(list(all_model_names))
    algo_categories, category_colors, legend_elements = get_color_map_and_legend()
    bar_colors = [category_colors.get(algo_categories.get(name, 'default')) for name in sorted_model_names]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(24, 8), sharey=False)
    fig.suptitle('Benchmark Aggregato - Performance Media su Tutti gli Scenari', fontsize=24)

    for i, (metric_key, title) in enumerate(metrics_to_plot.items()):
        ax = axes[i]
        means = [summary_stats.get(name, {}).get(f'{metric_key}_mean', 0) for name in sorted_model_names]
        stds = [summary_stats.get(name, {}).get(f'{metric_key}_std', 0) for name in sorted_model_names]
        if 'satisfaction' in metric_key:
            means = [v * 100 for v in means]; stds = [s * 100 for s in stds]
        ax.bar(sorted_model_names, means, yerr=stds, color=bar_colors, capsize=5, ecolor='black', alpha=0.8)
        ax.set_title(title, fontsize=16); ax.tick_params(axis='x', rotation=60, labelsize=12)
        ax.grid(True, linestyle='--', alpha=0.6); ax.set_ylabel('Valore Medio', fontsize=14)

    fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01), fontsize=14)
    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    final_plot_path = os.path.join(save_path, "final_benchmark_summary.png")
    plt.savefig(final_plot_path); print(f"\n>>> Grafico del benchmark finale salvato in: {final_plot_path} <<<"); plt.close(fig)

def run_benchmark_for_scenario(config_file, reward_func_to_use, num_simulations=1, models_to_retrain=[], steps_for_training=10000):
    scenario_name = os.path.basename(config_file).replace(".yaml", "")
    print(f"\n\n{'='*80}\nAVVIO BENCHMARK PER SCENARIO: {scenario_name}\n{'='*80}")

    with open(config_file, 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    model_dir = f'./saved_models/{scenario_name}/'; os.makedirs(model_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    algorithms_to_compare = {
        "AFAP": (ChargeAsFastAsPossible, {}), "ALAP": (ChargeAsLateAsPossible, {}), "RR": (RoundRobin, {}),
        "MPC_PuLP": (eMPC_V2G_PuLP, {'control_horizon': 5}),
        "SAC": (SAC, {}), "PPO": (PPO, {}), "A2C": (A2C, {}), "TD3": (TD3, {}), "DDPG": (DDPG, {}), 
        "DDPG+PER": (CustomDDPG, {'replay_buffer_class': PrioritizedReplayBuffer}), 
        "TQC": (TQC, {}), "TRPO": (TRPO, {}), "ARS": (ARS, {})
    }
    
    rl_model_classes = {
        "SAC": SAC, "PPO": PPO, "A2C": A2C, "TD3": TD3, "DDPG": DDPG,
        "DDPG+PER": CustomDDPG, "TQC": TQC, "TRPO": TRPO, "ARS": ARS
    }

    if models_to_retrain:
        env_id = f'evs-train-{scenario_name}'
        if env_id in registry: del registry[env_id]
        gym.register(id=env_id, entry_point='ev2gym.models.ev2gym_env:EV2Gym', kwargs={'config_file': config_file, 'generate_rnd_game': True, 'reward_function': reward_func_to_use, 'state_function': V2G_profit_max_loads})
        for name in models_to_retrain:
            if name in rl_model_classes:
                print(f"--- Addestramento per {name} sullo scenario {scenario_name} ---")
                env_train = gym.make(env_id)
                model_params = algorithms_to_compare.get(name, (None, {}))[1]
                model = rl_model_classes[name]("MlpPolicy", env_train, verbose=0, device=device, **model_params)
                model.learn(total_timesteps=steps_for_training, callback=ProgressCallback(total_timesteps=steps_for_training))
                model.save(os.path.join(model_dir, f'{name.lower().replace("+", "_")}_model.zip')); env_train.close()

    all_sim_results, all_sim_stats = [], []
    save_path = f'./results/evaluation_{scenario_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}/'; os.makedirs(save_path, exist_ok=True)

    for sim_num in range(num_simulations):
        print(f"\n--- VALUTAZIONE (Scenario: {scenario_name}): Simulazione {sim_num + 1}/{num_simulations} ---")
        env_replay = EV2Gym(config_file=config_file, generate_rnd_game=True, save_replay=True)
        reference_replay_path = f"replay/replay_{env_replay.sim_name}.pkl"
        full_simulation_length = env_replay.simulation_length
        
        for _ in range(full_simulation_length):
            if env_replay.step(np.zeros(env_replay.action_space.shape[0]))[2]: break
        env_replay.close()
        
        eval_env_id = f'evs-eval-{scenario_name}-{sim_num}'
        if eval_env_id in registry: del registry[eval_env_id]
        gym.register(id=eval_env_id, entry_point='ev2gym.models.ev2gym_env:EV2Gym', kwargs={'config_file': config_file, 'generate_rnd_game': False, 'load_from_replay_path': reference_replay_path, 'reward_function': reward_func_to_use, 'state_function': V2G_profit_max_loads})
        
        results_data, final_stats_collection = {}, {}
        for name, (algorithm_class, kwargs) in algorithms_to_compare.items():
            print(f"+ Esecuzione: {name}")
            try:
                is_rl_model = name in rl_model_classes
                env_instance = gym.make(eval_env_id) if is_rl_model else EV2Gym(config_file=config_file, load_from_replay_path=reference_replay_path)
                real_env = env_instance.unwrapped if is_rl_model else env_instance
                
                model_path = os.path.join(model_dir, f'{name.lower().replace("+", "_")}_model.zip')
                if is_rl_model and not os.path.exists(model_path):
                    print(f"!!! Modello {name} non trovato. Saltato."); continue

                model = rl_model_classes[name].load(model_path, env=env_instance, device=device) if is_rl_model else algorithm_class(env=real_env, **kwargs)
                obs, _ = env_instance.reset()

                power_history, soc_history = [], []
                done = False
                while not done:
                    # *** INIZIO BLOCCO CORRETTO ***
                    action = model.predict(obs, deterministic=True)[0] if is_rl_model else model.get_action(real_env)
                    # *** FINE BLOCCO CORRETTO ***
                    obs, _, done, _, _ = env_instance.step(action)
                    
                    power_history.append(np.sum(real_env.current_power_usage))
                    connected_evs = [ev for cs in real_env.charging_stations for ev in cs.evs_connected if ev]
                    soc_history.append(np.mean([ev.get_soc() for ev in connected_evs]) if connected_evs else (soc_history[-1] if soc_history else 0))
                
                actual_len = len(power_history)
                padded_power = np.zeros(full_simulation_length); padded_power[:actual_len] = power_history
                padded_soc = np.zeros(full_simulation_length); padded_soc[:actual_len] = soc_history
                
                total_load = np.sum([tr.inflexible_load_kw for tr in real_env.transformers if hasattr(tr, 'inflexible_load_kw')], axis=0)
                if not isinstance(total_load, np.ndarray): total_load = np.zeros(full_simulation_length)
                
                total_solar = np.sum([tr.solar_power_kw for tr in real_env.transformers if hasattr(tr, 'solar_power_kw')], axis=0)
                if not isinstance(total_solar, np.ndarray): total_solar = np.zeros(full_simulation_length)
                
                demand_history = total_load - total_solar

                final_stats_collection[name] = real_env.stats
                results_data[name] = {
                    'power': padded_power, 'demand': demand_history,
                    'prices_charge': real_env.charge_prices[0], 'prices_discharge': real_env.discharge_prices[0], 'soc': padded_soc
                }
                
                env_instance.close()
            except Exception as e:
                print(f"!!! ERRORE con '{name}': {e}. Saltato."); traceback.print_exc()
        all_sim_results.append(results_data); all_sim_stats.append(final_stats_collection)
        if os.path.exists(reference_replay_path): os.remove(reference_replay_path)

    aggregated_stats = {name: {metric: np.mean([s[name][metric] for s in all_sim_stats if s and name in s and metric in s[name]]) for metric in (all_sim_stats[0][name].keys() if all_sim_stats and all_sim_stats[0] and name in all_sim_stats[0] else [])} for name in algorithms_to_compare.keys()}
    
    plot_performance_metrics_bars(aggregated_stats, save_path, scenario_name)
    plot_temporal_graphs(all_sim_results, config, save_path, scenario_name)
    plot_soc_graphs(all_sim_results, config, save_path, scenario_name)
    
    return aggregated_stats, save_path

if __name__ == "__main__":
    scenarios_to_test = ["ev2gym/example_config_files/V2GProfitMax.yaml",
                         "ev2gym/example_config_files/V2GProfitPlusLoads.yaml",
                         "ev2gym/example_config_files/V2GProfitMax_1ev.yaml",
                         "ev2gym/example_config_files/V2GProfitPlusLoads_1ev.yaml"]
    num_sims_per_scenario = 1; steps_for_training = 50000; models_to_retrain = []

    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    
    print("Scegli la funzione di reward da utilizzare per le simulazioni:")
    for i, (name, func) in enumerate(available_rewards):
        doc = inspect.getdoc(func); short_doc = (doc.strip().split('\n')[0] if doc else "Nessuna descrizione.")
        print(f"{i + 1}. {name} - {short_doc}")

    reward_choice = -1
    while not (1 <= reward_choice <= len(available_rewards)):
        try: reward_choice = int(input(f"Inserisci la tua scelta (1 - {len(available_rewards)}): "))
        except ValueError: print("Inserisci un numero valido.")

    selected_reward_name, selected_reward_func = available_rewards[reward_choice - 1]
    print(f"\n--- Hai selezionato la funzione di reward: {selected_reward_name} ---")
    
    rl_model_keys = ["SAC", "PPO", "A2C", "TD3", "DDPG", "DDPG+PER", "TQC", "TRPO", "ARS"]
    if input("Vuoi riaddestrare i modelli RL? (s/n, default n): ").lower() == 's':
        print("Modelli disponibili: " + ", ".join(rl_model_keys))
        models_choice = input("Quali modelli vuoi addestrare? (es. 'SAC PPO', lascia vuoto per tutti): ")
        models_to_retrain = models_choice.split() if models_choice else rl_model_keys
        steps_input = input(f"Per quanti passi? (default: {steps_for_training}): ")
        if steps_input.strip(): steps_for_training = int(steps_input)

    all_scenario_stats = {}
    final_save_path = f'./results/final_benchmark_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}/'
    os.makedirs(final_save_path, exist_ok=True)
    
    for config_path in scenarios_to_test:
        if os.path.exists(config_path):
            scenario_stats, _ = run_benchmark_for_scenario(
                config_file=config_path, reward_func_to_use=selected_reward_func,
                num_simulations=num_sims_per_scenario, models_to_retrain=models_to_retrain, 
                steps_for_training=steps_for_training
            )
            scenario_name = os.path.basename(config_path).replace(".yaml", "")
            all_scenario_stats[scenario_name] = scenario_stats
        else:
            print(f"ATTENZIONE: File di configurazione non trovato: {config_path}. Scenario saltato.")
    
    calculate_and_plot_final_benchmark(all_scenario_stats, final_save_path)
    print("\n\n--- Tutti i benchmark sono stati completati. ---")
