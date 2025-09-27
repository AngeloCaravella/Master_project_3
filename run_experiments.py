# --- START OF FILE run_experiments.py ---

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
import inspect
import random
import subprocess
from collections import defaultdict
from glob import glob

# --- Importazioni dalla libreria custom ev2gym ---
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import ChargeAsFastAsPossible, ChargeAsLateAsPossible, RoundRobin
from ev2gym.baselines.pulp_mpc import OnlineMPC_Solver, ApproximateExplicitMPC
from ev2gym.rl_agent.custom_algorithms import CustomDDPG
from ev2gym.utilities.per_buffer import PrioritizedReplayBuffer
from ev2gym.rl_agent import reward as reward_module
from ev2gym.rl_agent.state import V2G_profit_max_loads

# --- Importazioni da librerie di RL ---
from stable_baselines3 import SAC, PPO, A2C, TD3, DDPG
from sb3_contrib import TQC, TRPO, ARS
from stable_baselines3.common.callbacks import BaseCallback
from matplotlib.patches import Patch
from stable_baselines3.common.vec_env import DummyVecEnv

# =====================================================================================
# --- CLASSI WRAPPER E AMBIENTI ---
# =====================================================================================
class CompatibilityWrapper(gym.Wrapper):
    def __init__(self, env, target_obs_shape, target_action_shape):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=target_obs_shape, dtype=np.float64)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=target_action_shape, dtype=np.float64)

    def _pad_observation(self, obs):
        padded_obs = np.zeros(self.observation_space.shape, dtype=np.float64)
        padded_obs[:obs.shape[0]] = obs
        return padded_obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._pad_observation(obs), info

    def step(self, action):
        action_size_needed = self.env.action_space.shape[0]
        sliced_action = action[:action_size_needed]
        obs, reward, terminated, truncated, info = self.env.step(sliced_action)
        return self._pad_observation(obs), reward, terminated, truncated, info

class MultiScenarioEnv(gym.Env):
    def __init__(self, config_files, reward_function, state_function):
        super(MultiScenarioEnv, self).__init__()
        self.config_files = config_files
        self.reward_function = reward_function
        self.state_function = state_function
        self.current_env = None
        max_obs_shape, max_action_shape = 0, 0
        for config in self.config_files:
            temp_env = EV2Gym(config_file=config, reward_function=reward_function, state_function=state_function)
            max_obs_shape = max(max_obs_shape, temp_env.observation_space.shape[0])
            max_action_shape = max(max_action_shape, temp_env.action_space.shape[0])
            temp_env.close()
        self.max_obs_shape = (max_obs_shape,)
        self.max_action_shape = (max_action_shape,)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=self.max_action_shape, dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.max_obs_shape, dtype=np.float64)

    def _pad_observation(self, obs):
        padded_obs = np.zeros(self.max_obs_shape, dtype=np.float64)
        padded_obs[:obs.shape[0]] = obs
        return padded_obs

    def reset(self, *, seed=None, options=None):
        if self.current_env: self.current_env.close()
        selected_config = random.choice(self.config_files)
        self.current_env = EV2Gym(config_file=selected_config, generate_rnd_game=True, reward_function=self.reward_function, state_function=self.state_function)
        obs, info = self.current_env.reset(seed=seed, options=options)
        return self._pad_observation(obs), info

    def step(self, action):
        if self.current_env is None: raise RuntimeError("reset() must be called before step().")
        action_size_needed = self.current_env.action_space.shape[0]
        sliced_action = action[:action_size_needed]
        obs, reward, terminated, truncated, info = self.current_env.step(sliced_action)
        return self._pad_observation(obs), reward, terminated, truncated, info

    def close(self):
        if self.current_env: self.current_env.close()

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

# =====================================================================================
# --- FUNZIONI DI PLOTTING (UNIFICATE) ---
# =====================================================================================
def get_color_map_and_legend(algorithms_to_plot):
    full_algo_categories = {
        "AFAP": "heuristic", "ALAP": "heuristic", "RR": "heuristic",
        "Online_MPC": "mpc", "Approx_Explicit_MPC": "mpc", "Online_MPC_Adaptive": "mpc",
        "PPO": "on-policy", "A2C": "on-policy", "TRPO": "on-policy", "ARS": "on-policy",
        "SAC": "off-policy", "TD3": "off-policy", "DDPG": "off-policy", "DDPG+PER": "off-policy", "TQC": "off-policy"
    }
    category_colors = {"heuristic": "#4C72B0", "mpc": "#55A868", "on-policy": "#C44E52", "off-policy": "#8172B2", "default": "#B2B2B2"}
    present_categories = {full_algo_categories[algo] for algo in algorithms_to_plot if algo in full_algo_categories}
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=cat_name) for cat_name, color in zip(['Heuristics', 'MPC', 'On-Policy RL', 'Off-Policy RL'], ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]) if color in [category_colors[cat] for cat in present_categories]]
    return {k: v for k, v in full_algo_categories.items() if k in algorithms_to_plot}, category_colors, legend_elements

def plot_performance_metrics(stats_collection, save_path, scenario_name, algorithms_to_plot):
    if not stats_collection: return
    metrics_map = {
        'total_profits': 'Profitto Totale (€)', 'average_user_satisfaction': 'Soddisfazione Utente Media (%)',
        'peak_transformer_loading_pct': 'Carico di Picco Trasformatore (%)', 'battery_degradation': 'Degradazione Totale Media (%)'
    }
    model_names = [name for name in algorithms_to_plot if name in stats_collection]
    if not model_names: return
    algo_categories, category_colors, legend_elements = get_color_map_and_legend(model_names)
    fig, axes = plt.subplots(2, 2, figsize=(20, 12)); axes = axes.flatten()
    fig.suptitle(f'Metriche di Performance - Scenario: {scenario_name}', fontsize=22)
    for i, (metric, title) in enumerate(metrics_map.items()):
        ax = axes[i]
        values = [stats_collection[name].get(metric, 0) for name in model_names]
        if 'satisfaction' in metric or 'degradation' in metric: values = [v * 100 for v in values]
        colors = [category_colors.get(algo_categories.get(name, "default"), category_colors["default"]) for name in model_names]
        ax.bar(model_names, values, color=colors)
        ax.set_title(title, fontsize=14); ax.tick_params(axis='x', rotation=45); ax.grid(True, linestyle='--', alpha=0.6)
        if '(%)' in title: ax.set_ylim(0, max(105, (max(values) if values else 0) * 1.1))
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(save_path, f"performance_{scenario_name}.png")); plt.close(fig)

def plot_degradation_details(stats_collection, save_path, scenario_name, algorithms_to_plot):
    if not stats_collection: return
    metrics_map = {
        'battery_degradation': 'Degradazione Totale Media (%)',
        'battery_degradation_cyclic': 'Degradazione Ciclica Media (%)',
        'battery_degradation_calendar': 'Degradazione Calendario Media (%)'
    }
    model_names = [name for name in algorithms_to_plot if name in stats_collection]
    if not model_names: return
    algo_categories, category_colors, legend_elements = get_color_map_and_legend(model_names)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
    fig.suptitle(f'Dettaglio Degradazione Batteria - Scenario: {scenario_name}', fontsize=22)
    for i, (metric, title) in enumerate(metrics_map.items()):
        ax = axes[i]
        values = [stats_collection[name].get(metric, 0) * 100 for name in model_names]
        colors = [category_colors.get(algo_categories.get(name, "default"), category_colors["default"]) for name in model_names]
        ax.bar(model_names, values, color=colors)
        ax.set_title(title, fontsize=16); ax.tick_params(axis='x', rotation=45, labelsize=12)
        if i == 0: ax.set_ylabel('Degradazione Media (%)', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements), bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(os.path.join(save_path, f"degradation_{scenario_name}.png")); plt.close(fig)

# =====================================================================================
# --- FUNZIONE DI BENCHMARK UNIFICATA ---
# =====================================================================================
def run_benchmark(config_files, reward_func, algorithms_to_run, num_simulations, model_dir, is_multi_scenario):
    all_scenario_stats = {}
    overall_save_path = f'./results/benchmark_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}/'
    os.makedirs(overall_save_path, exist_ok=True)

    max_obs_shape, max_action_shape = (0,), (0,)
    if is_multi_scenario:
        temp_env = MultiScenarioEnv(config_files, reward_func, V2G_profit_max_loads)
        max_obs_shape, max_action_shape = temp_env.observation_space.shape, temp_env.action_space.shape
        temp_env.close()

    for config_file in config_files:
        scenario_name = os.path.basename(config_file).replace(".yaml", "")
        print(f"\n\n{'='*80}\nAVVIO BENCHMARK PER SCENARIO: {scenario_name}\n{'='*80}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        scenario_save_path = os.path.join(overall_save_path, scenario_name); os.makedirs(scenario_save_path, exist_ok=True)

        all_sim_stats = []
        for sim_num in range(num_simulations):
            print(f"\n--- Simulazione {sim_num + 1}/{num_simulations} ---")
            env_replay = EV2Gym(config_file=config_file, generate_rnd_game=True, save_replay=True)
            replay_path = f"replay/replay_{env_replay.sim_name}.pkl"
            while not env_replay.step(np.zeros(env_replay.action_space.shape[0]))[2]: pass
            env_replay.close()

            eval_env_id = f'eval-env-{scenario_name}-{sim_num}'
            if eval_env_id in registry: del registry[eval_env_id]
            gym.register(id=eval_env_id, entry_point='ev2gym.models.ev2gym_env:EV2Gym', kwargs={'config_file': config_file, 'generate_rnd_game': False, 'load_from_replay_path': replay_path, 'reward_function': reward_func, 'state_function': V2G_profit_max_loads})

            final_stats_collection = {}
            for name, (algorithm_class, rl_class, kwargs) in algorithms_to_run.items():
                print(f"+ Esecuzione: {name}")
                try:
                    env_instance = gym.make(eval_env_id)
                    is_rl_model = rl_class is not None
                    if is_rl_model:
                        if is_multi_scenario: env_instance = CompatibilityWrapper(env_instance, max_obs_shape, max_action_shape)
                        model_path = os.path.join(model_dir, f'{name.lower().replace("+", "_")}_model.zip')
                        if not os.path.exists(model_path): print(f"!!! Modello {name} non trovato in {model_path}. Saltato."); continue
                        model = rl_class.load(model_path, env=env_instance, device=device)
                    else:
                        model = algorithm_class(env=env_instance.unwrapped, **kwargs)

                    obs, _ = env_instance.reset()
                    done = False
                    while not done:
                        if is_rl_model:
                            action, _ = model.predict(obs, deterministic=True)
                        else:
                            action = model.get_action(env_instance.unwrapped)
                        obs, _, done, _, _ = env_instance.step(action)
                    
                    stats = env_instance.unwrapped.stats
                    departed_evs = env_instance.unwrapped.departed_evs
                    if departed_evs:
                        stats['battery_degradation_calendar'] = np.mean([ev.calendar_loss for ev in departed_evs])
                        stats['battery_degradation_cyclic'] = np.mean([ev.cyclic_loss for ev in departed_evs])
                        stats['battery_degradation'] = stats['battery_degradation_calendar'] + stats['battery_degradation_cyclic']
                    final_stats_collection[name] = stats
                    env_instance.close()
                except Exception as e:
                    print(f"!!! ERRORE con '{name}': {e}. Saltato."); traceback.print_exc()
            
            all_sim_stats.append(final_stats_collection)
            if os.path.exists(replay_path): os.remove(replay_path)

        if all_sim_stats:
            aggregated_stats = {name: {metric: np.mean([s[name][metric] for s in all_sim_stats if name in s and metric in s[name]]) for metric in all_sim_stats[0].get(name, {})} for name in algorithms_to_run}
            all_scenario_stats[scenario_name] = aggregated_stats
            plot_performance_metrics(aggregated_stats, scenario_save_path, scenario_name, list(algorithms_to_run.keys()))
            plot_degradation_details(aggregated_stats, scenario_save_path, scenario_name, list(algorithms_to_run.keys()))

    print(f"\n--- Benchmark completato. Risultati salvati in: {overall_save_path} ---")

# =====================================================================================
# --- BLOCCO PRINCIPALE ---
# =====================================================================================
if __name__ == "__main__":
    if input("Vuoi eseguire 'Fit_battery.py' per calibrare il modello di degradazione? (s/n, default n): ").lower() == 's':
        print("--- Esecuzione di Fit_battery.py ---")
        try:
            subprocess.run(["python", "Fit_battery.py"], check=True)
            print("--- Fit_battery.py completato. ---")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ERRORE: {e}. Lo script procederà con i parametri esistenti.")

    plot_mode = input("\nScegli modalità grafici:\n1. Tesi (SAC, DDPG+PER, TQC + Baselines)\n2. Completa (Tutti gli algoritmi)\nScelta (default 1): ") or '1'
    is_thesis_mode = plot_mode == '1'

    # --- MODIFICA CHIAVE: Calcola MAX_CS prima di definire gli algoritmi ---
    config_path_for_cs = "ev2gym/example_config_files/"
    all_scenarios_for_cs = glob(os.path.join(config_path_for_cs, "*.yaml"))
    MAX_CS = 0
    if not all_scenarios_for_cs:
        print(f"ATTENZIONE: Nessun file di scenario trovato in {config_path_for_cs}, MAX_CS impostato a 10 di default.")
        MAX_CS = 10 # Un valore di fallback
    else:
        for scenario_file in all_scenarios_for_cs:
            with open(scenario_file, 'r') as f:
                config = yaml.safe_load(f)
                if 'number_of_charging_stations' in config:
                    MAX_CS = max(MAX_CS, config['number_of_charging_stations'])
    if MAX_CS == 0:
        raise ValueError("Impossibile determinare il numero massimo di stazioni di ricarica. Controlla i file di configurazione.")
    print(f"\nRilevato un massimo di {MAX_CS} stazioni di ricarica tra tutti gli scenari.")
    # --- FINE MODIFICA ---

    # --- DEFINIZIONE ALGORITMI ---
    ALL_ALGORITHMS = {
        "AFAP": (ChargeAsFastAsPossible, None, {}), "ALAP": (ChargeAsLateAsPossible, None, {}), "RR": (RoundRobin, None, {}),
        "Online_MPC": (OnlineMPC_Solver, None, {'control_horizon': 5}),
        "Online_MPC_Adaptive": (OnlineMPC_Solver, None, {
            'use_adaptive_horizon': True, 'h_min': 2, 'h_max': 5, 'lyapunov_alpha': 0.1
        }),
        "Approx_Explicit_MPC": (ApproximateExplicitMPC, None, {
            'control_horizon': 5,
            'max_cs': MAX_CS  # <-- Passa il valore calcolato
        }),
        "SAC": (None, SAC, {}), "PPO": (None, PPO, {}), "A2C": (None, A2C, {}), "TD3": (None, TD3, {}), "DDPG": (None, DDPG, {}),
        "DDPG+PER": (None, CustomDDPG, {'replay_buffer_class': PrioritizedReplayBuffer}),
        "TQC": (None, TQC, {}), "TRPO": (None, TRPO, {}), "ARS": (None, ARS, {})
    }
    THESIS_ALGORITHMS = {k: v for k, v in ALL_ALGORITHMS.items() if k in ["AFAP", "ALAP", "RR", "Online_MPC", "Online_MPC_Adaptive", "Approx_Explicit_MPC", "SAC", "DDPG+PER", "TQC"]}
    algorithms_to_run = THESIS_ALGORITHMS if is_thesis_mode else ALL_ALGORITHMS
    rl_models_to_run = {k: v for k, v in algorithms_to_run.items() if v[1] is not None}

    config_path = "ev2gym/example_config_files/"
    available_scenarios = sorted(glob(os.path.join(config_path, "*.yaml")))
    print("\nScenari disponibili:")
    for i, s in enumerate(available_scenarios): print(f"{i+1}. {os.path.basename(s)}")
    choices = input(f"Seleziona scenari (es. '1 3', 'tutti') (default: tutti): ").lower() or 'tutti'
    scenarios_to_test = available_scenarios if 'tutti' in choices else [available_scenarios[int(i)-1] for i in choices.split()]
    print(f"Scenari selezionati: {[os.path.basename(s) for s in scenarios_to_test]}")

    available_rewards = [(name, func) for name, func in inspect.getmembers(reward_module, inspect.isfunction) if inspect.getmodule(func) == reward_module]
    print("\nScegli la funzione di reward:")
    for i, (name, func) in enumerate(available_rewards):
        doc = inspect.getdoc(func); short_doc = (doc.strip().split('\n')[0] if doc else "Nessuna descrizione.")
        print(f"{i + 1}. {name} - {short_doc}")
    reward_choice = int(input(f"Scelta (default 3): ") or 3)
    selected_reward_func = available_rewards[reward_choice - 1][1]

    is_multi_scenario = len(scenarios_to_test) > 1
    mode_str = "Multi-Scenario" if is_multi_scenario else "Single-Domain"
    scenario_name_for_path = 'multi_scenario' if is_multi_scenario else os.path.basename(scenarios_to_test[0]).replace(".yaml", "")
    model_dir = f'./saved_models/{scenario_name_for_path}/'
    os.makedirs(model_dir, exist_ok=True)

    if input(f"\nVuoi addestrare i modelli RL in modalità {mode_str}? (s/n, default n): ").lower() == 's':
        steps_for_training = int(input("Per quanti passi? (default 100000): ") or 100000)
        train_env_id = f'ev-train-{scenario_name_for_path}'
        if is_multi_scenario:
            train_env = DummyVecEnv([lambda: MultiScenarioEnv(scenarios_to_test, selected_reward_func, V2G_profit_max_loads)])
        else:
            if train_env_id in registry: del registry[train_env_id]
            gym.register(id=train_env_id, entry_point='ev2gym.models.ev2gym_env:EV2Gym', kwargs={'config_file': scenarios_to_test[0], 'generate_rnd_game': True, 'reward_function': selected_reward_func, 'state_function': V2G_profit_max_loads})
            train_env = gym.make(train_env_id)

        for name, (_, rl_class, kwargs) in rl_models_to_run.items():
            print(f"--- Addestramento {name} in modalità {mode_str} ---")
            model = rl_class("MlpPolicy", train_env, verbose=0, device=("cuda" if torch.cuda.is_available() else "cpu"), **kwargs)
            model.learn(total_timesteps=steps_for_training, callback=ProgressCallback(total_timesteps=steps_for_training))
            model.save(os.path.join(model_dir, f'{name.lower().replace("+", "_")}_model.zip'))
        train_env.close()

    num_sims = int(input("\nQuante simulazioni di valutazione per scenario? (default 1): ") or 1)
    run_benchmark(scenarios_to_test, selected_reward_func, algorithms_to_run, num_sims, model_dir, is_multi_scenario)

    print("\n--- ESECUZIONE COMPLETATA ---")
