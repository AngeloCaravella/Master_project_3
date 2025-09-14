import math
import numpy as np
from collections import deque
from copy import deepcopy

# ===============================================================================================
# ========= NUOVA FUNZIONE ADATTIVA, VELOCE E OTTIMIZZATA PER IL PROFITTO =========================
# ===============================================================================================

def FastProfitAdaptiveReward(env, total_costs, user_satisfaction_list, *args):
    """
    Una funzione di reward adattiva, ottimizzata per la velocità computazionale e la massimizzazione del profitto.

    Privilegia in modo aggressivo il profitto economico, applicando penalità che si adattano
    dinamicamente in base alla performance recente del sistema riguardo la soddisfazione dell'utente
    e il sovraccarico del trasformatore.
    """
    # Inizializza i tracker di stato se non esistono. Per una maggiore pulizia,
    # questa logica andrebbe nella funzione __init__ o reset dell'ambiente.
    if not hasattr(env, 'satisfaction_history'):
        env.satisfaction_history = deque(maxlen=100)
    if not hasattr(env, 'overload_frequency'):
        env.overload_frequency = deque(maxlen=100)

    # La ricompensa principale è il profitto economico diretto.
    reward = total_costs
    reward_components = {'profit': reward}

    # Penalità Adattiva per la Soddisfazione dell'Utente
    avg_satisfaction = sum(env.satisfaction_history) / len(env.satisfaction_history) if env.satisfaction_history else 1.0
    satisfaction_severity_multiplier = 20.0 * (1 - avg_satisfaction)**2
    
    current_satisfaction_penalty = 0
    if user_satisfaction_list:
        min_satisfaction = min(user_satisfaction_list)
        if min_satisfaction < 0.95:
            current_satisfaction_penalty = -satisfaction_severity_multiplier * (1 - min_satisfaction)
    
    if current_satisfaction_penalty < 0:
        reward += current_satisfaction_penalty
        reward_components['satisfaction_penalty'] = current_satisfaction_penalty

    # Penalità Adattiva per il Sovraccarico del Trasformatore
    overload_freq = sum(env.overload_frequency) / len(env.overload_frequency) if env.overload_frequency else 0.0
    overload_severity_multiplier = 50.0 * overload_freq

    current_overload_amount = sum(tr.get_how_overloaded() for tr in env.transformers)
    if current_overload_amount > 0:
        overload_penalty = -5.0 - (overload_severity_multiplier * current_overload_amount)
        reward += overload_penalty
        reward_components['transformer_penalty'] = overload_penalty

    # Aggiorna le cronologie per il prossimo passo
    avg_current_satisfaction = sum(user_satisfaction_list) / len(user_satisfaction_list) if user_satisfaction_list else 1.0
    env.satisfaction_history.append(avg_current_satisfaction)
    env.overload_frequency.append(1 if current_overload_amount > 0 else 0)

    if hasattr(env, 'step_info'):
        env.step_info['reward_components'] = reward_components

    return reward

# ===============================================================================================
# ========= FUNZIONI ORIGINALI DEL PROGETTO (MANTENUTE PER COMPATIBILITÀ) =======================
# ===============================================================================================

def SelfBalancingAdaptiveReward(env, total_costs, user_satisfaction_list, *args):
    '''
    FUNZIONE DI REWARD ADATTIVA CON AUTO-BILANCIAMENTO E STABILITÀ MIGLIORATA.
    '''
    if not hasattr(env, 'reward_tracker'):
        env.reward_tracker = {
            'baseline_importance': {
                'profit': 1.5, 'tracking_error': 1.0, 'transformer_overload': 2.0,
                'user_satisfaction': 0.5, 'action_smoothness': 0.5
            },
            'adaptive_importance': {},
            'last_action_vector': np.zeros(env.action_space.shape),
            'satisfaction_history': deque(maxlen=24),
            'overload_history': deque(maxlen=24),
            'running_stats': {}
        }
        env.reward_tracker['adaptive_importance'] = deepcopy(env.reward_tracker['baseline_importance'])
    def update_welford_stats(stats_dict, key, value):
        if key not in stats_dict: stats_dict[key] = {'count': 0, 'mean': 0.0, 'M2': 0.0}
        stats = stats_dict[key]; stats['count'] += 1; delta = value - stats['mean']
        stats['mean'] += delta / stats['count']; delta2 = value - stats['mean']; stats['M2'] += delta * delta2
    def get_std_dev(stats_dict, key):
        stats = stats_dict.get(key)
        if not stats or stats['count'] < 2: return 1.0
        return math.sqrt(stats['M2'] / stats['count'])
    reward_components = {}; stats = env.reward_tracker['running_stats']
    current_action_total = env.current_power_usage[env.current_step-1]
    current_action_vector = env.last_rl_action if hasattr(env, 'last_rl_action') else np.array([current_action_total])
    raw_profit = total_costs; raw_tracking_error = (env.power_setpoints[env.current_step-1] - current_action_total)**2
    raw_overload = sum(tr.get_how_overloaded() for tr in env.transformers); raw_satisfaction_penalty = 0
    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is not None and ev.get_soc() < 0.99:
                time_to_departure = ev.time_of_departure - env.current_step
                urgency = (1 - ev.get_soc()) / max(time_to_departure, 1)
                raw_satisfaction_penalty += (1 - ev.get_soc()) * math.tanh(urgency * 5)
    last_action_vector = env.reward_tracker['last_action_vector']
    raw_smoothness_penalty = np.linalg.norm(current_action_vector - last_action_vector)**2
    env.reward_tracker['last_action_vector'] = current_action_vector
    for key, value in [('profit', raw_profit), ('tracking_error', raw_tracking_error), ('overload', raw_overload), ('satisfaction', raw_satisfaction_penalty), ('smoothness', raw_smoothness_penalty)]:
        update_welford_stats(stats, key, value)
    adaptive_weights = env.reward_tracker['adaptive_importance']; baseline_weights = env.reward_tracker['baseline_importance']
    avg_satisfaction_this_step = np.mean(user_satisfaction_list) if user_satisfaction_list else 1.0
    env.reward_tracker['satisfaction_history'].append(avg_satisfaction_this_step)
    if np.mean(env.reward_tracker['satisfaction_history']) < 0.85: adaptive_weights['user_satisfaction'] *= 1.01
    else: adaptive_weights['user_satisfaction'] *= 0.99
    adaptive_weights['user_satisfaction'] = max(baseline_weights['user_satisfaction'], adaptive_weights['user_satisfaction'])
    env.reward_tracker['overload_history'].append(raw_overload)
    if np.mean(env.reward_tracker['overload_history']) > 0.05: adaptive_weights['transformer_overload'] *= 1.01
    else: adaptive_weights['transformer_overload'] *= 0.99
    adaptive_weights['transformer_overload'] = max(baseline_weights['transformer_overload'], adaptive_weights['transformer_overload'])
    def normalize(key, value):
        mean = stats[key]['mean']; std = get_std_dev(stats, key)
        return (value - mean) / std
    reward_components['profit'] = adaptive_weights['profit'] * math.tanh(normalize('profit', raw_profit))
    reward_components['tracking_penalty'] = -adaptive_weights['tracking_error'] * normalize('tracking_error', raw_tracking_error)
    reward_components['transformer_penalty'] = -adaptive_weights['transformer_overload'] * normalize('overload', raw_overload)
    reward_components['satisfaction_penalty'] = -adaptive_weights['user_satisfaction'] * normalize('satisfaction', raw_satisfaction_penalty)
    reward_components['smoothness_penalty'] = -adaptive_weights['action_smoothness'] * normalize('smoothness', raw_smoothness_penalty)
    reward_components['tracking_bonus'] = max(0, 1.0 - 10 * raw_tracking_error / (stats['tracking_error']['mean'] + 1e-8))
    reward_components['no_overload_bonus'] = max(0, 1.0 - raw_overload)
    if np.mean(env.reward_tracker['satisfaction_history']) > 0.95: reward_components['high_satisfaction_bonus'] = 1.5
    total_reward = sum(reward_components.values()); update_welford_stats(stats, 'total_reward', total_reward)
    reward_mean = stats['total_reward']['mean']; reward_std = get_std_dev(stats, 'total_reward')
    clip_range = (reward_mean - 5 * reward_std, reward_mean + 5 * reward_std)
    clipped_reward = np.clip(total_reward, clip_range[0], clip_range[1])
    if hasattr(env, 'step_info'):
        env.step_info['reward_components'] = reward_components; env.step_info['final_reward'] = clipped_reward
    return clipped_reward

def ProfitFocusedReward(env, total_costs, user_satisfaction_list, *args):
    reward = total_costs
    reward_components = {'profit': reward}
    overload = sum(tr.get_how_overloaded() for tr in env.transformers)
    if overload > 0:
        transformer_penalty = -50 * (overload**2)
        reward += transformer_penalty
        reward_components['transformer_penalty'] = transformer_penalty
    satisfaction_penalty = 0
    for score in user_satisfaction_list:
        if score < 0.95:
            satisfaction_penalty -= 20 * (1 - score)
    if satisfaction_penalty < 0:
        reward += satisfaction_penalty
        reward_components['satisfaction_penalty'] = satisfaction_penalty
    num_fully_charged = sum(1 for score in user_satisfaction_list if score >= 0.99)
    if num_fully_charged > 0:
        fully_charged_bonus = 5.0 * num_fully_charged
        reward += fully_charged_bonus
        reward_components['fully_charged_bonus'] = fully_charged_bonus
    if hasattr(env, 'step_info'):
        env.step_info['reward_components'] = reward_components
    return reward

def AdaptiveStateBasedReward(env, total_costs, user_satisfaction_list, *args):
    reward = total_costs
    TRANSFORMER_CRITICAL_THRESHOLD = 0.95
    TRANSFORMER_PENALTY_SCALER = 200.0
    for tr in env.transformers:
        current_load = env.current_power_usage[env.current_step-1] / len(env.transformers)
        max_power = tr.max_power[env.current_step-1]
        if max_power > 0:
            load_percentage = current_load / max_power
            overload_amount = tr.get_how_overloaded()
            if overload_amount > 0:
                penalty_weight = TRANSFORMER_PENALTY_SCALER
                if load_percentage > TRANSFORMER_CRITICAL_THRESHOLD:
                    penalty_weight *= math.exp(20 * (load_percentage - TRANSFORMER_CRITICAL_THRESHOLD))
                reward -= penalty_weight * overload_amount
    SATISFACTION_PENALTY_SCALER = 1000.0
    URGENCY_EXP_FACTOR = 5.0
    for cs in env.charging_stations:
        for ev in cs.evs_connected:
            if ev is not None:
                soc = ev.get_soc()
                if soc >= 0.99: continue
                time_to_departure = ev.time_of_departure - env.current_step
                urgency_score = (1 - soc) / max(time_to_departure, 1)
                penalty_weight = SATISFACTION_PENALTY_SCALER * math.exp(URGENCY_EXP_FACTOR * urgency_score)
                reward -= penalty_weight * (1 - soc)
    TRACKING_ERROR_WEIGHT = 0.5
    tracking_error = (env.power_setpoints[env.current_step-1] - env.current_power_usage[env.current_step-1])**2
    reward -= TRACKING_ERROR_WEIGHT * tracking_error
    return reward

def BalancedProfitAndTracking(env, total_costs, user_satisfaction_list, *args):
    reward = total_costs
    tracking_error = (env.power_setpoints[env.current_step-1] - env.current_power_usage[env.current_step-1])**2
    reward -= 0.5 * tracking_error
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    for score in user_satisfaction_list:
        reward -= 1000 * math.exp(-10 * score)
    return reward

def SqTrError_TrPenalty_UserIncentives(env, _, user_satisfaction_list, *args):
    tr_max_limit = env.transformers[0].max_power[env.current_step-1]
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1],tr_max_limit) -
        env.current_power_usage[env.current_step-1])**2
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()
    for score in user_satisfaction_list:
        reward -= 2000 * (1 - score)
    return reward

def ProfitMax_TrPenalty_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    reward = total_costs
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    for score in user_satisfaction_list:
        reward -= 500 * math.exp(-10 * score)
    return reward

def profit_maximization(env, total_costs, user_satisfaction_list, *args):
    reward = total_costs
    for score in user_satisfaction_list:
        reward -= 1000 * math.exp(-10 * score)
    return reward

def SquaredTrackingErrorReward(env,*args):
    '''This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative'''
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
        env.current_power_usage[env.current_step-1])**2
    return reward

def profit_maximization_with_soft_constraints(env, total_costs, user_satisfaction_list, *args):
    """
    Questa funzione di reward è progettata per massimizzare il profitto,
    pur considerando la soddisfazione dell'utente e i limiti del trasformatore come "soft constraints".
    """
    reward = total_costs
    for tr in env.transformers:
        reward -= 10 * tr.get_how_overloaded()
    for score in user_satisfaction_list:
        reward -= 10 * math.exp(-10 * score)
    return reward
def SquaredTrackingErrorReward(env,*args):
    '''This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative'''
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
    env.current_power_usage[env.current_step-1])**2
    return reward
