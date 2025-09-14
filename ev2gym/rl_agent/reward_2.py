'''This file contains various example reward functions for the RL agent. Users can create their own reward function here or in their own file using the same structure as below
'''

import math

# NUOVA FUNZIONE DI REWARD BILANCIATA
def BalancedProfitAndTracking(env, total_costs, user_satisfaction_list, *args):
    '''
    MODIFICA: Questa è una nuova funzione di reward progettata per bilanciare tre obiettivi chiave
    identificati come critici dall'analisi dei grafici:
    1. Massimizzazione del profitto (il segnale di base).
    2. Penalità per l'errore di tracciamento (per ridurre l'errore enorme visto nei grafici).
    3. Penalità molto alta per la bassa soddisfazione utente (per risolvere il problema più grande).
    '''
    
    # 1. Inizia con il profitto come ricompensa base
    reward = total_costs
    
    # 2. Aggiungi una forte penalità per l'errore di tracciamento
    # Il fattore 0.5 è un iperparametro da sintonizzare, ma serve a rendere la penalità
    # competitiva con i valori di profitto/costo.
    tracking_error = (env.power_setpoints[env.current_step-1] - env.current_power_usage[env.current_step-1])**2
    reward -= 0.5 * tracking_error
    
    # 3. Mantieni la penalità per il sovraccarico del trasformatore (funzionava già bene)
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    
    # 4. Aggiungi una penalità molto più alta per la bassa soddisfazione utente
    for score in user_satisfaction_list:
        # La penalità è stata aumentata di 50 volte (da 100 a 1000) per forzare l'agente
        # a dare priorità alla ricarica completa degli EV.
        reward -= 1000 * math.exp(-10*score)
        
    return reward

def SquaredTrackingErrorReward(env,*args):
    '''This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative'''
    
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
        env.current_power_usage[env.current_step-1])**2
        
    return reward

def SqTrError_TrPenalty_UserIncentives(env, _, user_satisfaction_list, *args):
    ''' This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    It penalizes transofrmers that are overloaded    
    The reward is negative'''
    
    tr_max_limit = env.transformers[0].max_power[env.current_step-1]
    
    reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1],tr_max_limit) -
        env.current_power_usage[env.current_step-1])**2
            
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()
        
    for score in user_satisfaction_list:
        # MODIFICA: Aumentato il peso della penalità da 1000 a 2000 per migliorare la soddisfazione utente.
        reward -= 2000 * (1 - score)
        # CODICE PRECEDENTE:
        # reward -= 1000 * (1 - score)
                    
    return reward

def ProfitMax_TrPenalty_UserIncentives(env, total_costs, user_satisfaction_list, *args):
    
    reward = total_costs
    
    for tr in env.transformers:
        reward -= 100 * tr.get_how_overloaded()                        
    
    for score in user_satisfaction_list:
        # MODIFICA: Aumentato drasticamente il peso della penalità da 100 a 500 per forzare
        # l'agente a dare priorità alla soddisfazione utente rispetto al profitto.
        reward -= 500 * math.exp(-10*score)
        # CODICE PRECEDENTE:
        # reward -= 100 * math.exp(-10*score)
        
    return reward

def SquaredTrackingErrorRewardWithPenalty(env,*args):
    ''' This reward function is the squared tracking error that uses the minimum of the power setpoints and the charge power potential
    The reward is negative
    If the EV is not charging, the reward is penalized
    '''
    if env.current_power_usage[env.current_step-1] == 0 and env.charge_power_potential[env.current_step-2] != 0:
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2 - 100
    else:
        reward = - (min(env.power_setpoints[env.current_step-1], env.charge_power_potential[env.current_step-1]) -
            env.current_power_usage[env.current_step-1])**2
    
    return reward

def SimpleReward(env,*args):
    '''This reward function does not consider the charge power potential'''
    
    reward = - (env.power_setpoints[env.current_step-1] - env.current_power_usage[env.current_step-1])**2
    
    return reward

def MinimizeTrackerSurplusWithChargeRewards(env,*args):
    ''' This reward function minimizes the tracker surplus and gives a reward for charging '''
    
    reward = 0
    if env.power_setpoints[env.current_step-1] < env.current_power_usage[env.current_step-1]:
            reward -= (env.current_power_usage[env.current_step-1]-env.power_setpoints[env.current_step-1])**2

    reward += env.current_power_usage[env.current_step-1] #/75
    
    return reward

def profit_maximization(env, total_costs, user_satisfaction_list, *args):
    ''' This reward function is used for the profit maximization case '''
    
    reward = total_costs
    
    for score in user_satisfaction_list:
        # MODIFICA: Aumentato drasticamente il peso della penalità da 100 a 1000 per forzare
        # l'agente a dare priorità alla soddisfazione utente rispetto al profitto.
        reward -= 1000 * math.exp(-10*score)
        # CODICE PRECEDENTE:
        # reward -= 100 * math.exp(-10*score)
    
    return reward

def profit_maximization_with_soft_constraints(env, total_costs, user_satisfaction_list, *args):
    """
    Questa funzione di reward è progettata per massimizzare il profitto,
    pur considerando la soddisfazione dell'utente e i limiti del trasformatore come "soft constraints".
    """
    # La ricompensa di base è il profitto
    reward = total_costs

    # Aggiungiamo una piccola penalità per il sovraccarico del trasformatore
    for tr in env.transformers:
        reward -= 10 * tr.get_how_overloaded()

    # Aggiungiamo una piccola penalità per la bassa soddisfazione dell'utente
    for score in user_satisfaction_list:
        reward -= 10 * math.exp(-10 * score)

    return reward