# --- START OF FILE pulp_mpc.py ---

import pulp
import numpy as np
import joblib
import os

class OnlineMPC_Solver:
    """
    Risolve il problema di Model Predictive Control (MPC) ONLINE ad ogni step.
    Questa classe implementa una formulazione implicita (iterativa) del problema
    di ottimizzazione, che viene costruito e risolto tramite un solver MILP (PuLP)
    ad ogni chiamata di `get_action`.
    """
    def __init__(self, env, control_horizon=10, costo_degrado_kwh=0.02,
                 prezzo_ricarica_utente_kwh=0.5, mpc_desired_soc_factor=0.95, **kwargs):
        # Rimosso il print per pulire l'output durante l'addestramento
        self.env = env
        self.H = control_horizon
        self.costo_degrado_kwh = costo_degrado_kwh
        self.prezzo_ricarica_utente_kwh = prezzo_ricarica_utente_kwh
        self.mpc_desired_soc_factor = mpc_desired_soc_factor

    def get_action(self, env):
        current_step = env.current_step
        sim_length = env.simulation_length
        num_cs = env.cs
        timescale_h = env.timescale / 60.0
        transformer_limit = env.config['transformer']['max_power']

        horizon = min(self.H, sim_length - current_step)
        if horizon <= 0: return np.zeros(num_cs)

        E_initial, active_evs = np.zeros(num_cs), {}
        for i in range(num_cs):
            cs = env.charging_stations[i]
            ev = next((ev for ev in cs.evs_connected if ev is not None), None)
            if ev:
                eta_ch = np.mean(list(ev.charge_efficiency.values())) if isinstance(ev.charge_efficiency, dict) else ev.charge_efficiency
                eta_dis = np.mean(list(ev.discharge_efficiency.values())) if isinstance(ev.discharge_efficiency, dict) else ev.discharge_efficiency
                if eta_dis == 0: eta_dis = 0.9
                active_evs[i] = {'ev': ev, 'eta_ch': eta_ch, 'eta_dis': eta_dis}
                E_initial[i] = ev.get_soc() * ev.battery_capacity

        prob = pulp.LpProblem(f"Online_MPC_ProfitMax_{current_step}", pulp.LpMaximize)

        indices = [(i, j) for i in range(num_cs) for j in range(horizon)]
        P_ch = pulp.LpVariable.dicts("ChargePower", indices, lowBound=0)
        P_dis = pulp.LpVariable.dicts("DischargePower", indices, lowBound=0)
        is_charging = pulp.LpVariable.dicts("IsCharging", indices, cat='Binary')
        E = pulp.LpVariable.dicts("Energy", indices, lowBound=0)

        prices_charge = env.charge_prices[0, current_step : current_step + horizon]
        prices_discharge = env.discharge_prices[0, current_step : current_step + horizon]
        
        objective = pulp.lpSum(
            ((self.prezzo_ricarica_utente_kwh - prices_charge[t] - self.costo_degrado_kwh) * P_ch[i, t] +
             (prices_discharge[t] - self.costo_degrado_kwh) * P_dis[i, t]) * timescale_h
            for i in range(num_cs) for t in range(horizon)
        )
        prob.setObjective(objective)

        for cs_id, data in active_evs.items():
            ev, eta_ch, eta_dis = data['ev'], data['eta_ch'], data['eta_dis']
            for t in range(horizon):
                prob += P_ch[cs_id, t] <= ev.max_ac_charge_power * is_charging[cs_id, t]
                prob += P_dis[cs_id, t] <= abs(ev.max_discharge_power) * (1 - is_charging[cs_id, t])
                
                if t == 0:
                    prob += E[cs_id, t] == E_initial[cs_id] + (P_ch[cs_id, t] * eta_ch - P_dis[cs_id, t] / eta_dis) * timescale_h
                else:
                    prob += E[cs_id, t] == E[cs_id, t-1] + (P_ch[cs_id, t] * eta_ch - P_dis[cs_id, t] / eta_dis) * timescale_h

                prob += E[cs_id, t] >= ev.min_battery_capacity
                prob += E[cs_id, t] <= ev.battery_capacity

            departure_step_in_horizon = ev.time_of_departure - current_step - 1
            if 0 <= departure_step_in_horizon < horizon:
                prob += E[cs_id, departure_step_in_horizon] >= ev.desired_capacity * self.mpc_desired_soc_factor

        for i in range(num_cs):
            if i not in active_evs:
                for t in range(horizon):
                    prob += P_ch[i, t] == 0
                    prob += P_dis[i, t] == 0

        for t in range(horizon):
            prob += pulp.lpSum(P_ch[i, t] - P_dis[i, t] for i in range(num_cs)) <= transformer_limit

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] == 'Optimal':
            action = np.zeros(num_cs)
            for i in range(num_cs):
                charge = pulp.value(P_ch[i, 0])
                discharge = pulp.value(P_dis[i, 0])
                net_power = (charge or 0) - (discharge or 0)
                max_power = env.charging_stations[i].get_max_power()
                if max_power > 0: action[i] = net_power / max_power
            return np.clip(action, -1, 1)
        else:
            return np.zeros(num_cs)

class ApproximateExplicitMPC:
    """
    Implementa un controller MPC Esplicito Approssimato.
    """
    def __init__(self, env, model_path=None, control_horizon=10, max_cs=None, **kwargs):
        print(f"Inizializzazione controller MPC Esplicito Approssimato...")
        
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'mpc_approximator.joblib')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello non trovato in '{model_path}'. "
                                    "Esegui prima lo script 'train_mpc_approximator.py' per generarlo.")
        
        if max_cs is None:
            raise ValueError("Il parametro 'max_cs' (numero massimo di stazioni) deve essere fornito.")
        self.max_cs = max_cs

        self.model = joblib.load(model_path)
        self.H = control_horizon
        print(f"Modello caricato con successo da: {model_path}")

    def _build_state_vector(self, env):
        """
        Costruisce il vettore di stato con padding per garantire una dimensione fissa.
        """
        current_step = env.current_step
        horizon = min(self.H, env.simulation_length - current_step)
        num_cs_in_env = env.cs
        
        ev_socs = np.zeros(self.max_cs)
        ev_time_to_departure = np.zeros(self.max_cs)
        
        for i in range(num_cs_in_env):
            ev = next((ev for ev in env.charging_stations[i].evs_connected if ev is not None), None)
            if ev:
                ev_socs[i] = ev.get_soc()
                ev_time_to_departure[i] = max(0, ev.time_of_departure - current_step)

        prices_charge = env.charge_prices[0, current_step : current_step + horizon]
        prices_discharge = env.discharge_prices[0, current_step : current_step + horizon]
        
        padded_prices_ch = np.pad(prices_charge, (0, self.H - len(prices_charge)), 'edge')
        padded_prices_dis = np.pad(prices_discharge, (0, self.H - len(prices_discharge)), 'edge')

        state_vector = np.concatenate([
            ev_socs,
            ev_time_to_departure,
            padded_prices_ch,
            padded_prices_dis
        ])
        return state_vector.reshape(1, -1)

    def get_action(self, env):
        if env.current_step >= env.simulation_length - 1:
            return np.zeros(env.cs)
            
        state_vector = self._build_state_vector(env)
        
        predicted_powers = self.model.predict(state_vector)[0]
        
        action = np.zeros(env.cs)
        for i in range(env.cs):
            max_power = env.charging_stations[i].get_max_power()
            if max_power > 0:
                action[i] = predicted_powers[i] / max_power

        return np.clip(action, -1, 1)
