import pulp
import pickle
import numpy as np

class V2GProfitMaxOraclePuLP:
    def __init__(self, env, **kwargs):
        print("Inizializzazione dell'Oracolo Ottimale con PuLP/CBC...")
        self.env = env
        self.config = env.config
        self.replay_path = env.load_from_replay_path
        self.actions = self._solve_optimal_plan()

    def get_action(self, env):
        current_step = env.current_step
        if current_step < len(self.actions):
            return self.actions[current_step]
        else:
            return np.zeros(self.config['number_of_charging_stations'])

    def _solve_optimal_plan(self):
        with open(self.replay_path, 'rb') as f:
            replay_data = pickle.load(f)

        ev_schedules = {}
        if hasattr(replay_data, 'EVs'):
            for ev in replay_data.EVs:
                if ev.location not in ev_schedules:
                    ev_schedules[ev.location] = []
                ev_schedules[ev.location].append({
                    'arrival_time': ev.time_of_arrival,
                    'departure_time': ev.time_of_departure,
                    'max_ac_charge_power': ev.max_ac_charge_power,
                    'max_discharge_power': ev.max_discharge_power,
                    'arrival_soc': ev.battery_capacity_at_arrival / ev.battery_capacity if ev.battery_capacity > 0 else 0,
                    'desired_soc': ev.desired_capacity / ev.battery_capacity if ev.battery_capacity > 0 else 0,
                    'battery_capacity': ev.battery_capacity,
                })

        prices = replay_data.charge_prices[0]
        sim_length = self.config['simulation_length']
        num_cs = self.config['number_of_charging_stations']
        timescale_h = self.config['timescale'] / 3600.0
        transformer_limit = self.config['transformer']['max_power']

        prob = pulp.LpProblem("V2G_Optimal_Control", pulp.LpMaximize)

        power_indices = [(cs, t) for cs in range(num_cs) for t in range(sim_length)]
        power_vars = pulp.LpVariable.dicts("Power", power_indices, lowBound=-100, upBound=100, cat='Continuous')

        prob += pulp.lpSum(power_vars[cs_id, t] * -prices[t] * timescale_h for cs_id in range(num_cs) for t in range(sim_length))

        for t in range(sim_length):
            prob += pulp.lpSum(power_vars[cs_id, t] for cs_id in range(num_cs)) <= transformer_limit

        for cs_id, ev_list in ev_schedules.items():
            for ev in ev_list:
                arrival, departure = ev['arrival_time'], ev['departure_time']
                if arrival >= departure:
                    continue

                for t in range(arrival, departure):
                    prob += power_vars[cs_id, t] <= ev['max_ac_charge_power']
                    prob += power_vars[cs_id, t] >= -ev['max_discharge_power']

                initial_soc_kwh = ev['arrival_soc'] * ev['battery_capacity']
                target_soc_kwh = ev['desired_soc'] * ev['battery_capacity']
                net_energy_exchange = pulp.lpSum(power_vars[cs_id, t] * timescale_h for t in range(arrival, departure))

                # Rilassiamo il vincolo: l'energia finale deve essere ALMENO quella desiderata
                prob += initial_soc_kwh + net_energy_exchange >= target_soc_kwh
                
                # L'energia finale non può superare la capacità della batteria
                prob += initial_soc_kwh + net_energy_exchange <= ev['battery_capacity']

        print("Scrittura del file LP per il debug: oracle_problem.lp")
        prob.writeLP("oracle_problem.lp")

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] == 'Optimal':
            print("Soluzione ottimale trovata!")
            optimal_actions = np.zeros((sim_length, num_cs))
            for t in range(sim_length):
                for cs_id in range(num_cs):
                    if (cs_id, t) in power_vars:
                        optimal_actions[t, cs_id] = pulp.value(power_vars[cs_id, t]) if power_vars[cs_id, t].varValue is not None else 0
            return optimal_actions
        else:
            print(f"ATTENZIONE: Nessuna soluzione ottimale trovata. Stato: {pulp.LpStatus[prob.status]}")
            return np.zeros((sim_length, num_cs))