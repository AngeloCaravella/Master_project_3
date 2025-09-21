import pulp
import numpy as np

class eMPC_V2G_PuLP:
    """
    Un controller MPC (Model Predictive Control) per la gestione della ricarica
    e scarica (V2G) di veicoli elettrici, implementato con la libreria PuLP.

    L'obiettivo è massimizzare il profitto totale, tenendo conto di:
    - Prezzi dell'energia dalla rete (acquisto e vendita).
    - Ricavi dalla vendita di energia all'utente finale.
    - Costi operativi come il degrado della batteria.
    - La necessità di soddisfare le richieste dell'utente (SoC alla partenza).

    Include un "fattore di aggressività" per sbilanciare la strategia
    privilegiando il profitto a discapito dei costi operativi.
    """
    def __init__(self, env, control_horizon=10, w_departure_penalty=10.0,
                 costo_degrado_kwh=0.02, prezzo_ricarica_utente_kwh=0.5,
                 fattore_aggressivita_profitto=1.0, mpc_desired_soc_factor=0.95, **kwargs): # MODIFICA: Valore di default per aggressività cambiato a 1.0
        """
        Inizializzazione del controller MPC.

        Args:
            env: L'ambiente di simulazione (es. EV2Gym).
            control_horizon (int): L'orizzonte di previsione (H) in passi di simulazione.
            w_departure_penalty (float): Peso base della penalità per mancato raggiungimento del SoC alla partenza.
            costo_degrado_kwh (float): Il costo stimato in € per ogni kWh di energia scambiato dalla batteria.
            prezzo_ricarica_utente_kwh (float): Prezzo in € che l'utente paga per ogni kWh caricato.
            fattore_aggressivita_profitto (float): Cursore per bilanciare profitto e costi operativi.
                                                   1.0 = Strategia bilanciata e realistica.
                                                   0.0 = Strategia massimamente aggressiva, ignora degrado e soddisfazione utente.
        """
        print(f"Inizializzazione controller MPC con aggressività profitto: {fattore_aggressivita_profitto}...")
        self.env = env
        self.H = control_horizon
        self.w_departure_penalty = w_departure_penalty
        self.costo_degrado_kwh = costo_degrado_kwh
        self.prezzo_ricarica_utente_kwh = prezzo_ricarica_utente_kwh
        # MODIFICA: fattore_aggressivita_profitto non è più usato per pesare i costi, ma può essere usato per altre logiche se necessario
        self.fattore_aggressivita_profitto = np.clip(fattore_aggressivita_profitto, 0.0, 1.0)
        self.mpc_desired_soc_factor = mpc_desired_soc_factor

    def get_action(self, env):
        current_step = env.current_step
        sim_length = env.simulation_length
        num_cs = env.cs
        timescale_h = env.timescale / 60.0
        transformer_limit = env.config['transformer']['max_power']

        horizon = min(self.H, sim_length - current_step)
        if horizon <= 0:
            return np.zeros(num_cs)

        prob = pulp.LpProblem(f"MPC_ProfitMaximization_Step_{current_step}", pulp.LpMaximize)

        # --- Variabili di Decisione ---
        indices = [(i, j) for i in range(num_cs) for j in range(horizon)]
        P_ch = pulp.LpVariable.dicts("ChargePower", indices, lowBound=0)
        P_dis = pulp.LpVariable.dicts("DischargePower", indices, lowBound=0)
        is_charging = pulp.LpVariable.dicts("IsCharging", indices, cat='Binary')
        E = pulp.LpVariable.dicts("Energy", indices, lowBound=0)

        # --- Funzione Obiettivo RIVISTA ---
        objective = pulp.LpAffineExpression()
        prices_charge = env.charge_prices[0, current_step : current_step + horizon]
        prices_discharge = env.discharge_prices[0, current_step : current_step + horizon]


        for t in range(horizon):
            for i in range(num_cs):
                # --- Logica di Profitto Migliorata ---

                # 1. RICAVI: Dalla vendita di energia alla rete (V2G) e all'utente
                ricavo_da_vendita_rete = prices_discharge[t] * P_dis[i, t] * timescale_h
                ricavo_da_utente = self.prezzo_ricarica_utente_kwh * P_ch[i, t] * timescale_h

                # 2. COSTI: Acquisto di energia dalla rete e degrado batteria
                costo_acquisto_rete = prices_charge[t] * P_ch[i, t] * timescale_h
                costo_degrado = self.costo_degrado_kwh * (P_ch[i, t] + P_dis[i, t]) * timescale_h

                # Il profitto netto per ogni azione è ricavi - costi
                profitto_operativo_netto = (ricavo_da_vendita_rete + ricavo_da_utente) - (costo_acquisto_rete + costo_degrado)
                
                objective += profitto_operativo_netto

        # --- Vincoli del Sistema (Invariati) ---
        for cs_id in range(num_cs):
            cs = env.charging_stations[cs_id]
            ev = next((ev for ev in cs.evs_connected if ev is not None), None)

            if ev:
                eta_ch = np.mean(list(ev.charge_efficiency.values())) if isinstance(ev.charge_efficiency, dict) else ev.charge_efficiency
                eta_dis = np.mean(list(ev.discharge_efficiency.values())) if isinstance(ev.discharge_efficiency, dict) else ev.discharge_efficiency
                if eta_dis == 0: eta_dis = 0.9

                for t in range(horizon):
                    prob += P_ch[cs_id, t] <= ev.max_ac_charge_power * is_charging[cs_id, t]
                    prob += P_dis[cs_id, t] <= abs(ev.max_discharge_power) * (1 - is_charging[cs_id, t])

                for t in range(horizon):
                    energy_change = (eta_ch * P_ch[cs_id, t] - (1/eta_dis) * P_dis[cs_id, t]) * timescale_h
                    if t == 0:
                        prob += E[cs_id, t] == ev.get_soc() * ev.battery_capacity + energy_change
                    else:
                        prob += E[cs_id, t] == E[cs_id, t-1] + energy_change

                    prob += E[cs_id, t] >= ev.min_battery_capacity
                    prob += E[cs_id, t] <= ev.battery_capacity

                departure_step_in_horizon = ev.time_of_departure - current_step - 1
                if 0 <= departure_step_in_horizon < horizon:
                    final_E = E[cs_id, departure_step_in_horizon]
                    # Vincolo "duro" (hard constraint) per la soddisfazione utente
                    prob += final_E >= ev.desired_capacity * self.mpc_desired_soc_factor, f"User_{cs_id}_Departure_SoC"
                    
                    # Logica di penalità (soft constraint) rimossa dall'obiettivo per evitare conflitti con la massimizzazione del profitto
                    # soc_deviation = pulp.LpVariable(f"SoCDev_{cs_id}", lowBound=0)
                    # prob += ev.desired_capacity - final_E <= soc_deviation
                    # objective -= self.w_departure_penalty * soc_deviation # RIMOSSO
            else:
                for t in range(horizon):
                    prob += P_ch[cs_id, t] == 0
                    prob += P_dis[cs_id, t] == 0

        for t in range(horizon):
            prob += pulp.lpSum(P_ch[i, t] - P_dis[i, t] for i in range(num_cs)) <= transformer_limit

        # --- Soluzione del Problema ---
        prob.setObjective(objective)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] == 'Optimal':
            action = np.zeros(num_cs)
            for i in range(num_cs):
                charge = pulp.value(P_ch[i, 0])
                discharge = pulp.value(P_dis[i, 0])
                net_power = (charge if charge is not None else 0) - (discharge if discharge is not None else 0)
                max_power = env.charging_stations[i].get_max_power()
                if max_power > 0:
                    action[i] = net_power / max_power
            return np.clip(action, -1, 1)
        else:
            print(f"Attenzione: Soluzione MPC non ottimale trovata ({pulp.LpStatus[prob.status]}). L'agente non eseguirà azioni.")
            return np.zeros(num_cs)
