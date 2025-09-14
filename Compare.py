import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

# --- COSTANTI PER LE INTESTAZIONI ---
# Tabella 1: Infrastruttura e Rete
COL_FILE = "File"
COL_SCENARIO = "Scenario"
COL_DURATA_SIM = "Durata\n(ore)"
COL_STAZIONI = "Stazioni"
COL_POT_RETE = "Pot. Rete\n(kW)"
COL_CAPACITA_NETTA = "Cap. Netta\n(kW)"
COL_DOMANDA_MAX = "Dom. Max\n(kW)"
COL_STRESS_RETE = "Stress\nRete"
COL_TRAFFICO = "Traffico\nEV"

# Tabella 2: Dettagli EV
COL_V2G = "V2G"
COL_V2G_FACTOR = "Fattore\nPrezzo V2G"
COL_EFFICIENCY = "Efficienza\n(C/D)"
COL_EV_DIVERSI = "EV\nDiversi"
COL_BATT_DEFAULT = "Batt. Def.\n(kWh)"
COL_BATT_EMERGENZA = "Cap. Emergenza\n(kWh)"
COL_BATT_DESIDERATA = "Cap. Desiderata\n(%)"
COL_SOSTA_MINIMA = "Sosta Min.\n(min)"


def get_nested_val(data: Dict[str, Any], keys: List[str], default: Any) -> Any:
    """Accede in modo sicuro a un valore nidificato in un dizionario."""
    temp_dict = data
    for key in keys:
        if isinstance(temp_dict, dict) and key in temp_dict:
            temp_dict = temp_dict[key]
        else:
            return default
    return temp_dict


def _process_single_config(file_path: str, file_name: str) -> Optional[Dict[str, Any]]:
    """Estrae e calcola TUTTI i dati da un singolo file di configurazione."""
    print(f"  -> Analizzo il file: {file_name}")
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"ATTENZIONE: Impossibile processare {file_name}. Errore: {e}. Saltato.")
        return None

    # --- DATI PER TABELLA 1: INFRASTRUTTURA E RETE ---
    num_stations = get_nested_val(config, ['number_of_charging_stations'], 0)
    transformer_power = get_nested_val(config, ['transformer', 'max_power'], 0)
    
    station_config = config.get('charging_station', {})
    max_current = station_config.get('max_charge_current', 0)
    voltage = station_config.get('voltage', 0)
    phases = station_config.get('phases', 0)
    max_station_power_kw = (voltage * max_current * (np.sqrt(3) if phases == 3 else 1)) / 1000

    total_potential_demand_kw = num_stations * max_station_power_kw
    capacity_ratio = total_potential_demand_kw / transformer_power if transformer_power > 0 else float('inf')

    timescale = get_nested_val(config, ['timescale'], 15)
    sim_length = get_nested_val(config, ['simulation_length'], 1)
    sim_duration_hours = (sim_length * timescale) / 60

    loads_power = transformer_power * get_nested_val(config, ['inflexible_loads', 'inflexible_loads_capacity_multiplier_mean'], 0)
    solar_power = transformer_power * get_nested_val(config, ['solar_power', 'solar_power_capacity_multiplier_mean'], 0)
    net_capacity = transformer_power - loads_power + solar_power

    # --- DATI PER TABELLA 2: DETTAGLI EV ---
    ev_config = config.get('ev', {})
    charge_eff = get_nested_val(ev_config, ['charge_efficiency'], 1.0)
    discharge_eff = get_nested_val(ev_config, ['discharge_efficiency'], 1.0)
    
    return {
        COL_FILE: file_name.replace('.yaml', ''),
        COL_SCENARIO: str(config.get('scenario', 'N/D')).title(),
        COL_DURATA_SIM: f"{sim_duration_hours:.1f}",
        COL_STAZIONI: num_stations,
        COL_POT_RETE: transformer_power,
        COL_CAPACITA_NETTA: f"{net_capacity:.1f}",
        COL_DOMANDA_MAX: f"{total_potential_demand_kw:.1f}",
        COL_STRESS_RETE: f"{capacity_ratio:.2f}",
        COL_TRAFFICO: config.get('spawn_multiplier', 0),
        
        COL_V2G: "SI" if config.get('v2g_enabled', False) else "NO",
        COL_V2G_FACTOR: get_nested_val(config, ['discharge_price_factor'], 'N/A'),
        COL_EFFICIENCY: f"{charge_eff}/{discharge_eff}",
        COL_EV_DIVERSI: "SI" if get_nested_val(config, ['heterogeneous_ev_specs'], True) else "NO",
        COL_BATT_DEFAULT: get_nested_val(ev_config, ['battery_capacity'], 'N/A'),
        COL_BATT_EMERGENZA: get_nested_val(ev_config, ['min_emergency_battery_capacity'], 'N/A'),
        COL_BATT_DESIDERATA: f"{get_nested_val(ev_config, ['desired_capacity'], 1.0) * 100:.0f}",
        COL_SOSTA_MINIMA: get_nested_val(ev_config, ['min_time_of_stay'], 'N/A'),
    }

def plot_grid_summary_table(extracted_data: List[Dict[str, Any]]):
    """Genera la tabella di sintesi dell'infrastruttura e della rete."""
    cols_to_show = [COL_FILE, COL_SCENARIO, COL_DURATA_SIM, COL_STAZIONI, COL_POT_RETE, COL_CAPACITA_NETTA, COL_DOMANDA_MAX, COL_STRESS_RETE, COL_TRAFFICO]
    cell_data = [[str(row[col]) for col in cols_to_show] for row in extracted_data]
    
    fig, ax = plt.subplots(figsize=(len(cols_to_show) * 2.2, len(extracted_data) * 0.5 + 1.5))
    ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=cell_data, colLabels=cols_to_show, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.0, 2.0)

    for (i, j), cell in table.get_celld().items():
        if i == 0: cell.set_text_props(weight='bold', color='white'); cell.set_facecolor('#2c3e50')
        else:
            cell.set_facecolor('#ecf0f1' if i % 2 == 1 else 'white')
            if cols_to_show[j] == COL_CAPACITA_NETTA: cell.set_facecolor('#d9e2ec')
            if cols_to_show[j] == COL_STRESS_RETE:
                try:
                    ratio = float(cell.get_text().get_text())
                    if ratio > 1.05: cell.set_facecolor('#e74c3c'); cell.set_text_props(weight='bold', color='white')
                    elif ratio > 0.95: cell.set_facecolor('#f39c12')
                    else: cell.set_facecolor('#2ecc71')
                except (ValueError, TypeError): pass

    fig.suptitle('Sintesi Infrastruttura e Rete', fontsize=22, y=0.98)
    plt.savefig("Grid_Summary.png", dpi=200, bbox_inches='tight', pad_inches=0.7)
    print("\nTabella 'Grid_Summary.png' salvata con successo.")
    plt.close(fig)

def plot_ev_details_table(extracted_data: List[Dict[str, Any]]):
    """Genera la tabella con i dettagli fisici ed economici degli EV."""
    cols_to_show = [COL_FILE, COL_V2G, COL_V2G_FACTOR, COL_EFFICIENCY, COL_EV_DIVERSI, COL_BATT_DEFAULT, COL_BATT_EMERGENZA, COL_BATT_DESIDERATA, COL_SOSTA_MINIMA]
    cell_data = [[str(row[col]) for col in cols_to_show] for row in extracted_data]
    
    fig, ax = plt.subplots(figsize=(len(cols_to_show) * 2.2, len(extracted_data) * 0.5 + 1.5))
    ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=cell_data, colLabels=cols_to_show, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.0, 2.0)

    for (i, j), cell in table.get_celld().items():
        if i == 0: cell.set_text_props(weight='bold', color='white'); cell.set_facecolor('#34495e')
        else:
            cell.set_facecolor('#ecf0f1' if i % 2 == 1 else 'white')
            # Evidenzia parametri irrealistici
            if cols_to_show[j] == COL_EFFICIENCY and cell.get_text().get_text() == "1.0/1.0":
                cell.set_facecolor('#f39c12')
            if cols_to_show[j] == COL_V2G_FACTOR and cell.get_text().get_text() == "1":
                cell.set_facecolor('#f39c12')

    fig.suptitle('Dettagli Fisici ed Economici EV', fontsize=22, y=0.98)
    plt.savefig("EV_Details.png", dpi=200, bbox_inches='tight', pad_inches=0.7)
    print("Tabella 'EV_Details.png' salvata con successo.")
    plt.close(fig)
    
def create_legends():
    """Crea le due immagini di legenda separate."""
    # Legenda 1: Rete
    fig1, ax1 = plt.subplots(figsize=(16, 9)); ax1.axis('off')
    grid_text = (
        'GUIDA ALLA LETTURA: SINTESI INFRASTRUTTURA E RETE\n\n'
        '• Pot. Rete (kW): Limite fisico del trasformatore (capacità massima assoluta).\n'
        '• Cap. Netta (kW): Potenza *realmente disponibile* per gli EV. Formula: `Pot. Rete - Carichi + Solare`.\n'
        '• Dom. Max (kW): Domanda teorica se tutti gli EV caricassero alla massima potenza.\n\n'
        '--- INDICATORE CHIAVE: STRESS RETE ---\n'
        'Misura quanto la domanda potenziale di picco si avvicina (o supera) la capacità totale della rete.\n'
        'Formula di calcolo:   Stress Rete = Domanda Max (kW) / Pot. Rete (kW)\n\n'
        'Significato dei colori:\n'
        '    ■ Verde (< 0.95): SICURO. La rete ha ampio margine.\n'
        '    ■ Arancione (0.95-1.05): ATTENZIONE. Rete al limite. Controllo intelligente indispensabile.\n'
        '    ■ Rosso (> 1.05): CRITICO. Rete sottodimensionata. Sovraccarico probabile.'
    )
    fig1.suptitle('Guida all\'Interpretazione: Rete', fontsize=22, y=0.95)
    fig1.text(0.5, 0.5, grid_text, ha='center', va='center', fontsize=13.5, bbox={'facecolor': '#f8f9fa', 'edgecolor': 'gray', 'boxstyle': 'round,pad=1.5'})
    plt.savefig("Legend_Grid.png", dpi=200, bbox_inches='tight', pad_inches=0.7)
    print("Legenda 'Legend_Grid.png' salvata con successo.")
    plt.close(fig1)

    # Legenda 2: EV
    fig2, ax2 = plt.subplots(figsize=(16, 9)); ax2.axis('off')
    ev_text = (
        'GUIDA ALLA LETTURA: DETTAGLI FISICI ED ECONOMICI EV\n\n'
        '• Fattore Prezzo V2G: Rapporto tra prezzo di vendita (scarica) e acquisto (carica).\n'
        '• Efficienza (C/D): Efficienza di Carica / Scarica. `1.0/1.0` indica nessuna perdita (irrealistico).\n'
        '• EV Diversi: `SI` = scenario realistico con un mix di EV (batterie/potenze diverse).\n'
        '• Batt. Def. (kWh): Capacità della batteria per gli EV di default (se non diversi).\n'
        '• Cap. Emergenza (kWh): Riserva minima intoccabile della batteria per l\'utente.\n'
        '• Cap. Desiderata (%): Obiettivo di ricarica che l\'EV deve raggiungere prima della partenza.\n'
        '• Sosta Min. (min): Durata minima della sosta per ogni EV.\n\n'
        '--- SIGNIFICATO DEI COLORI ---\n'
        '    ■ Arancione: Evidenzia parametri fisicamente o economicamente irrealistici (es. efficienza 100%).'
    )
    fig2.suptitle('Guida all\'Interpretazione: Dettagli EV', fontsize=22, y=0.95)
    fig2.text(0.5, 0.5, ev_text, ha='center', va='center', fontsize=13.5, bbox={'facecolor': '#f8f9fa', 'edgecolor': 'gray', 'boxstyle': 'round,pad=1.5'})
    plt.savefig("Legend_EV_Details.png", dpi=200, bbox_inches='tight', pad_inches=0.7)
    print("Legenda 'Legend_EV_Details.png' salvata con successo.")
    plt.close(fig2)


def analyze_configs(config_directory: str, file_list: List[str]):
    """Funzione principale che orchestra l'analisi e la creazione delle immagini."""
    print("Avvio analisi dettagliata dei file di configurazione...")
    extracted_data = [
        data for file_name in file_list
        if (data := _process_single_config(os.path.join(config_directory, file_name), file_name)) is not None
    ]
    
    if not extracted_data:
        print("\nNessun dato valido estratto. Impossibile generare le immagini.")
        return

    plot_grid_summary_table(extracted_data)
    plot_ev_details_table(extracted_data)
    create_legends()
    
    print("\nAnalisi completata con successo!")


if __name__ == "__main__":
    CONFIG_DIR = "ev2gym/example_config_files"
    FILES_TO_ANALYZE = [
        "BusinessPST.yaml", "PublicPST.yaml", "simplePST.yaml",
        "V2GProfitMax.yaml", "V2GProfitPlusLoads.yaml",
        "V2GProfitMax_1ev.yaml", "V2GProfitPlusLoads_1ev.yaml"
    ]
    
    if not os.path.isdir(CONFIG_DIR):
        print(f"ERRORE: La directory '{CONFIG_DIR}' non è stata trovata.")
    else:
        analyze_configs(CONFIG_DIR, FILES_TO_ANALYZE)
