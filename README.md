# EV2Gym Simulation Orchestrator

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)

Un framework completo per la simulazione dell'interazione tra veicoli elettrici (EV) e la rete elettrica, con particolare attenzione alla ricarica intelligente, agli agenti di Reinforcement Learning (RL) e alle baseline di Model Predictive Control (MPC). Il progetto include ora un'interfaccia grafica (GUI) basata su Streamlit per una configurazione e un'esecuzione delle simulazioni più intuitive e visuali.

## Funzionalità Principali

-   **Simulazione EV-Grid**: Ambiente di simulazione dettagliato per la gestione della ricarica EV e l'interazione con la rete.
-   **Agenti RL**: Implementazione e addestramento di vari algoritmi di Reinforcement Learning (es. SAC, DDPG+PER, TQC) per la gestione ottimale della ricarica.
-   **Baseline MPC**: Confronto con strategie di controllo basate su Model Predictive Control (MPC) e euristiche (AFAP, ALAP, RR).
-   **Analisi Dati**: Strumenti per l'analisi dei file di configurazione e il fitting di modelli di degradazione della batteria.
-   **Interfaccia Grafica Streamlit**: GUI intuitiva per configurare ed eseguire le simulazioni, con output in tempo reale e visualizzatore di risultati.
-   **Visualizzazione Risultati**: Generazione automatica di grafici di performance e degradazione della batteria.

## Installazione

Per configurare il progetto in locale, segui questi passaggi:

1.  **Clona il repository** (se non l'hai già fatto):
    ```bash
    git clone https://github.com/AngeloCaravella/EV2Gym-Orchestrator.git # Sostituisci con il tuo URL del repository
    cd EV2Gym-Orchestrator
    ```

2.  **Crea e attiva un ambiente virtuale** (raccomandato):
    ```bash
    python -m venv venv
    # Su Windows
    .\venv\Scripts\activate
    # Su macOS/Linux
    source venv/bin/activate
    ```

3.  **Installa le dipendenze Python**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Installa Graphviz** (necessario per la generazione di diagrammi UML con `pyreverse`):
    *   Scarica e installa Graphviz dal sito ufficiale: [https://graphviz.org/download/](https://graphviz.org/download/)
    *   **Importante**: Assicurati di aggiungerlo al PATH di sistema durante l'installazione.

## Utilizzo

Il progetto può essere utilizzato tramite riga di comando (CLI) o tramite l'interfaccia grafica Streamlit.

### 1. Interfaccia Grafica (GUI) Streamlit

Questo è il metodo raccomandato per configurare ed eseguire le simulazioni in modo visuale.

1.  **Avvia l'applicazione Streamlit**:
    ```bash
    streamlit run streamlit_app.py
    ```
    In alternativa, puoi fare doppio clic sul file `launch_streamlit.bat` (solo su Windows).

2.  **Configura la simulazione**: Utilizza i controlli nell'interfaccia per selezionare scenari, algoritmi, funzioni di reward, file dei prezzi e altre opzioni.

3.  **Esegui la simulazione**: Clicca sul pulsante "Esegui Simulazione". L'output della console verrà mostrato in tempo reale nell'app.

4.  **Visualizza i risultati**: Dopo aver eseguito le simulazioni, puoi usare la sezione "Visualizzatore Risultati" per navigare tra le cartelle di benchmark e visualizzare i grafici generati.

### 2. Riga di Comando (CLI)

Per un controllo più granulare o per l'automazione, puoi eseguire lo script principale direttamente da riga di comando.

1.  **Esegui lo script principale**:
    ```bash
    python run_experiments.py --help
    ```
    Questo mostrerà tutte le opzioni disponibili. Ecco alcuni esempi:

    ```bash
    # Esegui un benchmark con scenari specifici e addestra i modelli RL
    python run_experiments.py --plot_mode thesis --scenarios V2GProfitMax_Het --reward_func SquaredTrackingErrorReward --train_rl_models --steps_for_training 50000 --num_sims 5

    # Esegui solo il benchmark per tutti gli scenari con il file prezzi di default
    python run_experiments.py --scenarios all --num_sims 1

    # Esegui Fit_battery.py prima del benchmark
    python run_experiments.py --run_fit_battery --scenarios all
    ```

### 3. Analisi dei File di Configurazione

Per generare tabelle riassuntive dei file di configurazione:

```bash
python Compare.py
```

### 4. Fitting del Modello di Degradazione Batteria

Per eseguire il fitting del modello di degradazione della batteria (genera un grafico):

```bash
python Fit_battery.py
```

## Struttura del Progetto

```
. # Root del progetto
├── ev2gym/
│   ├── baselines/          # Implementazioni di baseline (euristiche, MPC)
│   ├── data/               # File di dati (prezzi, profili di carico, specifiche EV)
│   ├── models/             # Modelli dell'ambiente (EV, caricatore, grid, ambiente Gym)
│   ├── rl_agent/           # Agenti RL e componenti correlati
│   ├── utilities/          # Funzioni di utilità e loader di dati
│   └── visuals/            # Funzioni per la visualizzazione
├── results/                # Cartella per i risultati delle simulazioni (grafici, dati)
├── saved_models/           # Cartella per i modelli RL addestrati
├── streamlit_app.py        # Interfaccia grafica Streamlit
├── run_experiments.py      # Script principale per l'esecuzione di esperimenti e benchmark
├── Compare.py              # Script per l'analisi comparativa dei file di configurazione
├── Fit_battery.py          # Script per il fitting del modello di degradazione della batteria
├── requirements.txt        # Dipendenze Python del progetto
├── launch_streamlit.bat    # Script per avviare l'app Streamlit (Windows)
└── README.md               # Questo file
```

## Dipendenze

Le dipendenze Python sono elencate in `requirements.txt`. Assicurati di installarle in un ambiente virtuale.

## Autore

Sviluppato da: **Angelo Caravella**

## Licenza

Questo progetto è rilasciato sotto licenza MIT. Vedi il file `LICENSE` per maggiori dettagli. (Crea un file `LICENSE` se non esiste)
