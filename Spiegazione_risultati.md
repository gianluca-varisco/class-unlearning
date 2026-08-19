# FUIA: Class Unlearning Scenario & Parameter Sensitivity Analysis

Questo modulo estende la valutazione dell'attacco **FUIA (Federated Unlearning Inversion Attack)**, proposto da Zhou et al. (*IEEE TIFS 2026*), implementando e validando formalmente lo scenario di **Class Unlearning** (Algoritmo 3 del paper).

---

## 🎯 Obiettivo dell'Attacco

Nello scenario di Class Unlearning, l'obiettivo dell'attaccante è inferire quale etichetta di classe ($c_{target}$) è stata rimossa dal modello globale durante il processo di Federated Unlearning.

L'attacco confronta i parametri dell'ultimo strato lineare di classificazione ($W^o$ e $W^u$) calcolando lo **Score di Discriminazione $S_d[i]$** per ciascuna classe $i \in \{0, \dots, S-1\}$:

$$v_{\mathrm{diff}}[i] = \|v_o[i] - v_u[i]\|_1, \quad b_{\mathrm{diff}}[i] = |b_o[i] - b_u[i]|$$

$$S_d[i] = \beta \cdot \frac{v_{\mathrm{diff}}[i]}{\sum_{j=0}^{S-1} v_{\mathrm{diff}}[j]} + (1 - \beta) \cdot \frac{b_{\mathrm{diff}}[i]}{\sum_{j=0}^{S-1} b_{\mathrm{diff}}[j]}$$

$$\mathrm{class\_id} = \arg\max_i (S_d[i])$$

---

## 📊 Risultati Sperimentali e Studio di Ablazione

Di seguito è riportato lo studio di sensitività parametrica condotto su dataset **MNIST** al variare delle epoche di pre-training centralizzato, dei campioni per client e dei round di addestramento federato:

| Scenario Sperimentale | `PRETRAIN_EPOCHS` | `DATA_PER_CLIENT` | `NUM_ROUNDS` | Acc. $W^o$ | Acc. $W^u$ | Esito Attacco | Score $S_d$ Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Pre-training pesante** | 5 | 1 | 80 | 99.2% | 99.2% | **FAILED** | $S_d[3] = 0.1752$ (pred. 5) |
| **2. Pre-training minimo** | 1 | 1 | 40 | 98.1% | 98.1% | **FAILED** | $S_d[3] = 0.1179$ (pred. 7) |
| **3. Pochi dati (No pre-train)** | 0 | 1 | 20 / 50 | 39.3% / 56.0% | ~56.0% | **SUCCESS** | $S_d[3] = 0.3819$ / $0.3637$ |
| **4. Setting Ottimale (Target: 3)** | 0 | 100 | 50 | **96.2%** | **86.0%** | **SUCCESS** | $S_d[3] = \mathbf{0.2940}$ |
| **5. Setting Ottimale (Target: 7)** | 0 | 100 | 50 | **96.2%** | **85.8%** | **SUCCESS** | $S_d[7] = \mathbf{0.3261}$ |

---

## 🔍 Analisi Metodologica dei Risultati

1. **Effetto Distorsivo del Pre-training Centralizzato (`PRETRAIN_EPOCHS > 0`):**  
   Pre-addestrare centralmente il modello sull'intero dataset fissa i pesi dell'ultimo layer di classificazione prima dell'avvio del Federated Learning ($>98\%$ accuracy). I successivi round di retraining ($W^u$) inducono modifiche minime, riducendo la discrepanza $\Delta W$ a rumore numerico e causando predizioni errate (falsi positivi).
   
2. **Volume dei Dati per Client (`DATA_PER_CLIENT`):**  
   Con `DATA_PER_CLIENT = 1` e nessun pre-training, l'attacco ha successo ma il modello globale raggiunge solo il $56.0\%$ di accuratezza a causa del ridotto volume di campioni complessivi (50 campioni totali). Incrementando a `DATA_PER_CLIENT = 100` (5.000 campioni federati), il modello converge al **96.2%** di accuratezza mantenendo la piena efficacia dell'attacco.

3. **Verifica Analitica dell'Unlearning (Calo del 10% dell'Accuratezza):**  
   Il calo dal **96.2%** ($W^o$) all'**85.8% - 86.0%** ($W^u$) sul test set completo a 10 classi corrisponde esattamente alla perdita teorica attesa di $\frac{1}{10}$ delle classi, attestando che la classe target è stata effettivamente rimossa.

4. **Robustezza tra Classi Differenti:**  
   L'attacco ha raggiunto esito positivo sia sulla cifra **3** ($S_d = 0.2940$) che sulla cifra **7** ($S_d = 0.3261$) a fronte di un rumore di fondo medio sulle classi residue compreso tra $0.06$ e $0.09$, confermando l'indipendenza dell'Algoritmo 3 dalla specifica etichetta target.

---

## 🚀 Come Riprodurre gli Esperimenti

### Requisiti e Setup
```bash
git clone <URL_REPOSITORY>
cd FUIA_Class_Unlearning
python3 -m venv env
source env/bin/activate  # Su Windows: env\Scripts\activate
pip install -r requirements.txt