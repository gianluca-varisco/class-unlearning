# Multi-Class Federated Unlearning via Class-Discriminative Pruning (FedPrune) & FUIA Vulnerability Analysis

Questo documento descrive l'estensione dell'algoritmo **FedPrune / Class-Discriminative Pruning (CDP)** (Wang et al., *ACM WWW 2022*) allo scenario di **Multi-Class Unlearning** ($N_{\mathrm{uc}} \in \{2, 3, 4\}$) su dataset MNIST, e la sua valutazione contro l'attacco di inferenza **FUIA (Top-k Class Discrimination)** formalizzato da Zhou et al. (*IEEE TIFS 2026*).

---

## 1. Motivazione Scientifica: Dalla Classe Singola al Multi-Class Pruning

Nello studio del **Class Unlearning**, la richiesta di oblio può coinvolgere contemporaneamente un insieme di $k$ categorie concettuali distinte ($C_{\mathrm{target}} = [c_1, c_2, \dots, c_k]$):
* **Nel Retraining da Zero:** Rimuovere più classi richiede il ricalcolo integrale dei round federati (50–80 round) a partire da $W_{\mathrm{init}}$, comportando un onere di calcolo e di comunicazione significativo.
* **Nel FedPrune Multi-Class:** L'obiettivo è rimuovere l'influenza congiunta dell'insieme $C_{\mathrm{target}}$ direttamente dal modello globale convergente $W^o$ tramite potatura mirata cumulativa e pochissimi round di recupero federato (fine-tuning di 3 round), riducendo i tempi di esecuzione da svariati minuti a pochi secondi.

L'esperimento si propone di rispondere a una domanda teorica chiave:
> *"L'attacco FUIA Top-k di Zhou et al. mantiene la capacità di discriminare l'intero gruppo di classi rimosse quando l'unlearning avviene tramite potatura selettiva congiunta anziché retraining?"*

---

## 2. Metodologia di FedPrune Multi-Class (Estensione da Wang et al., 2022)

L'adattamento dell'algoritmo per supportare $|C_{\mathrm{target}}| \ge 2$ si articola in tre fasi operative:

### 2.1 Calcolo della Sensibilità Convoluzionale Congiunta
Per individuare i filtri convoluzionali responsabili delle classi da eliminare, si valuta la risposta media dei canali al passaggio dell'unione di tutti i campioni appartenenti alle classi target ($x \in \bigcup_{c \in C_{\mathrm{target}}} D_c$):
$$CS_{k, C_{\mathrm{target}}} = \frac{1}{\sum_{c \in C_{\mathrm{target}}} |D_c|} \sum_{x \in \bigcup D_c} \text{mean}(a_k(x))$$

Nel modello a 2 stadi convoluzionali (`features[0]` con 32 filtri e `features[3]` con 64 filtri):
1. Si registrano i tensori di output tramite *forward hook* su entrambe le sezioni convoluzionali.
2. Si mediano spazialmente i canali su altezza, larghezza e batch per estrarre il vettore di sensibilità cumulativa.

### 2.2 Pruning Selettivo Multi-Classe e Azzeramento del Classificatore
Definito il rapporto di potatura $\rho$ (`prune_ratio = 0.3`):
1. **Canali Convoluzionali:** Si selezionano i primi $\lfloor K \cdot \rho \rfloor$ filtri con la sensibilità congiunta più elevata sia nel blocco `features[0]` sia in `features[3]` e se ne azzerano i pesi e i bias corrispondenti:
   $$W_{\text{features}}[idx] \leftarrow 0, \quad b_{\text{features}}[idx] \leftarrow 0$$
2. **Azzeramento Multiplo dell'Ultimo Layer:** A differenza del caso a classe singola, le righe della matrice di proiezione lineare `classifier[2]` e i rispettivi bias vengono azzerati per **tutte** le classi presenti in $C_{\mathrm{target}}$:
   $$\forall c \in C_{\mathrm{target}}: \quad W_{\text{classifier}}[c] \leftarrow 0, \quad b_{\text{classifier}}[c] \leftarrow 0$$
   Ciò azzera immediatamente i logit e la confidenza probabilistica di tutte le classi bersaglio.

### 2.3 Fine-Tuning Federato Multi-Classe (Federated Recovery)
La disattivazione congiunta dei filtri condivisi provoca un impatto prestazionale sulle rimanenti classi $C \setminus C_{\mathrm{target}}$:
* Viene avviata una sequenza federata breve di **3 round** con learning rate attenuato (`FL_LR * 0.5`).
* I client si addestrano unicamente sui campioni delle classi da preservare (tutti i dati con etichetta $y \in C_{\mathrm{target}}$ vengono filtrati via).
* Ad ogni round di aggregazione globale FedAvg, viene riapplicata la maschera di zeri sui neuroni del classificatore appartenenti a $C_{\mathrm{target}}$, congelando tali classi allo stato disimparato.

---

## 3. L'Attacco FUIA Multi-Class (Zhou et al., 2026, Top-k Ranking)

Per rilevare le classi disimparate, l'attaccante esegue l'**Algoritmo 3** generalizzato a Top-k confrontando il modello originale $W^o$ con il modello potato $W^u$:

1. **Variazione dei Pesi e dei Bias (Norma $L_1$):**
   $$\Delta w_i = \|w_o[i] - w_u[i]\|_1, \quad \Delta b_i = |b_o[i] - b_u[i]| \quad \forall i \in \{0, \dots, 9\}$$
2. **Score di Discriminazione Normalizzato $S_d[i]$:**
   $$S_d[i] = \beta \frac{\Delta w_i}{\sum_j \Delta w_j} + (1 - \beta) \frac{\Delta b_i}{\sum_j \Delta b_j} \quad (\beta = 0.5)$$
3. **Predizione Top-k:**
   Si ordinano gli indici in ordine decrescente di punteggio $S_d[i]$ e si estraggono i primi $k = |C_{\mathrm{target}}|$ elementi:
   $$\hat{C}_{\mathrm{target}} = \text{Top-}k(S_d)$$

---

## 4. Confronto Comparativo: Retraining vs. FedPrune Multi-Class

| Proprietà | Multi-Class Retraining (Exact FU) | Multi-Class FedPrune (Approximate FU) |
| :--- | :--- | :--- |
| **Punto di Partenza** | Inizializzazione stocastica casuale $W_{\text{init}}$ | Pesi convergenti del modello globale $W^o$ |
| **Numero di Round FU** | 50 – 80 round completi | **3 round** di fine-tuning compatto |
| **Tempo di Esecuzione** | Elevato (nell'ordine dei minuti) | **Pochi secondi** |
| **Comportamento Pesi Target** | Rimangono allo stato di inizializzazione casuale | Vengono esplicitamente azzerati e congelati |
| **Meccanismo di Divergenza** | Fluttuazione e mancata convergenza su $C_{\mathrm{target}}$ | $\Delta w_c = \|w_o[c] - 0\|_1 = \|w_o[c]\|_1$ per tutti i $c \in C_{\mathrm{target}}$ |
| **Comportamento dello Score $S_d$** | Margini soggetti a *backbone drift* ($N_{\mathrm{uc}} = 4$ degrada) | Picchi distinti per ciascuna classe target azzerata |

### Considerazioni sulla Resilienza dell'Attacco
Nel Retraining a 4 classi, la variazione dell'estrattore convoluzionale dovuta alla perdita del 40% dei campioni federati provocava un drifting parziale dei pesi, portando a esiti parziali (3/4 successi). 
In **FedPrune**, poiché l'azzeramento della testa del classificatore è deterministicamente imposto per tutte le classi rimosse ($\|w_o[c] - w_u[c]\|_1 \approx \|w_o[c]\|_1$), le classi target mantengono uno scostamento rispetto alle classi non rimosse che continuano a preservare la loro configurazione ottimizzata, confermando la piena suscettibilità di FedPrune all'attacco di identificazione FUIA anche in scenari multi-classe.

