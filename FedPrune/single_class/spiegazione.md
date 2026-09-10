# Federated Unlearning via Class-Discriminative Pruning (FedPrune): Implementazione Single-Class e Analisi FUIA

Questo documento descrive l'implementazione e la validazione sperimentale del metodo di unlearning approssimato **FedPrune / Class-Discriminative Pruning (CDP)**, introdotto da Wang et al. (*ACM WWW 2022*), applicato allo scenario **Single-Class Unlearning** su dataset MNIST, e la sua valutazione contro l'attacco di inferenza **FUIA (Federated Unlearning Inversion Attack)** formalizzato da Zhou et al. (*IEEE TIFS 2026*).

---

## 1. Motivazione Scientifica: Oltre il Retraining

Nel Federated Unlearning esatto (*Exact Unlearning*), la rimozione di una classe target $c$ viene realizzata rieseguendo l'intero processo di federazione da zero (*Retraining*), escludendo i dati della classe target fin dal primo round[cite: 2]. Sebbene garantisca la rimozione dei gradienti relativi alla classe[cite: 2], il retraining comporta un costo computazionale pari a $O(R \cdot K)$ round di comunicazione e addestramento locale[cite: 1, 2].

Per superare questo limite, abbiamo implementato la metodologia post-training proposta da Wang et al. (*FedPrune*), che consente di disimparare una specifica classe partendo direttamente dal modello globale già addestrato $W^o$, operando in due fasi sequenziali:
1. **Potatura mirata (Class-Discriminative Pruning)** dei parametri maggiormente responsabili del riconoscimento della classe target.
2. **Ripristino federato rapido (Federated Fine-Tuning)** limitato a pochissimi round sulle sole classi rimanenti, per sanare il danno collaterale subito dalle classi trattenute.

---

## 2. Fondamento Metodologico (Wang et al., WWW 2022)

### 2.1 Calcolo della Sensibilità di Canale (Class-Discriminative Sensitivity)
Nel paper di Wang et al., viene evidenziato come all'interno delle reti convoluzionali i singoli canali/filtri presentino gradi di attivazione eterogenei a seconda della classe in ingresso.

Per identificare i filtri maggiormente legati alla classe da dimenticare $c$, calcoliamo l'attivazione media per ciascun canale $k$ al passaggio dei soli campioni appartenenti alla classe target ($D_c$):
$$CS_{k, c} = \frac{1}{|D_c|} \sum_{x \in D_c} \text{mean}(a_k(x))$$

Nel nostro codice, abbiamo registrato le attivazioni tramite *forward hook* sui moduli convoluzionali della rete:
* `features[0]` (primo blocco convoluzionale: 32 filtri)[cite: 4]
* `features[3]` (secondo blocco convoluzionale: 64 filtri)[cite: 4]

### 2.2 Pruning Selettivo e Azzeramento del Classificatore
Definito un parametro di potatura $\rho$ (`prune_ratio = 0.3`):
1. **Filtri Convoluzionali:** Vengono individuati i primi $\lfloor K \cdot \rho \rfloor$ canali con il punteggio di sensibilità $CS_{k, c}$ più elevato all'interno di `features[0]` e `features[3]`, e i relativi pesi e bias vengono azzerati[cite: 4]:
   $$W_{\text{features}}[k] \leftarrow 0, \quad b_{\text{features}}[k] \leftarrow 0$$
2. **Neurone Finale di Classe:** La riga corrispondente alla classe target $c$ all'interno dell'ultimo strato lineare `classifier[2]` viene azzerata sia nei pesi che nei bias[cite: 4]:
   $$W_{\text{classifier}}[c] \leftarrow 0, \quad b_{\text{classifier}}[c] \leftarrow 0$$
   Questo passaggio annulla direttamente il contributo nei logit per la classe che si desidera disimparare.

### 2.3 Federated Recovery (Fine-Tuning Veloce)
Poiché i filtri convoluzionali azzerati estraevano pattern condivisi anche con le altre classi, il modello subisce un calo di accuratezza temporaneo sulle classi da mantenere (*collateral damage*).
Per recuperare le prestazioni senza rieseguire decine di round:
* Si avvia un ciclo federato compatto di soli **3 round** di fine-tuning con un learning rate dimezzato (`FL_LR * 0.5`)[cite: 1].
* Durante questi round, i client si addestrano **esclusivamente sui campioni delle 9 classi trattenute**[cite: 2].
* Al termine di ogni round di aggregazione FedAvg, i pesi del neurone target $c$ vengono ri-azzerati forzatamente per impedirne la ri-acquisizione.

---

## 3. Algoritmo di Attacco FUIA (Zhou et al., TIFS 2026)

Terminata la procedura di FedPrune su una singola classe target (es. classe $c = 3$[cite: 3]), il modello disimparato $W^u$ viene sottoposto all'**Algoritmo 3** del paper di Zhou et al. per valutare se la classe rimossa risulti identificabile[cite: 2].

L'attaccante calcola la variazione norma-$L_1$ dei pesi e la deviazione assoluta dei bias dell'ultimo layer (`classifier[2]`) rispetto al modello originale $W^o$[cite: 2, 4]:
$$\Delta w_i = \|w_o[i] - w_u[i]\|_1 \quad \forall i \in \{0, \dots, 9\}$$[cite: 2]
$$\Delta b_i = |b_o[i] - b_u[i]| \quad \forall i \in \{0, \dots, 9\}$$[cite: 2]

Normalizzando rispetto alla somma totale su tutte le classi[cite: 2]:
$$v_{\text{sum}} = \sum_{j=0}^{9} \Delta w_j, \quad b_{\text{sum}} = \sum_{j=0}^{9} \Delta b_j$$[cite: 2]

Viene calcolato lo **Score di Discriminazione $S_d[i]$** con $\beta = 0.5$[cite: 2, 3]:
$$S_d[i] = \beta \frac{\Delta w_i}{v_{\text{sum}}} + (1 - \beta) \frac{\Delta b_i}{b_{\text{sum}}}$$[cite: 2]

La classe predetta dall'attacco corrisponde all'indice con lo score massimo:
$$\hat{c} = \arg\max_i S_d[i]$$[cite: 2]

---

## 4. Analisi Comparativa: Retraining vs. FedPrune

| Proprietà Metodologica | Retraining da Zero (Exact FU) | FedPrune / CDP (Approximate FU) |
| :--- | :--- | :--- |
| **Punto di Partenza** | Checkpoint iniziale casuale $W_{\text{init}}$ | Modello finale addestrato $W^o$ |
| **Numero di Round FU** | 50 – 80 round completi[cite: 1] | **3 round** di fine-tuning rapido |
| **Tempo di Esecuzione FU** | Minuti (ri-addestramento intero) | **Pochi secondi** |
| **Comportamento Pesi Target** | Rimangono allo stato casuale iniziale | Vengono azzerati e congelati selettivamente |
| **Vulnerabilità a FUIA ($S_d$)** | **100% Success** (Divergenza netta) | **100% Success** (Divergenza netta) |

### Esito dell'Attacco su FedPrune
Mentre nel Retraining la divergenza su $\Delta w_c$ è dovuta al fatto che la classe target non riceve gradienti e resta ancorata a $W_{\text{init}}$, in **FedPrune** la divergenza è accentuata dall'azzeramento mirato ($W_{\text{classifier}}[c] \rightarrow 0$):
$$\Delta w_c = \|w_o[c] - 0\|_1 = \|w_o[c]\|_1$$
Poiché durante i 3 round di fine-tuning i pesi delle altre classi rimangono vicini ai valori originali $w_o[j]$ per continuità di convergenza, lo scostamento normalizzato $S_d[c]$ della classe target isola la classe rimossa al vertice del ranking, confermando che **l'attacco FUIA di Zhou et al. è efficace anche contro algoritmi di potatura post-training come FedPrune**.

