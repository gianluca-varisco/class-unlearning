# Multi-Class Unlearning: Motivazione e Obiettivo dell'Estensione (2, 3 e 4 Classi)

Questo modulo estende la validazione dell'attacco **FUIA (Federated Unlearning Inversion Attack)** allo scenario di **Multi-Class Unlearning**, analizzando la rimozione simultanea di un numero variabile di classi $N_{\mathrm{uc}} \in \{2, 3, 4\}$.

---

## 1. Perché stiamo testando la rimozione di 2, 3 e 4 classi?

Dopo aver validato l'efficacia dell'attacco sul caso a singola classe ($N_{\mathrm{uc}} = 1$), l'obiettivo è testare la **scalabilità e i limiti intrinseci dell'Algoritmo 3** di Zhou et al. (*IEEE TIFS 2026*), riproducendo empiricamente quanto documentato nella **Table I** e nella **Figure 12** del paper.

In particolare, il test risponde a tre domande scientifiche:

1. **Capacità di isolamento simultaneo (Top-k Ranking):**  
   Verificare se l'attaccante, estraendo i primi $k = N_{\mathrm{uc}}$ punteggi più elevati dallo Score di Discriminazione $S_d[i]$, sia in grado di isolare l'esatto sottoinsieme di classi cancellate senza falsi positivi.

2. **Validazione della soglia di degradazione teorica ($N_{\mathrm{uc}} = 4$):**  
   Secondo la Table I del paper, l'attacco FUIA ha un tasso di successo perfetto ($100\%$) quando si rimuovono fino a 3 classi ($N_{\mathrm{uc}} \in \{1, 2, 3\}$). Con 4 classi rimosse ($N_{\mathrm{uc}} = 4$), la probabilità di successo pieno scende all'**80%**, mentre nel **20%** dei casi l'attacco individua correttamente solo 3 classi su 4 ($N_{\mathrm{ic}} = 3$).

3. **Spiegazione del fenomeno di alterazione globale (*Backbone Drift*):**  
   Rimuovere 4 classi su 10 significa eliminare il **40% dell'intero dataset**. Una perturbazione così estesa non impatta unicamente l'ultimo strato lineare di classificazione, ma costringe anche gli strati convoluzionali precedenti a riorganizzare lo spazio latente delle feature. Questo incrementa il rumore di fondo, inducendo sovrapposizioni tra le feature di classi visivamente affini.

---

## 2. Quadro Teorico di Riferimento (Table I del Paper)

Il benchmark di riferimento definito dagli autori per valutare la probabilità predittiva dell'attacco è il seguente:

| Classi Rimosse ($N_{\mathrm{uc}}$) | Classi Identificate Correttamente ($N_{\mathrm{ic}}$) | Probabilità Predittiva Teorica | Comportamento Atteso |
| :---: | :---: | :---: | :--- |
| **1** | 1 | **100%** | Successo pieno (Top-1 isola la classe target) |
| **2** | 2 | **100%** | Successo pieno (Top-2 isolano le 2 classi) |
| **3** | 3 | **100%** | Successo pieno (Top-3 isolano le 3 classi) |
| **4** | 4 | **80%** | Successo pieno (Top-4 isolano tutte e 4 le classi) |
| **4** | 3 | **20%** | Successo parziale (3 classi corrette su 4) |

---

## 3. Metodologia di Calcolo Top-k

L'Algoritmo 3 viene generalizzato sostituendo la funzione $\arg\max$ con l'estrazione dei primi $k$ massimi:

1. Calcolo delle discrepanze su pesi e bias dell'ultimo layer per ogni classe $i \in \{0, \dots, 9\}$:
`v_diff[i] = ||v_o[i] - v_u[i]||_1`
`b_diff[i] = |b_o[i] - b_u[i]|`

2. Calcolo dello Score normalizzato $S_d[i]$ ($\beta = 0.5$):
   $$S_d[i] = \beta \cdot \frac{v_{\mathrm{diff}}[i]}{\sum_{j=0}^{9} v_{\mathrm{diff}}[j]} + (1 - \beta) \cdot \frac{b_{\mathrm{diff}}[i]}{\sum_{j=0}^{9} b_{\mathrm{diff}}[j]}$$

3. Predizione dell'insieme delle classi rimosse:
   $$\mathrm{classes\_predicted} = \mathrm{Top-}k_i (S_d[i]), \quad \text{con } k = N_{\mathrm{uc}}$$

---

## 4. Verifica Analitica dell'Unlearning (Metriche di Accuracy)

Per accertare che il retraining federato abbia effettivamente eliminato le classi selezionate, monitoriamo l'accuratezza del modello sul test set globale (10.000 campioni bilanciati su 10 classi). 

Il calo di accuratezza atteso sul modello unlearned $W^u$ rispetto all'originale $W^o$ ($\approx 96.2\%$) deve essere proporzionale al numero di classi rimosse:
* **$N_{\mathrm{uc}} = 2$:** calo teorico di circa il **20%** $\rightarrow \text{Acc } W^u \approx 76.0\% - 77.0\%$
* **$N_{\mathrm{uc}} = 3$:** calo teorico di circa il **30%** $\rightarrow \text{Acc } W^u \approx 66.0\% - 67.0\%$
* **$N_{\mathrm{uc}} = 4$:** calo teorico di circa il **40%** $\rightarrow \text{Acc } W^u \approx 56.0\% - 57.0\%$



