# Experimental Evaluation: Multi-Class Unlearning (FUIA)

Questo documento sintetizza i risultati empirici ottenuti dall'estensione dell'attacco **FUIA (Federated Unlearning Inversion Attack)** allo scenario di **Multi-Class Unlearning** su dataset MNIST, includendo l'analisi dell'impatto del parametro `DATA_PER_CLIENT`.

L'obiettivo dell'esperimento è testare la scalabilità e i limiti intrinseci dell'**Algoritmo 3** formalizzato da Zhou et al. (*IEEE TIFS 2026*), validando empiricamente la **Table I** e la degradazione teorica all'aumentare delle classi rimosse ($N_{\mathrm{uc}} \in \{2, 3, 4\}$).

---

## 1. Setup Sperimentale e Analisi del Parametro `DATA_PER_CLIENT`

Tutti i test sono stati condotti mediante **Retraining federato da zero** per garantire un unlearning esatto, utilizzando il medesimo stato iniziale dei parametri ($W_{\mathrm{init}}$ con `SEED = 42`) e senza pre-addestramento centralizzato per preservare la dinamica federata:

* **Architettura di rete:** CNN a 2 strati convoluzionali + MaxPool + Fully Connected (512 unità) + Linear Classifier (10 classi)
* **Algoritmo di aggregazione:** FedAvg (50 client totali, frazione di selezione per round = 0.2, ovvero 10 client per round)
* **Round di training federato:** 50 round
* **Epoche locali per client:** 3 epoche (Batch Size = 32, Learning Rate = 0.01 con linear decay)
* **Iperparametro di discriminazione:** $\beta = 0.5$

###  Motivazione Tecnica: Transizione da `DATA_PER_CLIENT = 1` a `100`

Nel paper originale (Sec. VI.B), gli autori utilizzano `DATA_PER_CLIENT = 1` abbinato a un pre-training centralizzato su 48.000 campioni. Rimuovendo il pre-training (necessario per evitare che il modello mantenga memoria centralizzata delle classi rimosse), la scelta del volume di dati locale impatta direttamente la stabilità dell'attacco:

1. **Criticità con `DATA_PER_CLIENT = 1` (Setup Minimale):**
   * Con 50 client e 1 campione ciascuno, il dataset federato globale è composto da appena **50 immagini totali** (~5 campioni per classe).
   * Il modello originale $W^o$ raggiunge un'accuratezza limitata ($\approx 56.0\%$).
   * Durante il retraining, alcune classi target (es. classe 3) appaiono su pochissimi client (2-3 campioni in tutto), producendo un segnale di divergenza $S_d$ troppo debole ($S_d = 0.0973$) che viene scavalcato dal rumore stocastico di classi trattenute (es. classe 1 con $S_d = 0.1630$), portando a un esito parziale non rappresentativo.

2. **Stabilizzazione con `DATA_PER_CLIENT = 100` (Setup Realistico):**
   * Con 100 campioni per client, il dataset federato raggiunge **5.000 immagini bilanciate IID** (~500 campioni per classe).
   * Il modello originale $W^o$ raggiunge una convergenza solida con **accuratezza al 96.2%**.
   * L'eliminazione delle classi produce un calo di accuratezza proporzionale e un segnale di deviazione sui pesi ($v_{\mathrm{diff}}$ e $b_{\mathrm{diff}}$) statisticamente robusto, consentendo di misurare con precisione i limiti effettivi dell'algoritmo di attacco.

---

## 2. Quadro Comparativo: Risultati Sperimentali vs. Table I (Paper)

| Test Sperimentale | Classi Rimosse ($N_{\mathrm{uc}}$) | $N_{\mathrm{ic}}$ Sperimentale | Esito Sperimentale | $N_{\mathrm{ic}}$ Atteso (Paper) | Probabilità Teorica | Accuratezza $W^o \rightarrow W^u$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test 1** | 2 (`[3, 4]`) | **2** | **FULL SUCCESS (100%)** | 2 | **100%** | $96.2\% \rightarrow 76.7\%$ ($\Delta -19.5\%$) |
| **Test 2** | 3 (`[2, 4, 6]`) | **3** | **FULL SUCCESS (100%)** | 3 | **100%** | $96.2\% \rightarrow 67.5\%$ ($\Delta -28.7\%$) |
| **Test 3** | 4 (`[2, 3, 6, 7]`) | **3** | **PARTIAL SUCCESS (75%)** | 3 | **20%** | $96.2\% \rightarrow 57.5\%$ ($\Delta -38.7\%$) |

---

## 3. Analisi Dettagliata delle Esecuzioni

### Test 1: Rimozione di 2 Classi ($N_{\mathrm{uc}} = 2$) — Target: `[3, 4]`

* **Accuratezza globale:**
  * Modello Originale ($W^o$): **96.2%**
  * Modello Unlearned ($W^u$): **76.7%** (calo del $19.5\%$, teoricamente atteso $\approx 20\%$)
* **Distribuzione Score di Discriminazione $S_d[i]$:**
  * Classe 0: $0.0812$
  * Classe 1: $0.0643$
  * Classe 2: $0.0902$
  * **Classe 3:** **$0.1796$** (Top-2 predizione)
  * **Classe 4:** **$0.2170$** (Top-1 predizione)
  * Classe 5: $0.0804$
  * Classe 6: $0.0842$
  * Classe 7: $0.0709$
  * Classe 8: $0.0728$
  * Classe 9: $0.0595$
* **Esito:** `FULL SUCCESS (100%)` — Tutte le $2$ classi rimosse sono state isolate al vertice del ranking.

---

### Test 2: Rimozione di 3 Classi ($N_{\mathrm{uc}} = 3$) — Target: `[2, 4, 6]`

* **Accuratezza globale:**
  * Modello Originale ($W^o$): **96.2%**
  * Modello Unlearned ($W^u$): **67.5%** (calo del $28.7\%$, teoricamente atteso $\approx 30\%$)
* **Distribuzione Score di Discriminazione $S_d[i]$:**
  * Classe 0: $0.0717$
  * Classe 1: $0.0832$
  * **Classe 2:** **$0.1545$** (Top-2 predizione)
  * Classe 3: $0.0876$
  * **Classe 4:** **$0.1616$** (Top-1 predizione)
  * Classe 5: $0.0804$
  * **Classe 6:** **$0.1428$** (Top-3 predizione)
  * Classe 7: $0.0983$
  * Classe 8: $0.0585$
  * Classe 9: $0.0613$
* **Esito:** `FULL SUCCESS (100%)` — Tutte le $3$ classi rimosse sono state isolate con margine netto rispetto al rumore medio ($S_{d,\mathrm{noise}} \approx 0.0773$).

---

### Test 3: Rimozione di 4 Classi ($N_{\mathrm{uc}} = 4$) — Target: `[2, 3, 6, 7]`

* **Accuratezza globale:**
  * Modello Originale ($W^o$): **96.2%**
  * Modello Unlearned ($W^u$): **57.5%** (calo del $38.7\%$, teoricamente atteso $\approx 40\%$)
* **Distribuzione Score di Discriminazione $S_d[i]$:**
  * Classe 0: $0.0756$
  * Classe 1: $0.0693$
  * **Classe 2:** **$0.1240$** (Top-2 predizione — Corretta)
  * Classe 3: $0.1135$ (Top-5 ranking — Esclusa per $\Delta = 0.0040$)
  * **Classe 4:** **$0.1175$** (Top-3 predizione — Falso Positivo)
  * **Classe 6:** **$0.1160$** (Top-4 predizione — Corretta)
  * **Classe 7:** **$0.1527$** (Top-1 predizione — Corretta)
  * Classe 8: $0.0501$
  * Classe 9: $0.0940$
* **Esito:** `PARTIAL (3/4)` — L'attacco identifica $3$ classi su $4$, riproducendo empiricamente il caso di degradazione documentato nella Table I del paper (probabilità di successo parziale pari al $20\%$).

---

## 4. Spiegazione Scientifica dei Risultati

La transizione dal $100\%$ di successo ($N_{\mathrm{uc}} \le 3$) alla comparsa di falsi positivi per $N_{\mathrm{uc}} = 4$ è motivata da due fattori teorici:

1. **Compressione del Margine di Discriminazione:**
   Poiché la somma degli score normalizzati è vincolata ($\sum S_d[i] = 1$), all'aumentare del numero di classi rimosse la quota media assegnata a ciascuna classe target decresce progressivamente (da $\approx 0.20$ con 2 classi a $\approx 0.12$ con 4 classi), riducendo il delta di separazione rispetto alle classi non rimosse ($0.07 - 0.09$).

2. **Feature Backbone Drift:**
   Rimuovere 4 classi su 10 corrisponde all'eliminazione del **40% dell'intera distribuzione di addestramento**. Una variazione così marcata non altera unicamente lo strato lineare di classificazione, ma induce una riconfigurazione dell'estrattore convoluzionale sottostante durante il retraining federato. Ciò genera interferenze costruttive su classi visivamente o semanticamente correlate (come la cifra `4` rispetto alla cifra `3`), portando occasionalmente una classe trattenuta a scavalcare di misura una classe target.

