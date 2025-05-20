# Rimozione artefatti ricostruzione di Feldkamp 3D su angoli limitati

Sandra Campacci
Federico Augelli
Professore di Riferimento: Prof. Elena Loli Piccolomini

## Obiettivi del progetto:

L’obiettivo di questo progetto è :
1. Creare un data set sintetico 3D con oggetti che simulano gli oggetti di interesse nel seno
2. Realizzare un framework tramite ASTRA che ricostruisce su angoli limitati con algoritmo di
Feldkamp e poi rimuove gli artefatti con una rete di post processing

## Descrizione del progetto:

1. Estendere il data set sintetico 2D denominato Coule e scaricabile dal seguente (link)[https://www.kaggle.com/datasets/loiboresearchgroup/coule-dataset] al 3D, con volumi di dimensione 256x256x50 (circa 25-30 volumi)
2. Utilizzare il codice fornito basato su ASTRA che ricostruisce il volume a partire da proiezioni sintetiche ottenute in geometria ad angoli limitati caratteristica delle tomosintesi mammaria.
3. Implementare una rete di tipo REsUnet 3D che prende in input la ricostruzione ottenuta al punto 2 e ha come target i volumi del dataset. La rete prende in input un sottoinsieme di Nf fette alla volta (testare un valore di Np compatibile con le risorse disponibili).

## Output Attesi:

1. Testa il progetto precedente nelle seguenti geometrie
    - 11 angoli in [-15,15]:
    - 11 angoli in [-8.5,8.5]
    - In entrambi i casi sia senza rumore sul sinogramma che con rumore di livello 0.01.

2. Per ciascuna delle esecuzioni al punto 1. , utilizza le seguenti misure di errore: Relative Error (RE), Peak Signal to Noise Ratio (PSNR) e Structural Similarity Index (SSIM) calcolati tra i volumi ricostruiti e quelli del data set.
3. Esegui i test descritti al punto 1. su volume selezionato del test set, mostrando le ricostruzioni e uno o due ritagli significativi, insieme alle metriche indicate nel punto 2.
4. Esegui i test sull’intero test set, calcolando la media delle metriche indicate e riportandole in una tabella.

### Bibliografia
In seguito si riportano alcuni papers di riferimento che possono aiutare nello svolgimento del progetto (utilizzare google scholar per trovare e scaricare gli articoli) La letterature scientifica (specialmente in questo ambito) è però generalmente molto complessa e richiede una serie di conoscenze di base difficili da raggiungere. Per questo motivo, si consiglia di fare uso dei numerosi blog e siti web che descrivono le metodologie richieste dal progetto con un linguaggio estremamente più semplice di quello dei papers, oltre che di LLM che possono aiutare nello sviluppo e nella comprensione del codice. Infine, ricordo che scopo di questo progetto è una “collaborazione” diretta col docente di riferimento, motivo per cui non esitate a mandarmi mail o fissare ricevimenti in un qualunque momento.


**Consegna**: Il progetto NON deve essere consegnato. I risultati ottenuti dall’esecuzione degli esperimenti del progetto saranno discussi in sede d’esame, con le modalità che verranno comunicate nelle prossime lezioni.
Morotti, Elena, Davide Evangelista, and Elena Loli Piccolomini. "A green prospective for learned
post-processing in sparse-view tomographic reconstruction."
7.8 (2021): 139.
