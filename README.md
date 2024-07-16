# Electricity-Price-Forecasting-Deep-Neural-Network-

Obiettivo di questo notebook è la previsione multivariata del prezzo dell'energia elettrica per le prossime 24 ore in Spagna, ma si utilizzano modelli multi output che prevedono contemporaneamente 24 valori successivi del prezzo.
Per semplificare si utilizzano come predittori solo il carico energetico passato, il prezzo, l'ora del giorno, il giorno della settimana e il mese, non si utilizzano quindi le informazioni meteorologiche. 


# Preparing the Data

Utilizzo `tf.datasets` per preparare i dati. La strategia generale è pulire, ridimensionare e dividere i dati prima di creare l'oggetto tf.dataset.

**Pulizia dei dati:** Compilare eventuali valori mancanti con un'interpolazione lineare del valore. Come fatto nel dataset di persistenza.

**Ridimensionamento dei dati:** In tutti i casi, i dati sono ridimensionati tramite minimo e massimo (min-max scaling).

**Features:** Come parte di questa analisi dei modelli, vengono preparati solo features contenenti consumo di energia, prezzo, giorno della settimana e mese dell'anno.

**Divisione dei dati:** Un anno di dati di test (8769 campioni orari) viene messo da parte per valutare tutti i modelli. I set di addestramento e di convalida vengono creati con uno split del 65/35, risultando in 9207 campioni di convalida - poco più di un anno.


# Windowing the Dataset

Per aumentare le performances si usa tf.dataset per creare un dataset con le windows-lags. 

Questo è un vettore di passi temporali precedenti (n_steps) che viene utilizzato per effettuare previsioni su un vettore target di passi futuri (n_horizon). 

Variabili utilizzate: n_steps = 72 e n_horizon = 24 e le 5 features estratte. 

Quindi utilizziamo gli ultimi 3 giorni (72 ore) per prevedere il giorno successivo (le successive 24 ore).

- Lo shape risultante per X sarà:  (dimensione_del_batch, n_steps, demensione_features) 
- Lo shape risultante per Y sarà:  (dimensione del batch, n_horizon).



# Model Configurations

Definisco un insieme di configurazioni dei modelli in modo che possiamo chiamare ed eseguire ciascun modello allo stesso modo. 

Il dizionario `cfg_model_run` conterrà il modello, la sua cronologia e il dataset di test generato.

I parametri predefiniti del modello sono:
- n_steps: ultimi 30 giorni
- n_horizon: prossime 24 ore
- learning rate: 3e-4


## CNN

Due strati Conv1D con 64 filtri ciascuno e dimensioni del kernel rispettivamente di 6 e 3. Dopo ogni strato Conv1D, uno strato maxpooling1D con dimensione di 2.


## CNN and LSTM with a skip connection

Strati CNN e LSTM, questa volta con una connessione con un salto (skip connection) diretta allo strato DNN comune, per evitare blocco del gradiente


# Evaluation of Training/Validation Results

Le curve loss dei 2 modelli sono piuttosto stabili. 

I modelli mostrano una curva di validation abbastanza piatta, mentre l'addestramento continua a diminuire. 

L'LSTM sembra iniziare a diventare molto sovra-adattato a partire da circa l'epoca xxx, dove la validation loss inizia a salire. 
