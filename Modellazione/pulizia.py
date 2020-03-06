from Persone_Fisiche.funzioni import *

"""
In questo file verrà effettuata la pulizia dei dati grezzi risultanti dall'estrazione, secondo le specifiche riportate
nel documento "Specifiche Estrazione". I passaggi per ottenere il dataset pronto per passsare alla fase di 
preprocessamento saranno descritti tramite commenti in cui verrà indicato il loro scopo e la strategia utilizzata
per raggiungerlo. 
In particolare tramite i passaggi qui riportati si punta ad ottenere un dataset pulito e coerente per quanto riguarda
i soli titoli di credito appartenenti a Persone Fisiche, la pulizia per la parte di estrazione riferita 
a Persone Giuridiche sarà effettuata tramite un altro file, in quanto le due situazioni presentano marcate differenze.
"""

# Caricamento da file excel dell'estrazione (CRC.xlsm) su dataframe Pandas.
df = load_raw_data_excel("datasets/CRC.xlsx")

# Separazione e salvataggio come file csv (tramite interfaccia grafica) dei due dataset, riferiti a Persone Fisiche e
# Giuridiche
dfPG = df[df.TipoPersonalità == "PG"].copy()  # Persone Giuridiche
save_dataset(dfPG)
dfPF = df[df.TipoPersonalità == "PF"].copy()  # Persone Fisiche
save_dataset(dfPF)

"""
----------------------------------------PULIZIA DATASET PERSONE FISICHE-------------------------------------------------
"""
# Load del dataset da file .csv (altrimenti il formato datetime di excel da problemi)
dfPF = load_raw_data("datasets/CRCPF.csv")

# Droppo le colonne Cessata e CessataDataInfo che non riguardano le Persone Fisiche in questo dataset
dfPF.drop(columns=["Cessata", "CessataDataInfo"], inplace=True)

# Imposto i NaT in datanascita e decedutodatainfo come 'assente'
dfPF[["DecedutoDataInfo"]] = dfPF[["DecedutoDataInfo"]].fillna("assente")
dfPF[["DataNascita"]] = dfPF[["DataNascita"]].fillna("assente")

# Sono presenti 2 titoli con indirizzo completamnete sconosciuto, li elimino dall'estrazione
dfPF = dfPF[~(dfPF['IndirizzoResidenza'] == "SCONOSCIUTO ALL'ANAGRAFE SCONOSCIUTO ALL'ANAGRAFE")]

# Imposto i valori mancanti nella colonna Cap a 0
dfPF[["Cap"]] = dfPF[["Cap"]].fillna(0)

# A questo punto i singoli titoli di credito riferiti a persone fisiche sono 54589

""" -----------------------------------------------NON DOVREBBE PIU ESSERE NECESSARIO CON L'ULTIMA ESTRAZIONE---------
# GESTIONE ANAGRAFICHE PERSONE FISICHE CHE IN REALTA' SONO AZIENDE
# Strategia: elimino le anagrafiche che sono sia persone fisiche che persone giuridiche in momenti diversi, in tale caso
# lo stesso idAnagrafica è presente in titoli indicati sia come apaprtenenti a una persona giuridica che persona fisica.
# Utilizzo il dataset df per individuare questi casi in quanto contiene sia PG che PF. Inoltre sono da segnalare come
# aziende anche i titoli che presentano un valore sulle colonne "Cessata" e "CessataDataInfo", e che non riportano la
# data di nascita.
aziendepersone = df.groupby("idAnagrafica").TipoPersonalità.nunique() > 1
listidpfpg = np.unique(df[df['idAnagrafica'].isin(aziendepersone[aziendepersone].index.values)].idAnagrafica.values)
listnodate = np.unique(dfPF.query("DataNascita=='assente'").idAnagrafica.values)
setnodate = set(listnodate)  # anagrafiche senza data di nascita
setpfpg = set(listidpfpg)  # anagrafiche presenti sia come PG che PF
setunionpfdf = setnodate.union(setpfpg)  # unione delle due liste precedenti
todeletelist = list(setunionpfdf)
# Trovo gli id delle anagrafiche in pf con cessata attività:
listcessatapf = np.unique(
    dfPF.query("Cessata==1").idAnagrafica.values)  # anagrafiche che presentano valore 1 in cessata
final_list_todelete = list(
    set(listcessatapf) | set(todeletelist))  # unisce le due liste per ottenere tutte le anagrafiche da eliminare
# Elimino queste anagrafiche:
dfPF = dfPF[~dfPF['idAnagrafica'].isin(final_list_todelete)]
# Ora non contiene più titoli riferiti alle anagrafiche aziendali, rimangono 54550 titoli

"""

# GESTIONE DATAEMISSIONETITOLO NULLE, ovvero tutti i titoli con tipo di credito "oneri d'esazione"
# STRATEGIA: Imposto valori nulli ad 'assente' e successivamente sostituirò con mediana della vetustà
dfPF[["DataEmissioneTitolo"]] = dfPF[["DataEmissioneTitolo"]].fillna("assente")

# GESTIONE DATAPRIMANOTIFICA NULLE
# STRATEGIA: Imposto I valori nulli di DataPrimaNotifica e DataPagamentoTotale ad 'assente'. Successivamente, dove
# il valore di DataPrimaNotifica risulta 'assente', imposto il suo valore uguale a datacaricotitolo.
# Successivamente controllo che DataPagamentoTotale, dove presente nei casi in cui ho impostato datanotifica=datacarico,
# sia ad una distanza di giorni massima da datacarico/datanotifica di 120 giorni.
# per i titoli in cui questa condizione è soddisfatta imposto pagato120giorni=valoretitolo
dfPF[["DataPrimaNotifica"]] = dfPF[["DataPrimaNotifica"]].fillna("assente")
dfPF[["DataPagamentoTotale"]] = dfPF[["DataPagamentoTotale"]].fillna("assente")
# Imposto dove il valore risulta assente, DataPrimaNotifica a DataCaricoTitolo
dfPF.loc[(dfPF.DataPrimaNotifica == 'assente'), 'DataPrimaNotifica'] = dfPF.query(
    "DataPrimaNotifica=='assente'").DataCaricoTitolo

# Creo nuova colonna per giorni differenza
dfPF["DifferenzaPagamento"] = 0

# Dove il titolo risulta essere stato pagato completamente in futuro calcolo la differenza in giorni tra DataPagamentoTotale
# e DataCaricoTitolo (diventata uguale a notifica nei casi selezionati ovviamente)
differencecolumn = (pd.to_datetime(
    dfPF.query("DataPagamentoTotale != 'assente' & DataCaricoTitolo==DataPrimaNotifica").DataPagamentoTotale) -
                    pd.to_datetime(dfPF.query(
                        "DataPagamentoTotale != 'assente' & DataCaricoTitolo==DataPrimaNotifica").DataCaricoTitolo)).dt.days
# Imposto la differenza calcolata nella colonna DifferenzaPagamento
dfPF.loc[((dfPF.DataPagamentoTotale != 'assente') & (
        dfPF.DataCaricoTitolo == dfPF.DataPrimaNotifica)), 'DifferenzaPagamento'] = differencecolumn
# Se la DifferenzaPagamento calcolata nei casi trattati è inferiore o uguale a 120 giorni imposto il campo Pagato120Giorni
# all'intero valore del titolo, in quanto è stato risarcito interamnete a 120 giorni dalla data di carico del titolo e
# dunque sicuramente anche a 120 giorni dalla data di notifica sconosciuta, in quanto DataPrimaNotifica è sempre più
# recente di DataCaricoTitolo
dfPF.loc[((dfPF.DifferenzaPagamento != 0) & (dfPF.DifferenzaPagamento <= 120)), "Pagato120Giorni"] = \
    dfPF.loc[((dfPF.DifferenzaPagamento != 0) & (dfPF.DifferenzaPagamento <= 120)), "ValoreTitolo"]

# Ho impostato il dovuto pagato a 120 giorni al valore del titolo, Droppo la colonna temporanea utilizzata per effettuare
# il calcolo:
dfPF.drop(columns="DifferenzaPagamento", inplace=True)

# GESTIONE DEI CASI IN CUI, PER ANAGRAFICHE DUPLICATE NEL DATABASE, UN UNICO DOCUMENTO DI CREDITO (OVVERO FORMATO DA
# TITOLI CHE HANNO STESSO VALORE DI idAnagrafica, DataCaricoTitolo e DataPrimaNotifica) PRESENTI DIFFERENTI VALORI NEI
# CAMPI Telefono, Deceduto, IndirizzoResidenza o Cap

# Gestione campo Deceduto
grouped = dfPF.groupby(["idAnagrafica", "DataCaricoTitolo", "DataPrimaNotifica", "Deceduto"])
grouptitlesc = grouped.agg(['count'])
groupcumcount = grouptitlesc.groupby(level=[0, 1, 2]).cumcount().to_frame()
groupcumcount = groupcumcount.reset_index()
groupcumcount.columns = ["idAnagrafica", "DataCaricoTitolo", "Dataprimanotifica", "Deceduto", "cumcount"]
groupquery = groupcumcount.query("cumcount==1")
decedutodiverso = np.unique(groupquery["idAnagrafica"]).tolist()

# Gestione campo Telefono
grouped = dfPF.groupby(["idAnagrafica", "DataCaricoTitolo", "DataPrimaNotifica", "Telefono"])
grouptitlesc = grouped.agg(['count'])
groupcumcount = grouptitlesc.groupby(level=[0, 1, 2]).cumcount().to_frame()
groupcumcount = groupcumcount.reset_index()
groupcumcount.columns = ["idAnagrafica", "DataCaricoTitolo", "Dataprimanotifica", "Telefono", "cumcount"]
groupquery = groupcumcount.query("cumcount==1")
telefonodiverso = np.unique(groupquery["idAnagrafica"]).tolist()

# Gestione campo IndirizzoResidenza
grouped = dfPF.groupby(["idAnagrafica", "DataCaricoTitolo", "DataPrimaNotifica", "IndirizzoResidenza"])
grouptitlesc = grouped.agg(['count'])
groupcumcount = grouptitlesc.groupby(level=[0, 1, 2]).cumcount().to_frame()
groupcumcount = groupcumcount.reset_index()
groupcumcount.columns = ["idAnagrafica", "DataCaricoTitolo", "Dataprimanotifica", "IndirizzoResidenza", "cumcount"]
groupquery = groupcumcount.query("cumcount==1")
indirizzodiverso = np.unique(groupquery["idAnagrafica"]).tolist()

# Gestione campo Cap
grouped = dfPF.groupby(["idAnagrafica", "DataCaricoTitolo", "DataPrimaNotifica", "Cap"])
grouptitlesc = grouped.agg(['count'])
groupcumcount = grouptitlesc.groupby(level=[0, 1, 2]).cumcount().to_frame()
groupcumcount = groupcumcount.reset_index()
groupcumcount.columns = ["idAnagrafica", "DataCaricoTitolo", "Dataprimanotifica", "Cap", "cumcount"]
groupquery = groupcumcount.query("cumcount==1")
capdiverso = np.unique(groupquery["idAnagrafica"]).tolist()

# Lista con tutte le anagrafiche che presentano almeno un anomalie fra tutti i campi considerati
anagraficadifferente = ((set(telefonodiverso).union(set(indirizzodiverso))).union(set(decedutodiverso))).union(
    set(capdiverso))

# Raggruppando per credito (idAnagrafica, DataCaricoTitolo, DataPrimaNotifica) rendo omogenei i campi Cap, IndirizzoResidenza
# e Telefono (il campo Deceduto verrà trattato successivamente), ovvero per ogni raggruppamento dovrà esserci un valore unico
# per tutti i titoli che lo compongono
grouploc = dfPF.loc[(dfPF.idAnagrafica.isin(anagraficadifferente))].groupby(
    ["idAnagrafica", "DataCaricoTitolo", "DataPrimaNotifica"])

# IndirizzoResidenza: scelgo il valore di MODA tra quelli assunti dai titoli appartenenti al gruppo (credito)
dfPF.loc[(dfPF.idAnagrafica.isin(anagraficadifferente), 'IndirizzoResidenza')] = grouploc[
    "IndirizzoResidenza"].transform(lambda x: x.mode()[0])

# Cap: scelgo il valore di MODA tra quelli assunti dai titoli appartenenti al gruppo (credito)
# sarà comunque da risistemare in una fase successiva perchè la moda quì scelta per il cap può essere riferito
# all'indirizzo non scelto come moda precedentemente se i titoli dei 2 indirizzi nel gruppo erano nella stessa quantità)
dfPF.loc[(dfPF.idAnagrafica.isin(anagraficadifferente), 'Cap')] = grouploc["Cap"].transform(lambda x: x.mode()[0])

# Telefono: scelgo il valore di MODA tra quelli assunti dai titoli appartenenti al gruppo (credito)
dfPF.loc[(dfPF.idAnagrafica.isin(anagraficadifferente), 'Telefono')] = grouploc["Telefono"].transform(
    lambda x: x.mode()[0])

# ORA TELEFONO-INDIRIZZO-CAP SONO OMOGENEI, DOPO AVER PROCEDUTO CON L'OMOGENIZZAZIONE DI DECEDUTO SI POTRANNO AGGREGARE I CREDITI NEL CREDITO CHE LI INCLUDE

# GESTIONE DATE DI DECESSO, Campo Età e Deceduto al tempo del carico del titolo vanno cacolati successivamente.
# Inizialmente imposto il campo deceduto a 1 su tutti i titoli appartenenti ad un anagrafica di una persona deceduta
# Metto lo stesso valore di data di decesso su tutte le righe appartenenti alla stessa anagrafica
# Gruppo per id, se un valore datadecesso diverso da "assente" è presente, lo assegno a tutte le altre righe del gruppo
# Strategia: trasformo ogni data in numero intero, data più recente sarà un numero pià grande. le date NaN saranno
#           sostituite da un numero molto grande a simboleggiare l'infinito. Dopo gruppo per idAnagrafica e imposto
#           tutte le date al minimo con ['DataDeceduto'].transform(min) in quanto la data meno recente corrisponde
#           alla prima registrazione certa di decesso del contribuente. Una volta impostato su tutti i titoli la data
#           meno recente, ritrasformo il tipo intero in data.
dfPF["Deceduto"] = dfPF.groupby(['idAnagrafica'])['Deceduto'].transform(max)

# Avevo precedentemente sostituito i NaN con 'assente' in quanto il NaN di excel non è ricercabile tramite query su Pandas
# Ora però devo sostituire tale valore con NaN nuovamente
dfPF["DataDeceduto"] = dfPF.DecedutoDataInfo.replace('assente', np.NaN)
# Numero token per infinito: 9999-99-99 con cui sostituire i NaN
dfPF[['DataDeceduto']] = dfPF[['DataDeceduto']].fillna(value='9999-99-99')
dfPF['DataDeceduto'] = dfPF['DataDeceduto'].astype(str)
dfPF['DataDeceduto'] = dfPF['DataDeceduto'].str.replace('\D', '').astype(int)
# Ora posso raggruppare e trovare la data di decesso meno recente per ogni idAnagrafico che risulta deceduto
dfPF['DataDeceduto'] = dfPF.groupby(['idAnagrafica'])['DataDeceduto'].transform(min)  # Ora test non contiene più id
dfPF['DataDeceduto'] = pd.to_datetime(dfPF['DataDeceduto'].astype(str), format='%Y%m%d', errors='coerce').dt.date
# Ora DataDeceduto contiene le date di decesso corrette per le anagrafiche decedute, nel'caso l'anagrafica non sia
# deceduta il valore assunto sara NaT.
# Rimangono alcuni casi in cui il contribuente risulta deceduto ma non c'era nessuna registrazione della data di decesso
# in nessun titolo, questi casi verranno gestiti al momento del calcolo del decesso.

# E' possibile droppare la colonna DecedutoDataInfo in quanto la nuova colonna la sostituisce correttamente
dfPF.drop(columns="DecedutoDataInfo", inplace=True)
# Metto ad 'assente' i NaT corrispondendi a non deceduti o in cui è assente completamnete la data di decesso
dfPF[["DataDeceduto"]] = dfPF[["DataDeceduto"]].fillna('assente')
# Imposto la data decesso per i deceduti senza mai data a 2000-01-01, considerandoli quindi deceduti da sempre
dfPF.loc[(dfPF.Deceduto == 1) & (dfPF.DataDeceduto == 'assente'), 'DataDeceduto'] = datetime.date(2000, 1, 1)

# CALCOLO VETUSTA' TITOLO
# Strategia: sottrazione tra valore in DataCaricoTitolo e valore in DataEmissioneTitolo
# Bisogna ignorare i titoli di tipo "oneri di esazione" che non hanno tale data
# Questi saranno sostituiti in una fase successiva con la mediana di vetustà sul dataset se non sono stati aggregati
dfPF["Vetusta"] = -1
dfPF.loc[(dfPF.TipoCredito != 'oneri di esazione'), "Vetusta"] = (
        pd.to_datetime(dfPF.query("TipoCredito!='oneri di esazione'")['DataCaricoTitolo'])
        - pd.to_datetime(dfPF.query("TipoCredito!='oneri di esazione'")['DataEmissioneTitolo'])).dt.days
# Inserisco la mediana della vetustà come vetustà dei titoli di tipo 'oneri di esazione' che la hanno nulla
dfPF["Vetusta"] = dfPF.Vetusta.replace(-1, np.NaN)
medianavetusta=dfPF["Vetusta"].median()
dfPF[["Vetusta"]] = dfPF[["Vetusta"]].fillna(dfPF["Vetusta"].median())
# Setto colonna come intero invece che float
dfPF.Vetusta = dfPF.Vetusta.astype('int64')

# CALCOLO ETA' CONTRIBUENTE NELLA COLONNA ETA
# Strategia: sottrazione tra valore in DataCaricoTitolo e valore in DataNascita
dfPF["Eta"] = pd.to_datetime(dfPF['DataCaricoTitolo']).dt.year - pd.to_datetime(dfPF['DataNascita']).dt.year
# Ora la colonna Eta contiene le età dei titoli associati ai contribuenti del titolo, droppo la colonna AnnoNascita
dfPF.drop(columns="AnnoNascita", inplace=True)

# CALCOLO IL VALORE CORRETTO DI DECEDUTO NELLA COLONNA CESSATA (AL TEMPO DELLA DATA CARICO DEL TITOLO)
# Strategia: ottengo giorni di differenza per sottrazione tra valore in DataCaricoTitolo e valore in DataPrimaNotifica,
#            perchè è possibile si sia venuti a conoscenza del decesso durante l'analisi del contribuente, ma esso era comunque deceduto
#            all'arrivo del ruolo (DataCaricoTitolo)
dfPF["DifferenzaDeceduto"] = -9999  # inizializzo a alto negativo che indica che sicuramente non è deceduto
# In questo momento tutti i possibili deceduti hanno campo Deceduto con valore 1,
# filtro tramite quel valore e calcolo la differenza:
dfPF.loc[(dfPF.Deceduto == 1), 'DifferenzaDeceduto'] = (pd.to_datetime(dfPF.query("Deceduto==1").DataPrimaNotifica)
                                                        - pd.to_datetime(
            dfPF.query("Deceduto==1").DataDeceduto)).dt.days
# Ora dove la differenza è >=0 metto Deceduto a 1 altrimenti 0 (non deceduto al momento della notifica):
dfPF.loc[(dfPF.DifferenzaDeceduto.astype(int) >= 0), "Deceduto"] = 1 # per chiarezza ma inutile
dfPF.loc[(dfPF.DifferenzaDeceduto.astype(int) < 0), "Deceduto"] = 0

# Realmente deceduti di cui si era conoscenza alla data di notifica del titolo:
# 318 titoli, droppo la colonna DifferenzaDeceduto
dfPF.drop(columns="DifferenzaDeceduto", inplace=True)

# ORA I DATI DEI TITOLI APPARTENENTI A PERSONE FISICHE SONO PRONTI PER L'AGGREGAZIONE IN CREDITI


"""
--------------------------------------------AGGREGAZIONE E CLASSIFICAZIONE----------------------------------------------
"""

# DROPPO LE COLONNE CHE NON VERRANNO PIU' UTILIZZATE PERCHE' NON RILEVANTI AI FINI DELLA CLASSIFICAZIONE:
dfPF.drop(
    columns=["IDTitoloCredito","DataNascita", "TipoPersonalità", "PEC", "DataEmissioneTitolo", "DataPagamentoTotale", "DataDeceduto"],
    inplace=True)

# Ottengo una colonna per ogni tipo di titolo di credito (8 in totale), il tipo del titolo corrisponderà alla colonna
# in cui assumerà il valore 1
dfPF = pd.concat([dfPF, pd.get_dummies(dfPF['TipoCredito'])], axis=1)
dfPF.columns = [c.replace(' ', '_') for c in dfPF.columns]  # tolgo lo spazio nei nomi
dfPF.drop(columns=["TipoCredito"], inplace=True)  # NON CI SONO PIU ONERI DI ESAZIONE
for col in ['Servizi_Agricoli', 'Servizi_Amianto', 'Servizi_Cimiteriali', 'Servizi_Extratariffa_', 'Servizi_Speciali',
            'Servizi_Verde', 'Tariffa_Rifiuti']: dfPF[col] = dfPF[col].astype('int64')
# astype('int64') è utilizzato per cambiare il tipo da booleano a intero, in quanto poi i titoli dovranno essere sommati

# Sistemo le colonne NumeroTitoliRecenti e TotaleTitoliRecenti, in modo che contengano il conteggio e la somma dei
# titoli per lo stesso idAnagrafica alla stessa data di carico del titolo (Ruolo):
dfPF["NumeroTitoliRecenti"] += 1
dfPF["TotaleTitoliRecenti"] += dfPF["ValoreTitolo"]

# AGGREGAZIONE DEI TITOLI APPARTENENTI ALLO STESSO DOCUMENTO DI CREDITO (=IDANAGRAFICA-DATACREDITO-DATAPRIMANOTIFICA)
# Strategia: raggruppo i titoli per idAnagrafica, DataCaricoTitolo e DataPrimaNotifica, e aggrego le colonne secondo
#           i parametri scelti (ad esempio "sum" per ValoreTitolo e Pagato120Giorni)

# Parametri di aggregazione:
# Dove come metodo di aggregazione è specificato 'max' in realtà si mantiene l'unico valore di quella colonna presente
# nel gruppo, in quanto sono stati tutti resi uguali nella fase di pulizia precedente.
# Vetustà è determinata dal minimo: titolo più vecchio del gruppo, i campi corrispondenti al tipo del titolo vengono
# sommati per indicare quanti e per che tipo di titoli erano presenti all'interno dello stesso documento di credito.
aggprm = {'Telefono': 'max', 'IndirizzoResidenza': 'max', 'Cap': 'max', 'CittadinanzaItaliana': 'max',
          'Deceduto': 'max',
          'ValoreTitolo': 'sum', 'Pagato120Giorni': 'sum', 'NumeroTitoliAperti': 'max', 'DovutoTitoliAperti': 'max', 'ImportoTitoliAperti': 'max',
          'NumeroTitoliSaldati': 'max', 'ImportoTitoliSaldati': 'max', 'NumeroTitoliRecenti': 'max',
          'TotaleTitoliRecenti': 'max', 'Eta': 'max', 'Vetusta': 'max', 'Servizi_Agricoli': 'sum',
          'Servizi_Amianto': 'sum',
          'Servizi_Cimiteriali': 'sum', 'Servizi_Extratariffa_': 'sum', 'Servizi_Speciali': 'sum',
          'Servizi_Verde': 'sum',
          'Tariffa_Rifiuti': 'sum'}
# Procedo con l'aggregazione dei titoli appartenenti allo stesso docuemnto/credito:
dfPF = dfPF.groupby(["idAnagrafica", "DataCaricoTitolo", "DataPrimaNotifica"]).agg(aggprm)
# Resetto gli index del raggruppamento, in modo da continuare a vedere idAnagrafica, DataCaricoTitolo e DataPrimaNotifica
# come colonne e campi dei crediti:
dfPF = dfPF.reset_index()
# Specifiche del dataset risultante:
# Totale: 23452 crediti appartenenti persone fisiche
# Totale: 15453 anagrafiche univoche di persone fisiche


# Calcolo la label di classe per ogni credito
# Strategia: la label di classe vale 1 se il credito è associato ad un "buon pagatore" dove il "buon pagatore" è tale
# se ha pagato ad Abaco almeno il 20% del valore del credito a 120 giorni dalla data di ricevimento della prima notifica
dfPF['label'] = dfPF['Pagato120Giorni'] / dfPF["ValoreTitolo"] >= 0.20
dfPF['label'] = dfPF['label'].astype("int64")

# Stime finali prima di passare alla fase di preprocessamento:
# 4365 crediti sono positivi, contro 19087 negativi.
# Il dataset risulta quindi decisamente sbilanciato in favore della classe negativa.
# Circa 18.6% classe positiva e 81.4% classe negativa.

# Salvo il dataset risultante:
save_dataset(dfPF)