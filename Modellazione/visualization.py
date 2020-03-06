from Persone_Fisiche.funzioni import *

#TODO load dataset and visualize

set_dataset_display_properties()

df=load_raw_data_excel("datasets/CRC.xlsx")
dfPF=load_raw_data("datasets/CRCPF.csv")

"""
1)Separare tutti i tipopersonalità in due dataset diversi
2)Salvare i due dataset come csv
"""

"""
dfPG=df[df.TipoPersonalità=="PG"].copy()
print(dfPG.head())
dfPF=df[df.TipoPersonalità=="PF"].copy()
print(dfPF.head())
save_dataset(dfPG)
save_dataset(dfPF)
"""


"""
-----------------------------------------------dfPF DATASET CON SOLE PERSONE FISICHE QUERY------------------------------

#QUERY attività cessata sono 158 titoli, quanti di questi hanno la data di cessata venuta dopo la data di assunzione (indica che erano aziende)
dfPF[dfPF["Cessata"]==1]

#QUERY 322 (di cui 95 delle casistiche con cessata nominate prima) rappresentano titoli con persone fisiche senza data di nascita, indica che probabilmente sono aziende
dfPF[dfPF["DataNascita"].isnull()]

#QUERY E' possibile che alcune anagrafiche abbiano la data di nascita su alcuni titoli (indicherebbe che non tutte le informazioni sulla stessa anagrafica provengono dalla stessa fonte)
test= dfPF.groupby("idAnagrafica").DataNascita.nunique(dropna=False)>1
test[test].index.values

#QUERY titoli con cap mancanti sono esteri o sconosciuti
dfPF[dfPF['Cap'].isnull()]

#QUERY titoli con indirizzo estero sconosciuto:
dfPF[dfPF['IndirizzoResidenza']=="ESTERO ESTERO EE"]

#QUERY titoli con indirizzo sconosciuto:
dfPF[dfPF['IndirizzoResidenza']=="SCONOSCIUTO ALL'ANAGRAFE SCONOSCIUTO ALL'ANAGRAFE"]

#QUERY titoli con anagrafiche con diversi indirizzi in diversi titoli:
morethanoneaddress=dfPF.groupby("idAnagrafica").IndirizzoResidenza.nunique()>1
idmorethanone=morethanoneaddress[morethanoneaddress].index.values
dfPF[dfPF['idAnagrafica'].isin(idmorethanone)]

#QUERY titoli con anagrafiche con diversi   CAP in diversi titoli, quanti casi sono dovuti ad errori? (cap uguale, stringa diversa)
testcap=dfPF[dfPF['idAnagrafica'].isin(idmorethanone)].groupby("idAnagrafica").Cap.nunique(dropna=False)>1
idcapmorethanone=testcap[testcap].index.values
dfPF[dfPF['idAnagrafica'].isin(idcapmorethanone)]

#QUERY tenend tutte le variabili stabilite in precedenza, lavoro con dei set contenenti gli id per comprendere l'origine 
 delle anomalie
set1=set(idlist)
set2=set(idmorethanone)
set3=set(idcapmorethanone)
#set3 è subset di set 2 ma set2 non è subset di set1, ciò significa che non tutti i contribuenti che 
    hanno indirizzi differenti in titoli differenti hanno anche dei ruoli spezzati (in titoli assunti nella stessa data 
    per lo stesso contribuente ci sono differenti valori per quanto riguarda il numero di titoli recenti)
#voglio trovare TUTTI i contribuenti che presentano nel tempo diversi indirizzi (e quindi caps) e stabilire 
 se ogni volta che l'indirizzo risulta diverso il ruolo è spezzato in due a seconda dell'indirizzo, in questo caso 
 si potrebbero trattare come ruoli diversi
setintersection=set1.intersection(set2)
listintersection=list(setintersection)
dfPF[dfPF['idAnagrafica'].isin(listintersection)]

#QUERY controllo che tutte le anagrafiche abbiano una unico valore per pec e deceduto
#PEC
testpec=dfPF.groupby("idAnagrafica").PEC.nunique()>1
idpecmorethanone=testpec[testpec].index.values #risulta vuoto, le pec sono coerenti per anagrafica
#DECEDUTO
testdec=dfPF.groupby("idAnagrafica").Deceduto.nunique()>1
idecmorethanone=testdec[testdec].index.values 
dfPF[dfPF['idAnagrafica'].isin(idecmorethanone)]#sono presenti titoli in cui una persona risulta deceduta per alcuni 
                                               ed altri no, ma è scorretto e dovuto a anagrafiche duplicate, corretto da
                                               passo della PULIZIA

#QUERY controllo che tutti i casi di deceduto=1 abbiano associata la data e se i casi in cui la data non c'è ci sia almeno un 
 altro titolo che la riporta
dfPF[["DecedutoDataInfo"]]=dfPF[["DecedutoDataInfo"]].fillna("assente")
dfPF.query("Deceduto==0 and DecedutoDataInfo!='assente'")
#non ci sono casi in cui deceduto==0 e ci sia una data
dfPF.query("Deceduto==1")
#ci sono 2015 titoli in cui il proprietario risulta deceduto
dfPF.query("Deceduto==1 and DecedutoDataInfo=='assente'")
#ci sono 29 titoli tra i 2015 segnalati deceduti in cui non è riportata la data del decesso
 
#QUERY controllo che le anagrafiche presentino la stessa data di nascita su ogni titolo
testa=dfPF.groupby("idAnagrafica").DataNascita.nunique()>1
testa=testa[testa].index.values #Risulta vuoto quindi non ci sono più date di nascita per anagrafica

#QUERY anagrafiche con titoli appartenenti a stesso ruolo con numero titoli aperti uguale e importo totale diverso
grouped=dfPF.groupby(["idAnagrafica","DataCaricoTitolo","ImportoTitoliAperti"])
grouptitlesc=grouped.agg(['count'])
groupcumcount=grouptitlesc.groupby(level=[0, 1]).cumcount().to_frame()
groupcumcount=groupcumcount.reset_index()
groupcumcount.columns=["idAnagrafica","DataCaricoTitolo","ImportoTitoliAperti","cumcount"]
groupquery=groupcumcount.query("cumcount==1") 
anagraficheanomale=np.unique(groupquery["idAnagrafica"])
dfPF[dfPF['idAnagrafica'].isin(anagraficheanomale.tolist())] #Per vedere tutti i titoli appartenenti ad anomalie 
#NOTA Non si capisce bene a cosa siano dovute queste anomalie, il conteggio del valore titoli aperti sembra a caso

#QUERY anagrafiche con titoli appartenenti a stesso ruolo con numero titoli recenti diverso dal conteggio titoli recenti
confronto il valore TITOLIRECENTI con il count dei titoli nel gruppo


#QUERY anagrafiche con diversi valori di numero di telefono
#i dati di ogni titolo sono dell'anagrafica, per calcolare pero i campi come ntitoli recenti ecc 
    si è usato il codice fiscale, quindi le incongruenze su più titoli con stessa anagrafica si devono a questo
testt=dfPF.groupby("idAnagrafica").Telefono.nunique()>1
testt=testt[testt].index.values #
dfPF[dfPF['idAnagrafica'].isin(testt.tolist())]










--------------------------------------INDIRIZZO O TELEFONO O DECEDUTO DIVERSI-------------------------------------------
#DECEDUTO
grouped=dfPF.groupby(["idAnagrafica","DataCaricoTitolo","DataPrimaNotifica","Deceduto"])
grouptitlesc=grouped.agg(['count'])
groupcumcount=grouptitlesc.groupby(level=[0, 1, 2]).cumcount().to_frame()
groupcumcount=groupcumcount.reset_index()
groupcumcount.columns=["idAnagrafica","DataCaricoTitolo","Dataprimanotifica","Deceduto","cumcount"]
groupquery=groupcumcount.query("cumcount==1")
decedutodiverso=np.unique(groupquery["idAnagrafica"]).tolist()

#TELEFONO
grouped=dfPF.groupby(["idAnagrafica","DataCaricoTitolo","DataPrimaNotifica","Telefono"])
grouptitlesc=grouped.agg(['count'])
groupcumcount=grouptitlesc.groupby(level=[0, 1, 2]).cumcount().to_frame()
groupcumcount=groupcumcount.reset_index()
groupcumcount.columns=["idAnagrafica","DataCaricoTitolo","Dataprimanotifica","Telefono","cumcount"]
groupquery=groupcumcount.query("cumcount==1")
telefonodiverso=np.unique(groupquery["idAnagrafica"]).tolist()

#INDIRIZZO
grouped=dfPF.groupby(["idAnagrafica","DataCaricoTitolo","DataPrimaNotifica","IndirizzoResidenza"])
grouptitlesc=grouped.agg(['count'])
groupcumcount=grouptitlesc.groupby(level=[0, 1, 2]).cumcount().to_frame()
groupcumcount=groupcumcount.reset_index()
groupcumcount.columns=["idAnagrafica","DataCaricoTitolo","Dataprimanotifica","IndirizzoResidenza","cumcount"]
groupquery=groupcumcount.query("cumcount==1")
indirizzodiverso=np.unique(groupquery["idAnagrafica"]).tolist()

#CAP (non deve avere valori nulli)
grouped=dfPF.groupby(["idAnagrafica","DataCaricoTitolo","DataPrimaNotifica","Cap"])
grouptitlesc=grouped.agg(['count'])
groupcumcount=grouptitlesc.groupby(level=[0, 1, 2]).cumcount().to_frame()
groupcumcount=groupcumcount.reset_index()
groupcumcount.columns=["idAnagrafica","DataCaricoTitolo","Dataprimanotifica","Cap","cumcount"]
groupquery=groupcumcount.query("cumcount==1")
capdiverso=np.unique(groupquery["idAnagrafica"]).tolist()

#LISTA CON TUTTE LE ANAGRAFICHE
anagraficadifferente=((set(telefonodiverso).union(set(indirizzodiverso))).union(set(decedutodiverso))).union(set(capdiverso))
#PER ELIMINARE TUTTI I TITOLI ASSOCIATI AD ANAGRAFICHE PROBLEMATICHE (1769 dopo eliminazione anagrafiche sia PF che PG)
#NON SONO DA ELIMINARE!!!!! dfPF = dfPF[~dfPF['idAnagrafica'].isin(anagraficadifferente)] !!!!!

#PER TRIPLA IDANAGRAFICA-DATACARICO-DATANOTIFICA UGUALI TRA TITOLI RENDO OMOGENEI I CAMPI NEL RAGGRUPPAMENTO ECCETTO DECEDUTO
grouploc=dfPF.loc[(dfPF.idAnagrafica.isin(anagraficadifferente) )].groupby(["idAnagrafica","DataCaricoTitolo","DataPrimaNotifica"])
#INDIRIZZO: MODA
dfPF.loc[(dfPF.idAnagrafica.isin(anagraficadifferente), 'IndirizzoResidenza' )]=grouploc["IndirizzoResidenza"].transform(lambda x: x.mode()[0])
#CAP: MODA (è comunque da ricalcolare perchè la moda che sceglie per il cap può essere riferito all'indirizzo non scelto
       se i titoli dei 2 indirizzi nel gruppo erano nella stessa quantità)
dfPF.loc[(dfPF.idAnagrafica.isin(anagraficadifferente), 'Cap' )]=grouploc["Cap"].transform(lambda x: x.mode()[0]) #NAN NON HASHABLE ERRORE DEVO PRIMA FILLARE
#TELEFONO: MODA
dfPF.loc[(dfPF.idAnagrafica.isin(anagraficadifferente), 'Telefono' )]=grouploc["Telefono"].transform(lambda x: x.mode()[0])
#ORA TELEFONO-INDIRIZZO-CAP SONO OMOGENEI, DOPO AVER PROCEDUTO CON L'OMOGENIZZAZIONE DI DECEDUTO SI POTRANNO AGGREGARE I CREDITI



----------------------------------------------------PULIZIA-------------------------------------------------------------

#PREPARAZIONE
1) impostare i NaT in datanascita e decedutodatainfo come 'assente' 
    dfPF[["DecedutoDataInfo"]]=dfPF[["DecedutoDataInfo"]].fillna("assente")
    dfPF[["DataNascita"]]=dfPF[["DataNascita"]].fillna("assente")
2) eliminare i 2 titoli con tutto sconosciuto ()
    dfPF = dfPF[~(dfPF['IndirizzoResidenza']=="SCONOSCIUTO ALL'ANAGRAFE SCONOSCIUTO ALL'ANAGRAFE")]
3) Imposto i nan di Cap a 0
    dfPF[["Cap"]]=dfPF[["Cap"]].fillna(0)
    
    

#GESTIONE ANAGRAFICHE PERSONE FISICHE CHE IN REALTA' SONO AZIENDE
#Elimino le anagrafiche che sono sia personefisiche che personegiuridiche in momenti diversi (df contiene sia pg che pf)
aziendepersone=df.groupby("idAnagrafica").TipoPersonalità.nunique()>1
listidpfpg=np.unique(df[df['idAnagrafica'].isin(aziendepersone[aziendepersone].index.values)].idAnagrafica.values)
#Sono 139 anagrafiche, le elimino dalla lista di persone fisiche, elimino anche quelle che non hanno date di nascita
listnodate=np.unique(dfPF.query("DataNascita=='assente'").idAnagrafica.values)
setnodate=set(listnodate)
setpfpg=set(listidpfpg)
setunionpfdf=setnodate.union(setpfpg) #contiene tutte le anagrafiche da eliminare
todeletelist=list(setunionpfdf)
dfPF[dfPF["idAnagrafica"].isin(todeletelist)] #Per vedere tutti i titoli che saranno eliminati (927)
                                               a questi sono da aggiungere altri non inclusi in cui l'anagrafica ha
                                               riportato in almeno in un titolo cessata attività.
#Trovo gli id delle anagrafiche in pf con cessata attività:
listcessatapf=np.unique(dfPF.query("Cessata==1").idAnagrafica.values)
final_list_todelete = list(set(listcessatapf) | set(todeletelist)) 
dfPF[dfPF["idAnagrafica"].isin(final_list_todelete)] #Totale titoli che verranno eliminati: 994
                                                      Totale anagrafiche eliminate: 224
#Elimino queste anagrafiche:
dfPF = dfPF[~dfPF['idAnagrafica'].isin(final_list_todelete)]  #Ora non contiene più titoli riferiti alle 
                                                                anagrafiche aziendali, rimangono 54550 titoli

#DROP Cessata e CessataDataInfo
dfPF.drop(columns=["Cessata","CessataDataInfo"], inplace=True)

#GESTIONE DATAEMISSIONETITOLO NULLE (sono tutti quelli di tipo "oneri d'esazione")
#STRATEGIA: Imposto valori nulli ad 'assente' e successivamente sostituirò con mediana della vetustà 
dfPF[["DataEmissioneTitolo"]]=dfPF[["DataEmissioneTitolo"]].fillna("assente") 

#GESTIONE DATAPRIMANOTIFICA NULLE
#STRATEGIA: Imposto valori nulli di datanotifica e datapagamentototale ad 'assente' poi imposto dove assente 
 dataprimanotifica uguale a datacaricotitolo, successivamente controllo che datapagamentototale, dove presente nei casi
 in cui ho impostato datanotifica=datacarico, sia a una differenza di massimo 120 giorni da datacarico/datanotifica,
 per i titoli in cui questa condizione è soddisfatta imposto pagato120giorni=valoretitolo
dfPF[["DataPrimaNotifica"]]=dfPF[["DataPrimaNotifica"]].fillna("assente")
dfPF.loc[(dfPF.DataPrimaNotifica == 'assente'), 'DataPrimaNotifica'] = dfPF.query("DataPrimaNotifica=='assente'").DataCaricoTitolo
dfPF[["DataPagamentoTotale"]]=dfPF[["DataPagamentoTotale"]].fillna("assente")
dfPF["DifferenzaPagamento"]=0 #Creo nuova colonna per giorni differenza
differencecolumn=
    (pd.to_datetime(dfPF.query("DataPagamentoTotale != 'assente' & DataCaricoTitolo==DataPrimaNotifica").DataPagamentoTotale) -
    pd.to_datetime(dfPF.query("DataPagamentoTotale != 'assente' & DataCaricoTitolo==DataPrimaNotifica").DataCaricoTitolo)).dt.days
dfPF.loc[((dfPF.DataPagamentoTotale != 'assente') & (dfPF.DataCaricoTitolo==dfPF.DataPrimaNotifica)), 'DifferenzaPagamento']=
    differencecolumn
dfPF.loc[((dfPF.DifferenzaPagamento != 0) & (dfPF.DifferenzaPagamento <= 120)),"Pagato120Giorni"] =
    dfPF.loc[((dfPF.DifferenzaPagamento != 0) & (dfPF.DifferenzaPagamento <= 120)),"ValoreTitolo"]
#Ho impostato il dovuto pagato in 120 giorni al valore del titolo, droppo la colonna inutile
dfPF.drop(columns="DifferenzaPagamento",inplace=True)


#GESTIONE BLOCCHI ANAGRAFICA-DATACARICO-DATATITOLO CON ANAGRAFICHE DUPLICATE
------------------VEDI SEZIONE "INDIRIZZO O TELEFONO O DECEDUTO DIVERSI"---- 


#GESTIONE DATE DI DECESSO ----------Campo Età e Deceduto vanno cacolati successivamente------------
#Inizialmente imposto il campo deceduto a 1 su tutti i titoli appartenenti ad un anagrafica di una persona deceduta
dfPF["Deceduto"]=dfPF.groupby(['idAnagrafica'])['Deceduto'].transform(max)

#Metto lo stesso valore di data di nascita su tutte le righe appartenenti alla stessa anagrafica
#Gruppo per id, se un valore datadecesso diverso da "assente" è presente, lo assegno a tutte le altre righe del gruppo
#Strategia: trasformo ogni data in numero intero, data più recente sarà un numero pià grande. le date NaN saranno 
            sostituite da un numero molto grande a simboleggiare l'infinito. Dopo gruppo per idAnagrafica e imposto
            tutte le date al minimo con ['DataDeceduto'].transform(min) in quanto la data meno recente corrisponde
            alla prima registrazione certa di decesso del contribuente. Una volta impostato su tutti i titoli la data
            meno recente, ritrasformo il tipo intero in data.
dfPF["DataDeceduto"]=dfPF.DecedutoDataInfo.replace('assente', np.NaN) #Avevo riempito i nan con assente, ma ora servenan
####
    test=dfPF.groupby("idAnagrafica").DataDeceduto.nunique()>1 
    test=test[test].index.values  #Non vuoto: ci sono anagrafiche con più date di decesso non nulle
####
#Numero token per infinito: 9999-99-99 con cui sostituire i NaN
dfPF[['DataDeceduto']] = dfPF[['DataDeceduto']].fillna(value='9999-99-99')
dfPF['DataDeceduto'] = dfPF['DataDeceduto'].astype(str)
dfPF['DataDeceduto'] = dfPF['DataDeceduto'].str.replace('\D', '').astype(int) #Ora posso gruppare e trovare minimo
dfPF['DataDeceduto'] = dfPF.groupby(['idAnagrafica'])['DataDeceduto'].transform(min) #Ora test non contiene più id
dfPF['DataDeceduto'] = pd.to_datetime(dfPF['DataDeceduto'].astype(str), format='%Y%m%d', errors='coerce').datetime.date
#Ora DataDeceduto contiene le date di decesso corrette per le anagrafiche decedute, nel'caso l'anagrafica non sia 
 deceduta il valore assunto sara NaT. Rimangono alcuni casi in cui il contribuente risulta deceduto ma non c'era nessuna
 registrazione della data di decesso in nessun titolo, questi casi verranno gestiti al momento del calcolo del decesso
#E' possibile droppare la colonna DecedutoDataInfo in quanto la nuova colonna la sostituisce correttamente
dfPF.drop(columns="DecedutoDataInfo", inplace=True)
#Metto ad assente i NaT
dfPF[["DataDeceduto"]] = dfPF[["DataDeceduto"]].fillna('assente') 
#Imposto la data decesso per i deceduti senza mai data a 2000-01-01
dfPF.loc[(dfPF.Deceduto==1)&(dfPF.DataDeceduto=='assente'), 'DataDeceduto']=datetime.date(2000,1,1)

#CALCOLO VETUSTA' TITOLO
#Strategia: sottrazione tra valore in DataCaricoTitolo e valore in DataEmissioneTitolo
#Bisogna ignorare i titoli di tipo "oneri di esazione" che non hanno tale data
#Questi saranno sostituiti in una fase successiva con la mediana di vetustà sul dataset se non sono stati aggregati
dfPF["Vetusta"]=-1
dfPF.loc[(dfPF.TipoCredito!='oneri di esazione'),"Vetusta"] = (pd.to_datetime(dfPF.query("TipoCredito!='oneri di esazione'")['DataCaricoTitolo']) 
                                                        - pd.to_datetime(dfPF.query("TipoCredito!='oneri di esazione'")['DataEmissioneTitolo'])).dt.days
#Fillo i titoli 'oneri' con la mediana
dfPF["Vetusta"]=dfPF.Vetusta.replace(-1, np.NaN)
dfPF[["Vetusta"]]=dfPF[["Vetusta"]].fillna(dfPF["Vetusta"].median())
#Setto colonna come intero invece che float
dfPF.Vetusta = dfPF.Vetusta.astype('int64')


#CALCOLO ETA' CONTRIBUENTE NELLA COLONNA ETA'
#Strategia: sottrazione tra valore in DataCaricoTitolo e valore in DataNascita
dfPF["Eta"] = pd.to_datetime(dfPF['DataCaricoTitolo']).dt.year - pd.to_datetime(dfPF['DataNascita']).dt.year
#Ora la colonna Eta contiene le età dei titoli associati ai contribuenti del titolo, droppo la colonna AnnoNascita
dfPF.drop(columns="AnnoNascita",inplace=True)

#CALCOLO IL VALORE CORRETTO DI DECEDUTO NELLA COLONNA DECEDUTO (AL TEMPO DELLA DATA CARICO DEL TITOLO)
#Strategia: ottengo giorni di differenza per sottrazione tra valore in DataCaricoTitolo e valore in DataPrimaNotifica, 
    perchè è possibile si sia venuti a conoscenza del decesso durante l'analisi del contribuente e non all'arrivo del ruolo
dfPF["DifferenzaDeceduto"]=-9999 #inizializzo a alto negativo che indica che sicuramente non è deceduto
#In questo momento tutti i possibili deceduti hanno campo deceduto ad 1, filtro tramite quel valore e calcolo la differenza
dfPF.loc[(dfPF.Deceduto==1), 'DifferenzaDeceduto'] = (pd.to_datetime(dfPF.query("Deceduto==1").DataPrimaNotifica) 
                                                     - pd.to_datetime(dfPF.query("Deceduto==1").DataDeceduto)).dt.days
#Ora dove la differenza è >=0 metto Deceduto a 1 altrimenti 0
dfPF.loc[(dfPF.DifferenzaDeceduto.astype(int)>=0),"Deceduto"]=1 
dfPF.loc[(dfPF.DifferenzaDeceduto.astype(int)<0),"Deceduto"]=0

#Realmente deceduti di cui si era conoscenza alla data di notifica del titolo: 296 titoli, droppo la colonna DifferenzaDeceduto
dfPF.drop(columns="DifferenzaDeceduto", inplace=True)


#I DATI SONO ORA PRONTI ALL'AGGREGAZIONE, SALVO IL DATASET CONTENENTE I TITOLI DELLE PERSONE FISICHE
save_dataset(dfPF)

"""





"""
--------------------------------------------AGGREGAZIONE E CLASSIFICAZIONE----------------------------------------------
#PREPARAZIONE:
1) DROP DELLE COLONNE INUTILIZZATE
    dfPF.drop(columns=["DataNascita","TipoPersonalità","PEC","DataEmissioneTitolo","DataPagamentoTotale","DataDeceduto"], inplace=True)
2) GET DUMMIES TipoCredito:
    dfPF = pd.concat([dfPF, pd.get_dummies(dfPF['TipoCredito'])], axis=1)
    dfPF.columns = [c.replace(' ', '_') for c in dfPF.columns] #tolgo lo spazio nei nomi
    dfPF.drop(columns=["TipoCredito"], inplace=True)
    for col in ['Servizi_Agricoli', 'Servizi_Amianto', 'Servizi_Cimiteriali', 'Servizi_Extratariffa_',
                'Servizi_Speciali', 'Servizi_Verde', 'Tariffa_Rifiuti', 'oneri_di_esazione']:
                 dfPF[col] = dfPF[col].astype('int64') #Per vedere già i campi come interi
3) SISTEMO COLONNE NumeroTitoliRecenti e TotaleTitoliRecenti, in modo che contengano il conteggio e la somma dei titoli 
    per lo stesso idAnagrafica alla stessa data di carico del titolo (Ruolo):
    dfPF["NumeroTitoliRecenti"]+=1
    dfPF["TotaleTitoliRecenti"]+=dfPF["ValoreTitolo"]
    
#AGGREGAZIONE DEI TITOLI APPARTENENTI ALLO STESSO DOCUMENTO DI CREDITO (=IDANAGRAFICA-DATACREDITO-DATAPRIMANOTIFICA)
#Strategia: raggruppo i titoli per idAnagrafica, DataCaricoTitolo e DataPrimaNotifica, e aggrego le colonne secondo
    i parametri scelti (ad esempio "sum" per ValoreTitolo e Pagato120Giorni)

#Parametri di aggregazione:
#Dove come metodo di aggregazione è specificato 'max' in realtà si mantiene l'unico valore di quella colonna presente
 nel gruppo, in quanto sono stati tutti resi uguali nella fase di pulizia precedente. 
 Vetustà è determinata dal minimo: titolo più vecchio del gruppo, i campi corrispondenti al tipo del titolo vengono
 sommati per indicare quanti e per che tipo di titoli erano presenti all'interno dello stesso documento di credito.
aggprm={'Telefono': 'max', 'IndirizzoResidenza': 'max', 'Cap': 'max', 'CittadinanzaItaliana' : 'max', 'Deceduto': 'max',
     'ValoreTitolo' : 'sum', 'Pagato120Giorni': 'sum', 'NumeroTitoliAperti': 'max', 'ImportoTitoliAperti': 'max', 
     'NumeroTitoliSaldati': 'max', 'ImportoTitoliSaldati': 'max', 'NumeroTitoliRecenti': 'max', 
     'TotaleTitoliRecenti': 'max', 'Eta': 'max', 'Vetusta': 'max', 'Servizi_Agricoli': 'sum', 'Servizi_Amianto': 'sum',
     'Servizi_Cimiteriali': 'sum', 'Servizi_Extratariffa_': 'sum', 'Servizi_Speciali': 'sum', 'Servizi_Verde': 'sum',
     'Tariffa_Rifiuti': 'sum', 'oneri_di_esazione': 'sum'}
#Procedo con l'aggregazione dei titoli appartenenti allo stesso docuemnto/credito:
dfPF=dfPF.groupby(["idAnagrafica","DataCaricoTitolo","DataPrimaNotifica"]).agg(aggprm)
dfPF=dfPF.reset_index()
#Totale: 23452 crediti per persone fisiche
#Totale: 15453 anagrafiche univoche per persone fisiche

#Calcolo la label di classe per ogni credito
dfPF['label']=dfPF['Pagato120Giorni']/dfPF["ValoreTitolo"]>=0.20
dfPF['label']=dfPF['label'].astype("int64")
#4365 crediti sono positivi, contro 19087 negativi, il dataset è decisamente imbalanciato in favore della classe negativa.
#Circa 18.6% classe positiva e 81.4% classe negativa.








"""








"""
------------ERRONEI?----------------------------------------------------------------------------------------------------

#Controllo anomalie per ruolo: quanti titoli con stesso contribuente e stessa data assunzione ruolo hanno numero titoli aperti o recenti differente
ruoloanagrafica=dfPF.groupby(["idAnagrafica","DataCaricoTitolo"])
testanagrafica=ruoloanagrafica.NumeroTitoliRecenti.nunique()>1              ------PROBABILMENTE SBAGLIATO, VEDI SOTTO---
testanagrafica=testanagrafica.reset_index() 
testanagrafica=pd.DataFrame({'idAnagrafica':testanagrafica["idAnagrafica"], 
                            'DataCaricoTitolo':testanagrafica["DataCaricoTitolo"],
                            "NumeroTitoliRecenti":testanagrafica["NumeroTitoliRecenti"]})
idlist=testanagrafica.query("NumeroTitoliRecenti==True").idAnagrafica
idlist = idlist.drop_duplicates()
idlist=idlist.values
#titoli appartenenti a queste 499 anagrafiche problematiche (3978):
dfPF[dfPF['idAnagrafica'].isin(idlist)]





#GESTIONE INDIRIZZI DIFFERENTI PER STESSA DATA DI ASSUNZIONE SU STESSA ANAGRAFICA
#Strategia proposta: raggruppare per anagrafica, datacaricotitolo e cambiare tutti i valori nella moda
dfPF['IndirizzoModa'] = dfPF.groupby(['idAnagrafica','DataCaricoTitolo'])["IndirizzoResidenza"].transform(lambda x: x.mode()[0])
#ORA E' DA SISTEMARE IL CAP, COSA CHE SI FARA' TRAMITE GEOCODING

#GESTIONE DEI TITOLI SENZA CAP, di cui uno è da eliminare perchè non ha nemmeno l'indirizzo
#Droppo l'anagrafica anomala senza indirizzo ne cap 
dfPF = dfPF[~(dfPF['IndirizzoResidenza']=="SCONOSCIUTO ALL'ANAGRAFE SCONOSCIUTO ALL'ANAGRAFE")]
#Imposto a zero il cap degli esteri e credo un nuovo campo binario : "Estero" impostato a 1 solo per quelli cambiati
dfPF[["Cap"]] = dfPF[["Cap"]].fillna(0)
#Aggiungo la nuova colonna estero
dfPF['Estero'] = 0                                                  -------ERRONEO, CI SONO ESTERI CON CAP
#Setto a 1 i titoli esteri (Cap=0)                                          BISOGNA GEOCODING---------------
dfPF.loc[dfPF['Cap']==0,'Estero']=1

-----------------------------------INDIVIDUAZIONE TITOLI ANOMALI RUOLI SPEZZATI-----------------------------------------

#Vista delle anagrafiche con ruoli spezzati (stessa data di carico e diverso numero titoli aperti)
grouped=dfPF.groupby(["idAnagrafica","DataCaricoTitolo","NumeroTitoliAperti"])
grouptitlesc=grouped.agg(['count'])
groupcumcount=grouptitlesc.groupby(level=[0, 1]).cumcount().to_frame()
groupcumcount=groupcumcount.reset_index()
groupcumcount.columns=["idAnagrafica","DataCaricoTitolo","NumeroTitoliAperti","cumcount"]
groupquery=groupcumcount.query("cumcount==1") #cumcount mette ad ogni riga appartenente a un gruppo un indice da 0, 
                                               se c'è almeno una riga che vale 1 su una coppia id-data allora significa
                                               che quel ruolo è stato spezzato
anagraficheanomale=np.unique(groupquery["idAnagrafica"]) #Numpy array contenenti tutte le anagrafiche che hanno almeno 
                                                          un ruolo spezzato, sarebbe possibile eliminare anche solo il 
                                                          ruolo prendendo idAnagrafica e DataCarico, e poi eliminando
                                                          tutti i titoli con quei 2 campi
dfPF[dfPF['idAnagrafica'].isin(anagraficheanomale.tolist())] #Per vedere tutti i titoli appartenenti ad anomalie 
                                                            (3930 dopo eliminazione aziende)
#Provo a ricavare idAnagrafica e DataCarico per identificare i ruoli piuttosto che le anagrafiche
groupquery["zip"]=list(zip(groupquery["idAnagrafica"],groupquery["DataCaricoTitolo"])) #colonna con tuple data-id 
dfPF["zip"]=list(zip(dfPF["idAnagrafica"],dfPF["DataCaricoTitolo"]))
titolianomali=groupquery["zip"].tolist()
dfPF[dfPF['zip'].isin(titolianomali)] #Per vedere tutti i titoli anomali (3087 dopo eliminazione aziende)
dfPF = dfPF[~dfPF['zip'].isin(titolianomali)]  #Per eliminare i titoli anomali dal dataframe (54555-3087=51468)
#Droppo colonna zip
dfPF.drop(columns="zip", inplace=True)

#UNA VOLTA RIMOSSI QUESTI TITOLI E QUELLI APPARTENENTI AD ANAGRAFICHE DI AZIENDE ANCHE SE PF (+1 senza indirizzo)
    RIMANGONO UN TOTALE DI 51468 TITOLI, AGGREGATI PER RUOLO/ANAGRAFICA SONO 22068

#CONTROLLO CHE TUTTI I TITOLI PER STESSA ANAGRAFICA E STESSA DATA ABBIANO LO STESSO NUMERO DI RECENTI
conteggio=dfPF.groupby(["idAnagrafica","DataCaricoTitolo"])["NumeroTitoliRecenti"].nunique()>0
conteggio=conteggio[conteggio].index.values #E' vuoto dunque non ci sono anomalie



"""

