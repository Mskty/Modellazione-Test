from Persone_Fisiche.funzioni import *

# regex regular expression
"""
In questo file verrà elaborato il dataset riferito a sole Persone Fisiche, già pulito e classificato,
allo scopo di prepararlo all'utilizzo da parte dell'algoritmo di apprendimento per ottenere il modello di classificazione.
In particolare sarà necessario ricavare più informazioni possibili dai dati disponibili per ogni credito 
(una riga della tabella corrisponde a un credito) ed elaborare campi come IndirizzoResidenza e Cap al fine di 
renderli numerici e utilizzabili dagli algoritmi.

"""

# Load dataset
dfPF = load_raw_data("datasets/CRCPFclean.csv")

"""
# Elimino crediti dove sono presenti oneri di esazione in quanto questi titoli sono errati.
dfPF=dfPF[~dfPF.index.isin(dfPF.loc[(dfPF.oneri_di_esazione>0)].index)]
"""

# Aggrego le 8 colonne tipo titolo che non danno informazione in un unica colonna che indica quanti titoli sono nel
# credito

dfPF["TitoliCredito"]=dfPF["Servizi_Agricoli"] + dfPF["Servizi_Amianto"] + dfPF["Servizi_Cimiteriali"] + dfPF["Servizi_Extratariffa_"] \
                      + dfPF["Servizi_Speciali"] + dfPF["Servizi_Verde"] + dfPF["Tariffa_Rifiuti"]
# Droppo le colonne corrispondenti al tipo_titolo
dfPF.drop(columns=["Servizi_Agricoli","Servizi_Amianto","Servizi_Cimiteriali","Servizi_Extratariffa_","Servizi_Speciali","Servizi_Verde","Tariffa_Rifiuti"], inplace=True)
# Colonna nuovo cliente
# Strategia: viene aggiunta l'informazione, per ogni credito, se il contribuente corrispondente era nuovo al momento
# della presa in carico del titolo oppure no, questa informazione può essere estratta controllando che, per il credito,
# il valore dei campi NumeroTitoliAperti e NumeroTitoliSaldati sia esattamente zero.
dfPF["NuovoContribuente"] = (dfPF["NumeroTitoliSaldati"] == 0) & (dfPF["NumeroTitoliAperti"] == 0)
dfPF["NuovoContribuente"] = dfPF["NuovoContribuente"].astype("int64")

# Recupero cap da indirizzo con espressione regolare
dfPF["nuovo_Cap"]=dfPF["IndirizzoResidenza"].str.findall(r'(\d{5})').str[-1] # estrazione cap da indirizzo

# Recupero la provincia dall'indirizzo con espressione regolare
dfPF["Provincia"]=dfPF["IndirizzoResidenza"].str.split().str[-1] # Errato quando c'è una stringa che finisce come ' X TV', prende ' X ' ma non TV

# I crediti con nuovo_cap==0 o provincia=='EE' sono esteri, creo una nuova colonna per indicarlo
# E fillo i cap nulli (tutti esteri) con il valore 0
dfPF["Estero"]=0
dfPF.loc[(dfPF.nuovo_Cap=='00000') | (dfPF.Provincia=='EE'), "Estero"]=1
# Imposto il cap di tutti gli esteri a 00000
dfPF.loc[(dfPF.Estero==1), "nuovo_Cap"] = '00000'
# Rimangono 4 anomalie che non hanno cap e non sono esteri, le elimino
dfPF[["nuovo_Cap"]]=dfPF[["nuovo_Cap"]].fillna('00000')
# Imposto cap a nuovo cap
dfPF["Cap"]=dfPF["nuovo_Cap"]
dfPF["Cap"]=dfPF["Cap"].astype("int64")
# Droppo nuovo_Cap
dfPF.drop(columns="nuovo_Cap", inplace=True)

#TODO ANCHE SOLO CONVERTIRE ZIPCODE IN COORDINATE E' UN MIGLIORAMENTO

# Nuova colonna: ImportoTitoliAperti/importoTitoliSaldati, vale 0 per i nuovi contribuenti
# L'idea è che il valore del rapporto sia più alto per i contribuenti peggiori, se >1 allora ha più aperto che saldato
dfPF["RapportoImporto"]=0
dfPF.loc[(dfPF.NuovoContribuente!=1), "RapportoImporto"]=dfPF.loc[(dfPF.NuovoContribuente!=1)].ImportoTitoliAperti/dfPF.loc[(dfPF.NuovoContribuente!=1)].ImportoTitoliSaldati
inf=np.inf
dfPF.loc[(dfPF.RapportoImporto==inf),"RapportoImporto"]=-1
# Imposto tutti i contribuenti con rapporto infinito, ovvero ci sono titoli aperti e nessuno saldato al massimo dei rapporti
#615 circa
max_rapporto=1000 #oppure dfPF.RapportoImporto.max() che è 615
dfPF.loc[(dfPF.RapportoImporto==-1),"RapportoImporto"]= max_rapporto

# Importo Dovuto rispetto ad ancora da pagare dei titoli aperti dove presenti, varia da 0 a 1, dove è più basso è migliore (significa che ha già pagato tutto)
# Nella gran parte dei casi sarà 1 (ovverto il contribuente deve ancora pagare tutto il dovuto)
dfPF["RapportoDovutoAperti"]=0
dfPF.loc[(dfPF.NumeroTitoliAperti!=0), "RapportoDovutoAperti"]=dfPF.loc[(dfPF.NumeroTitoliAperti!=0)].ImportoTitoliAperti/dfPF.loc[(dfPF.NumeroTitoliAperti!=0)].DovutoTitoliAperti


"""
#VISUALIZZAZIONE?
#histogram 1 vs 0
plt.hist(x = [dfPF[dfPF['label']==1]['Vetusta'], dfPF[dfPF['label']==0]['Vetusta']],
         stacked=True, color = ['g','r'],label = ['Buono','Cattivo'])
plt.title('Histogramma della Vetusta per label')
plt.xlabel('Vetusta')
plt.ylabel('# di crediti')
plt.legend()
#Kdeplot
a = sns.FacetGrid(dfPF, hue = 'label', aspect=4 )
a.map(sns.kdeplot, 'Cap', shade= True )
a.set(xlim=(0 , dfPF['Cap'].max()))
a.add_legend()

#correlationheatmap

"""


"""
# Elimino le righe con Cap mancante o errato, fanno riferimento a crediti stranieri o ambigui
dfPF=dfPF[~dfPF.index.isin(dfPF.loc[(dfPF.nuovo_Cap!=dfPF.Cap)].index)]
# Vedo Cap come interi
dfPF["Cap"]=dfPF["nuovo_Cap"].astype("int64")
# Droppo colonna nuovo_cap
dfPF.drop(columns="nuovo_Cap", inplace=True)

"""
"""
# Fill dei cap mancanti con la mediana
median=dfPF[["nuovo_Cap"]].median()
dfPF[["nuovo_Cap"]]=dfPF[["nuovo_Cap"]].fillna(median) # mediana sui cap nulli
dfPF["nuovo_Cap"]=dfPF["nuovo_Cap"].astype("int64")
"""


"""
# Gestione IndirizzoResidenza e Cap
# Colonna latitudine e colonna longitudine
# Strategia: viene effettuato geocoding sull'indirizzo di residenza ed estratto latitudine-longitudine ma anche cap
# Viene utilizzata la libreria geopy

# Imposto geolocator di default (utilizza OpenStreetMap, che è gratuito a differenza di google maps)
geolocator = Nominatim(user_agent="myGeocoder")

# Imposto un interfallo di tempo tra una e l'altra richiesta di geolocalizzazione, questo perchè potrebbe dare errore per
# Denial of service a causa del numero elevato di richieste effettuate
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
# Creo la colonna con latitudine-longitudine-altitudine
start = time.perf_counter()
dfPF["Coordinate"] = dfPF["IndirizzoResidenza"].apply(geocode)
end= time.perf_counter() - start # richiede più di 23000 secondi ovvero 383 minuti ovvero 6 ore e mezzo
# Creo una tupla per recuperare i tre dati sincolarmente dalla colonna coordinate
dfPF["Punto"] = dfPF["Coordinate"].apply(lambda loc: tuple(loc.point) if loc else None)
# Splitto i 3 dati inclusi nelle tuple della colonna Punto e ottengo una colonna Latitudine e una Longitudine, viene
# anche creata la colonna corrispondente all'altitudine che però sarà probabilmente inutile dato che tale dato è
# raramente disponibile
dfPF[["Latitudine", "Longitudine", "Altitudine"]] = pd.DataFrame(dfPF["Punto"].tolist(), index=dfPF.index)

#Test su uno slice con le prime 50 righe del Dataframe
dfPFtest=dfPF.iloc[:50].copy()

#IN ALTERNATIVA AL GEOCODING: Sistemo la colonna Cap con regex su indirizzo
dfPF["nuovo_Cap"]=dfPF["IndirizzoResidenza"].str.extract(r'(\d{5}\-?\d{0,4})')

"""


"""
#OUTLIER DETECTION
upper_lim = dfPF['NumeroTitoliAperti'].quantile(.95)
lower_lim = dfPF['NumeroTitoliAperti'].quantile(.05)

dfPF[(dfPF['NumeroTitoliAperti'] >= upper_lim) | (dfPF['NumeroTitoliAperti'] <= lower_lim)]

#TEST VELOCE PER CLASSIFICATORE: CAP RECUPERATO DA INDIRIZZO CON REGEX, VALORI NULLI MESSI A MEDIANA, INDIRIZZO IGNORATO
dfPF["nuovo_Cap"]=dfPF["IndirizzoResidenza"].str.extract(r'(\d{5}\)')
median=dfPF[["nuovo_Cap"]].median()
dfPF[["nuovo_Cap"]]=dfPF[["nuovo_Cap"]].fillna(median)
dfPF["nuovo_Cap"]=dfPF["nuovo_Cap"].astype("int64")


"""