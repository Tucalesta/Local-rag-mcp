Ecco la sequenza ordinata di comandi e passaggi necessari per replicare l'ambiente e far funzionare il sistema RAG su un'altra macchina:

---

### 1. Prerequisiti di sistema
- Assicurati di avere **Python 3.10+** installato.
- Installa e avvia **Ollama** sulla macchina di destinazione ([ollama.com](https://ollama.com)).

---

### 2. Download dei modelli Ollama necessari
Apri il terminale ed esegui i comandi per scaricare i modelli usati dal sistema:

```bash
ollama run nomic-embed-text
ollama run gemma3
ollama run llama3.2-vision
```
*(Una volta completato il download di ciascun modello puoi uscire digitando `/bye` o premendo `Ctrl + D`)*.

- `nomic-embed-text` calcola gli embedding di testo (documenti e query).
- `gemma3` genera le risposte alle domande (leggero, adatto anche a macchine con poca VRAM/GPU condivisa con altri usi).
- `llama3.2-vision` genera le didascalie delle immagini indicizzate (vedi sezione 5). Il nome del modello vision e' configurabile in `config.json` (`vision_model`), quindi puoi sostituirlo con un altro modello vision gia' scaricato senza modificare il codice.

---

### 3. Configurazione dell'Ambiente Virtuale Python
Spostati nella cartella del progetto (`local-rag-mcp`) ed attiva il virtual environment:

```cmd
python -m venv venv
.\venv\Scripts\Activate
```
*(Su Windows puoi anche fare semplicemente doppio clic sul file `activate_venv.bat`)*.

---

### 4. Installazione delle dipendenze Python
Installa le librerie necessarie tramite il file `requirements.txt`:

```cmd
pip install -r requirements.txt
```

*(In alternativa puoi installarle manualmente con: `pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama llama-index-vector-stores-chroma chromadb pypdf python-docx openpyxl mcp`)*.

---

### 5. Preparazione dei file e Indicizzazione (`ingest.py`)
1. Inserisci i documenti che desideri indicizzare nella cartella `docs/` (supporta `.pdf`, `.txt`, `.md`, `.docx`, `.xlsx`) e/o immagini (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`).
2. Configura i percorsi delle cartelle da scansionare in `config.json`.
3. Avvia (o aggiorna) l'indicizzazione:

```cmd
python ingest.py
```

*Opzioni disponibili in `config.json`:*

| Chiave | Default | Significato |
|---|---|---|
| `db_path` | `./chroma_db` | Percorso del database vettoriale. Puo' essere assoluto (es. su un altro disco) per spostare il db liberamente. |
| `manifest_path` | `./ingest_manifest.json` | Percorso del file manifest che tiene traccia dei file indicizzati. Puo' essere relativo o assoluto. |
| `res_dir` | `./res` | Percorso della cartella contenente i file GPX per il reverse geocoding delle foto. Puo' essere relativo o assoluto. |
| `llm_model` | `gemma3` | Modello Ollama usato da `mcp_server.py` per generare le risposte alle query (letto anche da `health_check`). |
| `vision_model` | `llama3.2-vision` | Modello Ollama usato da `ingest.py` per generare le didascalie delle immagini. |
| `gpx_max_distance_km` | `15` | Distanza massima (km) entro cui una foto con GPS viene associata a una localita' nel database GPX personale (cartella `res/`). |
| `caption_prompt_version` | `v3` | Versione del prompt usato per le didascalie (definite in `captioning.py`: `v1` = originale, `v2` = telegrafico a campi in italiano, `v2_en` = stesso in inglese, `v3` = come v2 con accenti corretti e righe vuote omesse). Le righe tipo "Soggetto: nessuno" e le parole chiave ripetute vengono comunque eliminate dopo la generazione. |
| `caption_max_side` | `768` | Lato massimo (px) a cui le foto vengono riscalate prima di essere inviate al modello vision. `0` = file originale. |
| `caption_options` | `{"temperature": 0.2, "num_predict": 400, "num_ctx": 2048}` | Opzioni di generazione passate a Ollama per le didascalie. `num_ctx` ridotto fa stare `qwen3-vl:8b-instruct` interamente in 8 GB di VRAM. |
| `caption_only` | `false` | Se `true` (oppure lanciando `ingest.py --caption-only`) genera solo le didascalie nella cache, senza toccare indice, manifest ed embedding. |
| `caption_timeout` | `180` | Timeout (secondi) per la didascalia di una singola foto. |
| `caption_cache_path` | `captions.jsonl` | File JSONL con le didascalie generate (una riga per foto/modello/prompt). Se relativo, e' risolto rispetto alla cartella del manifest. |
| `embed_model` | `nomic-embed-text` | Modello Ollama per gli embedding, usato da `ingest.py`, `mcp_server.py` e `query.py`. Cambiarlo richiede una nuova `collection_name` (vedi "Passaggio a bge-m3"). |
| `collection_name` | `local_docs` | Nome della collezione ChromaDB. Una collezione e' legata a un solo modello di embedding. |
| `embed_query_instruction`, `embed_text_instruction` | (assenti) | Solo per i modelli di embedding che li richiedono (es. `qwen3-embedding`): istruzione da anteporre alle ricerche / ai testi. Con `bge-m3` non servono. |
| `top_k_docs` | `3` | Passaggi di documenti restituiti da una ricerca (`search_local_docs`, `query.py`). |
| `top_k_photos` | `10` | Foto restituite da una ricerca (`find_photos`, `foto:` in `query.py`); massimo 50. Ogni ricerca puo' comunque indicare un valore diverso. |
| `force_reindex` | `false` | Vedi sotto: ricostruzione completa da zero. |
| per ogni voce di `directories` | `index_images: true` | Metti `false` su una cartella specifica per indicizzarne solo i documenti testuali, saltando le immagini. |

*Note su `ingest.py`:*
- **Indicizzazione incrementale**: dalla versione attuale, `ingest.py` NON cancella piu' il database ad ogni esecuzione. Confronta ogni file con un manifest salvato (`ingest_manifest.json`) e processa solo i file nuovi o modificati (stessa data di modifica + dimensione = invariato, saltato). I file rimossi dal disco vengono rimossi anche dall'indice. Se non trovi novita' vedrai il messaggio: `Nessuna modifica rispetto all'ultima indicizzazione: nulla da aggiornare.`
- **Forzare una reindicizzazione completa da zero**: aggiungi `"force_reindex": true` in `config.json` e rilancia `python ingest.py`. Ignora il manifest esistente e ricostruisce tutto (utile dopo aver cambiato modello di embedding, schema dei metadata, o se si sospetta un indice corrotto/inconsistente). Ricordati di rimetterlo su `false` (o toglierlo) subito dopo, altrimenti ogni lancio successivo ripartira' sempre da zero.
- Esclude automaticamente cartelle di sistema e il Cestino (es. `$RECYCLE.BIN`, `System Volume Information`) incontrate durante la scansione; le cartelle radice indicate esplicitamente in `config.json` (es. `E:/`) non vengono escluse per questo motivo anche se Windows le marca con l'attributo SYSTEM (comune sulle radici dei dischi).
- Le immagini troppo piccole o con pochi colori distinti (probabili icone/pittogrammi, non foto) vengono riconosciute automaticamente e saltate.
- Se una foto ha coordinate GPS, `ingest.py` cerca la localita' piu' vicina nei GPX di `res/` (vedi sezione 8) e la registra nei metadati (`location_name`, `region`) e nel testo indicizzato (riga "Luogo: ..."). Le foto con GPS ma senza una localita' trovata entro `gpx_max_distance_km` finiscono in `geocoding_gaps.json`, utile per capire se il database GPX ha delle lacune da colmare.
- **Cache delle didascalie**: ogni didascalia generata viene salvata in `caption_cache_path`. Con lo stesso file (data e dimensione invariate), lo stesso modello e la stessa versione di prompt viene riutilizzata senza richiamare il modello: quindi anche `force_reindex` o un cambio di modello embedding non fanno rigenerare le didascalie. Cambiando `vision_model` o `caption_prompt_version` le foto vengono invece ridescritte (le vecchie didascalie restano nel file).
- **Confronto tra modelli**: `python bench_captions.py` genera un report HTML (in `bench_out/`) con le didascalie di piu' modelli affiancate, tempi, uso della GPU e possibilita' di votarle. Con `--variants 1024 768:2048` confronta anche risoluzioni della foto e finestre di contesto diverse (utili se il modello non sta tutto in VRAM). Chiudere ComfyUI prima di lanciarlo.
- **Confronto tra embedding**: `python bench_embeddings.py --results bench_out/<data>/results.json --queries queries_test.txt` prende le didascalie di un banco di prova e misura quanto bene ogni modello di embedding ritrova le foto per le ricerche scritte in `queries_test.txt` (istruzioni nel file).
- **Modalita' solo didascalie**: `ingest.bat --caption-only` (o `python ingest.py --caption-only`) descrive tutte le foto e le salva nella cache, senza aprire il database vettoriale, senza aggiornare il manifest e senza fare embedding: cosi' il modello vision resta l'unico caricato in VRAM. Si puo' interrompere con Ctrl+C e riprendere: le foto gia' in cache vengono saltate. Ogni 25 foto scrive nel log l'avanzamento e il tempo residuo stimato; si ferma da sola dopo 5 errori consecutivi (Ollama spento, modello mancante). Finito, un normale `ingest.bat` indicizza tutto leggendo le didascalie dalla cache.
- **Formato dell'indice per le foto** (`INDEX_TEXT_VERSION` in `ingest.py`): il testo che viene trasformato in vettore contiene solo cio' che ha significato per la ricerca: `Luogo: ...` (se trovato), `Cartella: ...` (nomi delle cartelle sotto la radice scansionata, es. `2014 / Dolomiti`) e la didascalia. Data, coordinate, fotocamera, nome del file e percorso NON sono nel testo ma nei metadati: `date_taken`, `year` (numero), `date_ts`, `year_source` (`exif` oppure `cartella`, se la foto non ha una data EXIF valida e l'anno e' stato dedotto dal nome di una cartella), `gps_lat`, `gps_lon`, `location_name`, `region`, `location_km`, `camera`, `folder`, `path`, `file_name`, `type`. I metadati servono ai filtri di `find_photos` e vengono restituiti con ogni risultato. Se `INDEX_TEXT_VERSION` cambia, i file gia' indicizzati vengono reindicizzati (le didascalie restano in cache: costa solo l'embedding). La data dello scatto e' presa da `DateTimeOriginal` quando c'e'.
- **File rimossi e cartelle non raggiungibili**: a fine giro i file che il manifest conosceva ma che non si trovano piu' vengono tolti dall'indice e dal manifest. Se pero' una cartella radice di `directories` non e' raggiungibile (disco scollegato, rete spenta) oppure non contiene nessun file da indicizzare (punto di montaggio vuoto), i suoi file restano nell'indice e nel manifest e il log lo segnala con un avviso. Se l'hai svuotata di proposito, togli la cartella da `directories`: al giro successivo i suoi file verranno rimossi. Limite: se una cartella e' raggiungibile solo in parte (rete che cade a meta' scansione) il caso non viene riconosciuto.
- **Manutenzione della cache**: `ingest.bat --prune-cache` compatta `captions.jsonl` (una riga per foto/modello/prompt, via le righe illeggibili e le didascalie di file che non esistono piu'). Opzioni: `--dry-run` mostra cosa farebbe senza modificare nulla; `--only-current` tiene solo le didascalie del `vision_model` e del `caption_prompt_version` in uso; `--force` salta i controlli di sicurezza. Prima di riscrivere crea una copia `captions.jsonl.bak-<data>`. Le didascalie di file sotto una cartella radice non raggiungibile non vengono mai toccate; il comando rifiuta di procedere se la cache e' stata modificata da meno di 2 minuti (ingest in corso) o se oltre meta' delle didascalie sembra riferirsi a file spariti. Da lanciare con `ingest.py` fermo.
- Registra le operazioni nella cartella `logs/` con una conservazione di 7 giorni.

#### Passaggio a `bge-m3` (nuova collezione)
Il test degli embedding (`bench_embeddings.py`) ha dato `bge-m3` nettamente davanti a `nomic-embed-text` sulle ricerche in italiano (foto giusta al primo posto nell'89% dei casi contro il 56%). Cambiare embedding richiede una NUOVA collezione, perche' i vettori hanno dimensioni diverse, con il suo manifest. Finche' `config.json` non contiene le chiavi del punto 3, tutto continua a usare la collezione `local_docs` con `nomic-embed-text`, server MCP compreso.

1. `ollama pull bge-m3`
2. Descrivi tutte le foto: `ingest.bat --caption-only` (si puo' interrompere e riprendere).
3. In `config.json` aggiungi:
   ```json
   "embed_model": "bge-m3",
   "collection_name": "local_docs_bge_m3",
   "manifest_path": "E:/local_rag_db/manifest/ingest_manifest_bge-m3.json",
   ```
   (e togli il vecchio `manifest_path`).
4. Lancia `ingest.bat` senza `--caption-only`: legge le didascalie dalla cache e indicizza foto e documenti nella nuova collezione. Se il manifest elencasse file ma la collezione fosse vuota, ignora il manifest e reindicizza tutto da solo.
5. Riavvia il server MCP (o Claude Desktop) perche' rilegga la configurazione.

La vecchia collezione `local_docs` resta nel database ma non viene piu' usata; si puo' eliminare a mano quando non serve piu'.

---

### 6. Interrogazione da CLI (`query.py`)
Per fare domande ai tuoi documenti da riga di comando:

```cmd
python query.py
```
*(Digita `aiuto` per l'elenco dei comandi, `exit` o `quit` per uscire)*.

- **Domanda sui documenti**: scrivi la domanda; il modello locale (`llm_model`) risponde usando i passaggi trovati e mostra le fonti.
- **Foto**: `foto: castello nella nebbia | luogo=San Leo | anno=2014`. I filtri dopo `|` sono facoltativi: `luogo`, `regione`, `anno`, `da`, `a` (intervallo di anni), `fotocamera`, `vicino` (luogo di riferimento; per omonimi `Nome, Regione`), `raggio` (km, default 20), `n` (quante foto). Si puo' anche cercare solo con i filtri: `foto: | luogo=Cortina | anno=2014`.
- `panoramica`: anni, localita', regioni e fotocamere presenti nell'archivio foto.

---

### 7. Registrazione del Connettore MCP (`mcp_server.py`)
Il connettore MCP espone questi strumenti a client AI esterni (nessuno fa scrivere testo al modello locale, salvo `search_local_docs` con `answer=true`):

- `search_local_docs(question, top_k, answer)`: passaggi dei documenti (PDF, Word, Excel, TXT, MD) piu' pertinenti, con le fonti.
- `find_photos(query, location, region, year, year_from, year_to, camera, near, radius_km, top_k)`: cerca foto per significato della descrizione e/o con filtri esatti su luogo, anno, fotocamera e distanza da un luogo. Restituisce percorso, data, luogo, coordinate, fotocamera e descrizione. Se un filtro non corrisponde a nulla elenca i valori disponibili.
- `photo_archive_overview()`: anni, localita', regioni e fotocamere presenti (con il numero di foto).
- `health_check()`: stato di database, collezione e modelli Ollama.

#### **Opzione A: Registrazione per Claude Desktop**
Modifica (o crea) il file di configurazione in `%APPDATA%\Claude\claude_desktop_config.json` aggiungendo il server sotto `"mcpServers"`:

```json
{
  "mcpServers": {
    "local-rag": {
      "command": "C:\\percorso\\al\\tuo\\local-rag-mcp\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\percorso\\al\\tuo\\local-rag-mcp\\mcp_server.py"
      ]
    }
  }
}
```

#### **Opzione B: Registrazione per Gemini CLI / Antigravity**
Puoi registrarlo a livello di singolo progetto creando il file `.agents/mcp_config.json` all'interno della cartella di progetto:

```json
{
  "mcpServers": {
    "local-rag": {
      "command": "C:/percorso/al/tuo/local-rag-mcp/venv/Scripts/python.exe",
      "args": [
        "C:/percorso/al/tuo/local-rag-mcp/mcp_server.py"
      ]
    }
  }
}
```
Oppure a livello globale creando/modificando il file `%USERPROFILE%\.gemini\config\mcp_config.json`.

---

### 8. Database GPX personale per il geocoding delle foto (`res/`)
La cartella `res/` contiene file `.gpx` con **waypoint** (non tracce): ogni `<wpt>` e' una localita' con nome, e il nome del file GPX (es. `Toscana.gpx`) diventa la regione associata. E' un database creato a mano/da un altro progetto, volutamente incompleto (solo le coordinate che servivano di volta in volta) - per questo esiste `geocoding_gaps.json`, che segnala i casi scoperti invece di far finta che tutto sia coperto.

Per ampliare la copertura basta aggiungere `<wpt lat="..." lon="..."><name>Nome Localita'</name></wpt>` al file GPX della regione interessata (o crearne uno nuovo per una regione/paese non ancora presente) e rilanciare `ingest.py`.
