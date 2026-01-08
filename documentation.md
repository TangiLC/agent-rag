# DOCUMENTATION — RAG PoC (CPU local, monolithe)

## Objectif

Construire un assistant RAG simple, reproductible et “source-strict” avec outils :

- Corpus statique de PDFs (principalement scrap Wikipedia) en local
- Extraction texte → chunking overlap (avec coupe sur ponctuation) → embeddings → retrieval → rerank → génération LLM
- Réponses sans hallucination : toute affirmation doit être supportée par des extraits du corpus avec sources (doc/page/chunk)

Le système est volontairement minimaliste (PoC), sans framework agent complexe ni dépendances.

---

## Contraintes de conception

- Exécution CPU local
- Architecture monolithique (un orchestrateur, appel de quelques modules)
- Corpus principalement PDF
- Session only (pas de mémoire persistante côté agent à ce stade)

---

## Arborescence (actuelle)

- rag_agent.py
  - orchestrateur runtime : charge les artefacts, vérifie compatibilité, effectue retrieval
- build_index.py
  - “offline build” : lit PDFs, génère chunks, calcule embeddings, écrit les artefacts dans data/
- config.py
  - paramètres centraux : chemins, chunking, embeddings, artefacts
- tools/
  - pdf_loader.py : extraction texte page par page
  - chunking.py : découpe en chunks overlap + coupe à la ponctuation
  - embeddings.py : encodage E5-small des chunks
  - retrieval_bruteforce.py : retrieval brute-force (dot product sur embeddings normalisés)
  - vector_store_faiss.py : FAISS (index/save/load/search)
  - reranker.py : cross-encoder FR
  - llama-\_server.py : wrapper llama.cpp (Llama 3.2 3B Q5)
- rag_docs/
  - corpus PDF local (ignoré par git)
- data/
  - artefacts du build (embeddings/chunks/manifest) (ignoré par git)

---

## Choix techniques — bibliothèques

### Extraction PDF

- PyMuPDF (`pymupdf`, import `fitz`)
  - Motivation : robustesse sur PDFs “web” / générés (fonts CID/Type0, XObjects, etc.)
  - Fonctionnement : extraction page par page via `page.get_text("text")`

### Chunking

- Découpage en caractères (pas de tokenizer à ce stade)
- Overlap fixé (en caractères)
- Affinage “phrase-aware” :
  - on vise une taille cible CHUNK_SIZE_CHARS
  - on tente de couper sur ponctuation forte `[. ? ! \\n]`
  - fenêtre souple : [CHUNK_SOFT_MIN, CHUNK_SOFT_MAX]
  - fallback : coupe dure si pas de ponctuation dans la fenêtre
- Les chunks ne traversent pas les pages :
  - citations strictes et simples (doc/page/chunk_id)

### Embeddings

- Modèle : `intfloat/multilingual-e5-small`
- Lib : `sentence-transformers` + `torch` CPU
- Préfixes E5 :
  - passages : `passage: {chunk_text}`
  - requêtes : `query: {user_question}`
- Normalisation : `normalize_embeddings=True`
  - permet cosine similarity via dot product (et plus tard FAISS IP)

### Retrieval (PoC)

- Brute-force numpy
  - calcul : `scores = embeddings @ query_vec`
  - top-k via `argpartition` + `argsort`
- Pourquoi brute-force avant FAISS :
  - valider la pertinence et les artefacts (chunks/embeddings/mapping) sans complexité supplémentaire
  - éviter de déboguer qualité + indexation en même temps
  - suffisant pour volumes PoC (quelques milliers / dizaines de milliers de chunks)

### Reranking

- Cross-encoder FR : `antoinelouis/crossencoder-mMiniLMv2-L12-mmarcoFR`
- Stratégie :
  - retrieval dense top_k (20-40) → rerank → keep top_n (4-8)
- Objectif :
  - améliorer la précision et fournir un contexte court de haute qualité

### LLM

- Modèle : Llama 3.2 3B Instruct
- Quantization : Q5 (CPU)
- Runtime : llama.cpp (GGUF)
- Rôle :
  - analyse de la requête utilisateur, reformulation en json structuré
  - génération finale “strictement à partir du contexte”
  - aucune connaissance externe autorisée (PoC)

---

## Artefacts “build” (data/)

Écrits par `build_index.py`, consommés par `rag_agent.py`.

- chunks.jsonl
  - une ligne JSON par chunk :
    - chunk_id, doc_id, page, text
- chunk_ids.txt
  - liste alignée avec les embeddings (ordre strict)
- embeddings.npy
  - matrice float32 shape = (N_chunks, dim)
- manifest.json
  - métadonnées de build (date, paramètres chunking, modèle embedding, infos corpus)
  - utilisé au runtime pour vérifier compatibilité

---

## Logging

- Logging natif Python partout
- Format strict : `<timestamp>:INFO|message`
- `basicConfig()` appelé une seule fois dans les points d’entrée (`build_index.py`, `rag_agent.py`)
- Dans les modules tools : `logger = logging.getLogger(__name__)`
- Observabilité optionnelle par Lanfuse V3 (décorateur @observe sur les appels LLM)

---

## Guardrails PoC (principes)

Même si la couche “guardrails” sera enrichie plus tard, le PoC vise déjà :

- Réponse uniquement si on a du contexte issu du corpus
- Citations obligatoires et traçables :
  - doc_id + page + chunk_id
- “Je ne trouve pas dans le corpus” si retrieval/rerank insuffisant
- Pas de web, pas de connaissances externes

---

## Flux de bout en bout

### 1/ Build (offline)

1. Lire les PDFs depuis pdf_out/
2. Extraire texte page par page (PyMuPDF)
3. Chunker pages (overlap + coupe à la ponctuation)
4. Embeddings des chunks (E5-small)
5. Sauvegarder artefacts (chunks + embeddings + manifest)

Commande :

- python build_index.py

### 2/ Runtime (online)

1. Charger manifest + vérifier compatibilité config
2. Charger chunks + embeddings
3. (PoC actuel) retrieval brute-force top-k + afficher sources
4. (à venir) rerank FR puis LLM

Commande :

- python rag_agent.py

---

## Paramètres clés (config.py)

- CHUNK_SIZE_CHARS (ex 1000)
- CHUNK_OVERLAP_CHARS (ex 200)
- CHUNK_SIZE_FINE (ex 60 → fenêtre 940-1060)
- EMBEDDING_MODEL_NAME (E5-small)
- EMBEDDING_BATCH_SIZE (ex 32)
- chemins artefacts dans data/

---

## Roadmap

1. Reranker FR au-dessus du brute-force (valider précision)
2. FAISS (drop-in, sans changer le reste)
3. LLM llama.cpp (Llama 3.2 3B Q5) + prompt “source strict”
4. Post-check minimal : refuser si la réponse contient des phrases non sourcées
5. Tests adversariaux (hors corpus, paraphrases, pièges)

---

## Notes de simplicité

- Pas de classes complexes : fonctions + dataclasses suffisantes
- Pas de framework agent : router contrôlé par code
- Pas d’incrémental pour le build en PoC (rebuild complet)
- Chaque étape est testable indépendamment (build, chunking, embeddings, retrieval)
