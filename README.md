# PoC Agent RAG et tools

![Python](https://img.shields.io/badge/Python-3.12-blue)
![llama.cpp](https://img.shields.io/badge/llama.cpp-local-informational)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-success)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-multilingual-orange)
![Langfuse](https://img.shields.io/badge/Langfuse-observability_optional-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Sommaire

- [Présentation](#présentation)
- [Principes de conception](#principes-de-conception)
- [Architecture du projet](#architecture-du-projet)
- [Pipeline RAG](#pipeline-rag)
- [Outils intégrés](#outils-intégrés)
- [LLM et exécution](#llm-et-exécution)
- [Observabilité](#observabilité)
- [Configuration](#configuration)
- [Frontend Streamlit](#frontend-streamlit)
- [Installation et démarrage](#installation-et-démarrage)
- [Limitations connues](#limitations-connues)
- [Pistes d’évolution](#pistes-dévolution)
- [Licence](#licence)

---

## Présentation

Ce projet est un **PoC d’agent conversationnel RAG** (Retrieval-Augmented Generation) exécuté **localement**, combinant :

- un orchestrateur Python minimaliste
- un LLM servi via `llama.cpp`
- une recherche documentaire vectorielle (FAISS)
- des outils spécialisés (géographiques, calculatoires, etc.)
- une observabilité optionnelle via Langfuse

Le projet vise avant tout la **compréhension du pipeline**, la **maîtrise du flux** et la **lisibilité du code**, sans dépendre d’un framework lourd.
La phase d'initialisation (chunking, embedding) est gérée par : [`build_index.py`](build_index.py)
La phase d'orchestration (main entrée) est gérée par [`rag_agent.py`](rag_agent.py)

---

## Principes de conception

- Pipeline explicite, lisible, traçable
- Aucune dépendance cloud obligatoire
- Séparation stricte des responsabilités
- Fallbacks clairs (RAG → outils → LLM)
- Observabilité non bloquante
- Projet pensé comme un laboratoire, pas comme un produit

---

## Architecture du projet

Rôle central : `rag_agent_dialog.py`, qui orchestre l’ensemble du cycle question → réponse.

Schéma simplifié :

```
Utilisateur ———> rag_agent_dialog.py
                  |--> Détection outil
                  | |
                  | +--> tools/geo_tools.py
                  |
                  |--> Pipeline RAG
                  | |
                  | +--> embeddings
                  | +--> FAISS
                  | +--> reranking
                  |
                  |--> LLM (llama.cpp HTTP)
                  |
Réponse finale <——+
```

Les dossiers [`tools/`](tools/) regroupent les briques fonctionnelles indépendantes.

- Configuration : [`config.py`](config.py)
- Exemple de variables d’environnement à personnaliser : [`.env.template`](.env.template)

---

## Pipeline RAG

Le pipeline suit les étapes suivantes :

- Chargement des documents (PDF)
- Découpage en pages puis en chunks
- Génération d’embeddings multilingues
- Indexation FAISS
- Recherche top-k
- Reranking (activé)
- Garde-fou par score minimal
- Décision finale :

  - réponse basée sur le corpus
  - ou fallback outil
  - ou fallback LLM pur

- Embeddings : [`tools/embeddings.py`](tools/embeddings.py)
- Chunking : [`tools/chunking.py`](tools/chunking.py)

---

## Outils intégrés

Le projet intègre des outils appelables dynamiquement par l’agent, par exemple :

- calcul de distances géographiques
- durées de trajets ou de vols théoriques
- calculs déterministes simples

Les outils sont pensés pour être facilement ajoutés, sans impacter le reste du pipeline.

---

## LLM et exécution

- Le LLM est servi via **llama.cpp**
- Exécution en **serveur HTTP local**
- Appels synchrones depuis l’orchestrateur
- Gestion explicite des échecs (sortie vide, indisponibilité)

Technologie :
https://github.com/ggerganov/llama.cpp

---

## Observabilité

L’observabilité est **optionnelle** et basée sur **Langfuse v3.x**.

- Activation via variables d’environnement
- Utilisation des décorateurs `@observe`
- Aucune dépendance forte dans le code métier
- Désactivation transparente si Langfuse est absent

Technologie :
https://langfuse.com

---

## Configuration

- Un fichier `.env.template` est fourni dans le dépôt (exemple de configuration, clés Langfuse)
- Le fichier `config.py` centralise les paramètres applicatifs :
  - seuils RAG
  - configuration LLM
  - activation des composants

Aucune clé sensible n’est versionnée.

---

## Frontend Streamlit

Le projet inclut un frontend minimal via Streamlit pour interagir avec l’agent en mode chat.

Lancement (depuis le terminal, à la racine du repo) :

```bash
streamlit run streamlit/app.py
```

---

## Installation et démarrage

Prérequis :

- Python 3.12
- llama.cpp compilé localement
- Modèle GGUF compatible

Étapes générales :

- créer l’environnement Python
- installer les dépendances
- lancer le serveur llama.cpp
- démarrer l’agent via l’orchestrateur

---

## Limitations connues

- Pas de mémoire long-terme persistante
- Pipeline séquentiel (pas de parallélisme avancé)
- Qualité fortement dépendante du corpus
- Pas conçu pour un usage multi-utilisateur

---

## Pistes d’évolution

To-do potentiel :

- [ ] indexation incrémentale
- [ ] reranker GPU optionnel
- [ ] mémoire conversationnelle
- [ ] nouveaux outils spécialisés
- [ ] visualisation des traces

---

## Licence

Ce projet est distribué sous **licence MIT**.

Le texte complet de la licence est disponible dans le fichier :
[`LICENSE`](LICENSE)
