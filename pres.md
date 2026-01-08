# Présentation agent-rag

## 1. Contexte et objectif du projet

Construire un agent RAG local, compréhensible de bout en bout, sans framework agent lourd, capable de :

- interroger un corpus documentaire
- utiliser des outils déterministes
- fallback proprement vers un LLM
- rester observable et déboguable

Points de cadrage :

- PoC personnel / laboratoire
- priorité à la maîtrise du pipeline plutôt qu’à la performance brute
- pas de dépendance cloud obligatoire

## 2. RAG : agentique ou workflow ?

Choix assumé d’un workflow contrôlé, plutôt qu’un agent “auto-réflexif”.

- Le RAG est un pipeline : ingestion → embeddings → retrieval → reranking → génération
- La “part agentique” est limitée au routage :
  - RAG
  - outils
  - LLM pur (fallback)

Pourquoi ce choix :

- moins de magie
- plus simple à déboguer
- plus prédictible pour un PoC

## 3. L’agent : orchestration centrale

Un seul agent, un orchestrateur Python, qui décide quoi faire.

Fonctionnement :

- entrée : question utilisateur
- décisions successives :
  1. est-ce qu’un outil est pertinent ?
  2. sinon, RAG sur le corpus
  3. si insuffisant, fallback LLM

Note :

- pas de “planification multi-étapes”
- décisions explicites dans le code

## 4. Les outils

Les outils sont déterministes, simples, et apportent de la valeur là où le RAG n’en a pas.

Exemples (2 outils) :

- outil géographique : coordonnées GPS / distance
- outil de calcul exact : durée à vol d’oiseau

Pourquoi des outils :

- éviter d’interroger le corpus pour du calcul
- éviter que le LLM “invente” des chiffres
- pipeline plus robuste

## 5. Difficultés rencontrées

### 5.1 Chunking : taille et overlap

Problème réel :

- découpage en caractères + overlap
- cas limite : dernier chunk dont la fin == fin du document
- risque : boucle infinie si la logique n’avance plus

Leçon :

- garantir une progression stricte (index de début/fin)
- gérer explicitement le dernier chunk
- ajouter des garde-fous (ex : break si next_start == start)

### 5.2 Observabilité : Langfuse v3

Problème :

- API modifiée entre v2 et v3
- anciennes méthodes invalides (ex : traces)
- passage aux décorateurs `@observe`

Difficulté :

- doc/versions pas toujours alignées
- adapter l’architecture existante sans la polluer

Leçon :

- observabilité doit rester optionnelle
- ne jamais coupler le cœur métier à l’outil

## 6. Choix techniques et architecture

Chaque brique est choisie pour sa simplicité et sa lisibilité.
Développement brique par brique, en agilité, avec test à chaque étape.

1. création du corpus RAG
   scrapping / chunking / embedding / index
2. création des tools
   coordonnées GPS / calcul de distance / calcul de durée
3. logging des étapes clés
   logging natif python sortie en terminal
4. Mise en place du LLM
   build serveur llama.cpp HTTP en local
5. Mise en place de la génération de réponse
   LLM avec guardrails
6. Mise en place de l'interprétation de la requête utilisateur
   LLM avec réponse structurée json, routage
7. Observabilité Langfuse
8. Front <à venir>

Base :

- Python 3.12
- exécution CPU locale

Librairies / briques :

- PyMuPDF : extraction PDF robuste
- SentenceTransformers (E5 multilingue) : embeddings
- FAISS : recherche vectorielle
- reranking francophone léger : "antoinelouis/crossencoder-mMiniLMv2-L12-mmarcoFR"
- llama.cpp : LLM local via serveur HTTP
- Langfuse : observabilité optionnelle

Architecture :

- orchestrateur unique (ex : `rag_agent_dialog.py`)
- modules outils isolés (ex : `tools/`)
- pas de framework agent (LangChain, etc.)

## 7. Observabilité : brancher sans coupler

Langfuse / Langsmith sont des greffons, pas des piliers.

Principe :

- activation via `.env`
- décorateurs autour des fonctions clés
- si l’outil est absent → le système fonctionne quand même

Bénéfices :

- traçage des décisions
- débogage du pipeline
- lecture simple des latences et erreurs

## 8. Conclusion

- projet volontairement simple
- pipeline maîtrisé
- agent explicite (routage RAG / outils / LLM)
- bonne base pour itérations futures (sans promesses) :
  - nouveaux outils
  - mémoire conversationnelle optionnelle
  - amélioration du retrieval/rerank/timeout
