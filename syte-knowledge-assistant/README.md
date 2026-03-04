# Syte Knowledge Assistant

> RAG-System mit Qdrant, LLM-Integration und Chat-Interface.

---

## Phase 1: Setup - Schritt-fuer-Schritt

### Voraussetzungen

| Tool            | Version   | Pruefen mit              | Installieren                                      |
|-----------------|-----------|--------------------------|---------------------------------------------------|
| Python          | 3.11+     | `python3 --version`      | python.org                                        |
| Docker          | 24+       | `docker --version`       | docs.docker.com/get-docker/                       |
| Docker Compose  | v2+       | `docker compose version` | (kommt mit Docker Desktop)                        |
| Git             | 2.x       | `git --version`          | git-scm.com                                       |

---

### Schritt 1: Repository klonen / Projekt initialisieren

```bash
cd ~/projects
cd syte-knowledge-assistant
```

### Schritt 2: Python Virtual Environment erstellen

```bash
python3 -m venv .venv

# Aktivieren (Linux/Mac):
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

which python   # sollte .venv/bin/python zeigen
python --version  # sollte 3.11+ sein
```

### Schritt 3: Python-Abhaengigkeiten installieren

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Falls NVIDIA GPU vorhanden:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Schritt 4: Qdrant starten (Docker)

```bash
docker compose up -d
docker compose ps   # Sollte: syte-qdrant ... running

# Dashboard: http://localhost:6333/dashboard
```

### Schritt 5: API-Keys konfigurieren

```bash
cp .env.template .env
nano .env   # oder: code .env
```

Eintragen:
```
ANTHROPIC_API_KEY=sk-ant-api03-dein-key-hier
```

### Schritt 6: Embedding-Modell herunterladen

```bash
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)
print(f'Model loaded! Dim: {model.get_sentence_embedding_dimension()}')
"
```

Falls Download fehlschlaegt, Alternative in settings.yaml:
```yaml
embedding:
  model_name: "BAAI/bge-m3"
```

### Schritt 7: Setup verifizieren

```bash
python scripts/verify_setup.py
```

### Schritt 8: Test-Dokumente vorbereiten

```
data/test/relevant/     <-- Soll-Dokumente hier
data/test/distractors/  <-- Stoerdokumente hier
```

## Nuetzliche Befehle

```bash
docker compose up -d       # Qdrant starten
docker compose down        # Qdrant stoppen
docker compose logs -f     # Qdrant Logs
source .venv/bin/activate  # Python env aktivieren
python scripts/verify_setup.py  # Setup pruefen
```
