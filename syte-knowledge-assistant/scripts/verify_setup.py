#!/usr/bin/env python3
"""
Syte Knowledge Assistant — Phase 1 Setup Verification
======================================================
Run this script after completing the setup steps to verify
everything is correctly configured before moving to Phase 2.

Usage:
    python scripts/verify_setup.py
"""

import sys
import os
import importlib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Pretty output helpers
# ============================================================
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0
warnings = 0


def check(name: str, condition: bool, error_msg: str = "", warn_only: bool = False):
    global passed, failed, warnings
    if condition:
        print(f"  {GREEN}✓{RESET} {name}")
        passed += 1
    elif warn_only:
        print(f"  {YELLOW}⚠{RESET} {name} — {error_msg}")
        warnings += 1
    else:
        print(f"  {RED}✗{RESET} {name} — {error_msg}")
        failed += 1


def section(title: str):
    print(f"\n{BLUE}{BOLD}{'='*60}{RESET}")
    print(f"{BLUE}{BOLD}  {title}{RESET}")
    print(f"{BLUE}{BOLD}{'='*60}{RESET}")


# ============================================================
# 1. Python Environment
# ============================================================
section("1. Python Environment")

check(
    f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    sys.version_info >= (3, 11),
    f"Python 3.11+ required, you have {sys.version_info.major}.{sys.version_info.minor}"
)

# Check critical packages
critical_packages = {
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
    "qdrant_client": "qdrant-client",
    "torch": "torch",
    "transformers": "transformers",
    "sentence_transformers": "sentence-transformers",
    "langchain_text_splitters": "langchain-text-splitters",
    "anthropic": "anthropic",
    "loguru": "loguru",
}

for import_name, pip_name in critical_packages.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "installed")
        check(f"{pip_name} ({version})", True)
    except ImportError:
        check(f"{pip_name}", False, f"pip install {pip_name}")

# Optional packages
optional_packages = {
    "chainlit": "chainlit",
    "docling": "docling",
    "rich": "rich",
    "pandas": "pandas",
}

for import_name, pip_name in optional_packages.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "installed")
        check(f"{pip_name} ({version})", True)
    except ImportError:
        check(f"{pip_name}", False, f"pip install {pip_name} — needed for later phases", warn_only=True)


# ============================================================
# 2. Project Structure
# ============================================================
section("2. Project Structure")

required_dirs = [
    "config",
    "config/prompts",
    "src",
    "src/ingestion",
    "src/retrieval",
    "src/generation",
    "src/interface",
    "src/evaluation",
    "data",
    "data/inbox",
    "data/indexed",
    "data/test",
    "data/test/relevant",
    "data/test/distractors",
]

for d in required_dirs:
    path = PROJECT_ROOT / d
    check(f"Directory: {d}/", path.is_dir(), f"mkdir -p {d}")

required_files = [
    "config/settings.yaml",
    "docker-compose.yml",
    "requirements.txt",
]

for f in required_files:
    path = PROJECT_ROOT / f
    check(f"File: {f}", path.is_file(), f"File missing: {f}")

# .env check
env_path = PROJECT_ROOT / ".env"
check(
    "File: .env",
    env_path.is_file(),
    "Copy .env.template to .env and add your API keys",
    warn_only=True
)


# ============================================================
# 3. Configuration
# ============================================================
section("3. Configuration Loading")

try:
    from config import load_config, Settings
    config = load_config()
    settings = Settings(config)
    check("settings.yaml loads successfully", True)
    check(f"Qdrant host: {settings.qdrant_host}", True)
    check(f"Embedding model: {settings.embedding_model}", True)
    check(f"LLM provider: {settings.llm_provider}", True)
    check(f"Chunk size: {settings.chunk_size} tokens", True)
    check(f"Collection: {settings.qdrant_collection}", True)
except Exception as e:
    check("settings.yaml loads successfully", False, str(e))


# ============================================================
# 4. API Keys
# ============================================================
section("4. API Keys")

anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
check(
    "ANTHROPIC_API_KEY set",
    bool(anthropic_key) and anthropic_key != "sk-ant-your-key-here",
    "Set ANTHROPIC_API_KEY in .env (or skip if using local LLM only)",
    warn_only=True
)


# ============================================================
# 5. Qdrant Connection
# ============================================================
section("5. Qdrant Connection")

try:
    from qdrant_client import QdrantClient
    client = QdrantClient(host="localhost", port=6333, timeout=5)
    # Try to get collections (proves connection works)
    collections = client.get_collections()
    check("Qdrant is running and reachable", True)
    check(f"Collections found: {len(collections.collections)}", True)
except Exception as e:
    error_str = str(e)
    if "Connection refused" in error_str or "connect" in error_str.lower():
        check("Qdrant is running", False, "Start Qdrant: docker compose up -d")
    else:
        check("Qdrant connection", False, error_str)


# ============================================================
# 6. Embedding Model (Quick Load Test)
# ============================================================
section("6. Embedding Model (Download Check)")

try:
    from sentence_transformers import SentenceTransformer

    model_name = settings.embedding_model if 'settings' in dir() else "Qwen/Qwen3-Embedding-0.6B"
    print(f"  ... Checking if model '{model_name}' is available (this may download ~1.2GB on first run)...")

    # Just check if we can instantiate — don't run inference yet
    # Use trust_remote_code for Qwen models
    model = SentenceTransformer(model_name, trust_remote_code=True)
    emb_dim = model.get_sentence_embedding_dimension()
    check(f"Embedding model loaded: {model_name} (dim={emb_dim})", True)

    # Quick test embedding
    test_emb = model.encode(["Test query for Syte Knowledge Assistant"])
    check(f"Test embedding generated: shape={test_emb.shape}", test_emb.shape[1] > 0)

except Exception as e:
    check(f"Embedding model", False, f"{e}\nTry: pip install sentence-transformers torch")


# ============================================================
# 7. LLM Connection Test
# ============================================================
section("7. LLM Connection Test")

if anthropic_key and anthropic_key != "sk-ant-your-key-here":
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say 'Syte Assistant ready' in exactly those words."}]
        )
        reply = response.content[0].text
        check(f"Claude API works: '{reply[:50]}'", True)
    except Exception as e:
        check("Claude API connection", False, str(e))
else:
    check("Claude API test", False, "ANTHROPIC_API_KEY not set — skipping", warn_only=True)


# ============================================================
# Summary
# ============================================================
section("SUMMARY")
total = passed + failed + warnings
print(f"\n  {GREEN}Passed:   {passed}/{total}{RESET}")
if warnings > 0:
    print(f"  {YELLOW}Warnings: {warnings}/{total}{RESET}")
if failed > 0:
    print(f"  {RED}Failed:   {failed}/{total}{RESET}")
    print(f"\n  {RED}{BOLD}⚠ Fix the failed checks before proceeding to Phase 2.{RESET}")
else:
    print(f"\n  {GREEN}{BOLD}✓ All critical checks passed! Ready for Phase 2.{RESET}")

print()
sys.exit(1 if failed > 0 else 0)
