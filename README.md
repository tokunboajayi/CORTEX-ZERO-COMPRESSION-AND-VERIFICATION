# Algorithim - Unified AI Systems

This repository contains two core AI systems:

1. **HALT-NN** - Evidence-Grounded Anti-Hallucination System
2. **NS-ARC** - Neural-Symbolic Adaptive Resonance Compressor

---

# HALT-NN: Evidence-Grounded Anti-Hallucination System

**Cortex Zero Core** combines **Jøsang Subjective Logic** with a **7-Phase Truth Pipeline** to minimize hallucinations by forcing evidence-grounded answering.

## Core Principle

> **No non-trivial factual claim without mapped evidence and a verified support score.**

Unsupported claims **CANNOT** be stated as facts. The system will abstain or present uncertainty.

## Algorithm Components

### 1. Jøsang Subjective Logic (Original Core)

```
Opinion = (Belief, Disbelief, Uncertainty)  where b + d + u = 1.0
```

| Operator | Purpose |
|----------|---------|
| **Discount** | Scale opinion by source reliability |
| **Consensus** | Fuse multiple opinions mathematically |
| **ESS** | Epistemic Stability Score for truth confidence |

### 2. HALT-NN Pipeline (7 Phases)

```
Query → Intent → Claims → Evidence → NLI Graph → Verify → Generate → Audit
```

| Phase | Action |
|-------|--------|
| 1. Intent | Classify query type, detect constraints |
| 2. Decompose | Break into atomic claims (MUST_CITE, DERIVATION, etc.) |
| 3. Retrieve | Find relevant evidence from sources |
| 4. Graph | Build Claim↔Evidence links with NLI scoring |
| 5. Verify | **KILL SWITCH** - mark claims SUPPORTED/UNSUPPORTED/DISPUTED |
| 6. Generate | Emit ONLY supported claims with citations |
| 7. Calibrate | Compute confidence, suggest improvements |

## Quick Start (HALT-NN)

```python
from cortex_zero_core import run_halt_pipeline, HaltEvidence, SourceTier

evidence = [
    HaltEvidence.create("Python is a programming language.", "wiki", SourceTier.TIER_A),
]

audit = run_halt_pipeline("What is Python?", evidence)
print(audit.answer_text)         # "[SUPPORTED] What is Python? [Source: wiki]"
print(audit.overall_confidence)  # 0.94
```

## HALT-NN Files

| File | Purpose |
|------|---------|
| `cortex_zero_core/models.py` | Opinion, HaltClaim, HaltEvidence, AnswerAudit |
| `cortex_zero_core/logic.py` | Jøsang operators (discount, consensus, ESS) |
| `cortex_zero_core/halt_pipeline.py` | 7-phase verification pipeline |
| `api.py` | FastAPI REST wrapper |

---

# 👹 NS-ARC: Neural-Symbolic Adaptive Resonance Compressor

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**NS-ARC** is a next-generation lossless compression system that combines:
- 🦀 **Rust Core** - High-performance compression engine
- 🧠 **Neural AI** - Learned entropy modeling & VQ-VAE image compression
- 📊 **Smart Routing** - File-type aware compression selection
- 🗄️ **Corpus Dedup** - SQLite-based chunk deduplication

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Semantic Router** | Automatically detects file types (Image/Video/Code/Logs/Binary) |
| **Mode Selection** | Store, Zstd, Format-Lossless, Semantic Split, Corpus Resonance |
| **AI Features** | Neural Codec Selector, Semantic Tokenizer, VQ-VAE |
| **Deduplication** | Content-addressable storage with SQLite chunk index |
| **Dashboard** | Streamlit web UI for compression & decompression |

## 🚀 Quick Start (NS-ARC)

```bash
# Build the Rust core
cd ns-arc-demon
cargo build --release

# Run the dashboard
cd ..
streamlit run ns_arc_dashboard.py
```

## 🎯 Compression Modes

| Mode | Use Case | Expected Ratio |
|------|----------|----------------|
| **Store** | Pre-compressed (PNG, ZIP) | 1.0x |
| **Zstd** | General binary | 2-5x |
| **Format-Lossless** | PNG → oxipng optimization | 1.1-1.3x |
| **Semantic Split** | Logs/JSON/Code | 5-20x |
| **Corpus Resonance** | Dedup across files | 2-10x |
| **VQ-VAE** (lossy) | Images with quality tradeoff | 10-50x |

## NS-ARC Files

| File | Purpose |
|------|---------|
| `ns-arc-demon/` | Rust compression core |
| `ns_arc_dashboard.py` | Streamlit web UI |
| `ns_arc_vision.py` | VQ-VAE neural codec |
| `ns_arc_specialized.py` | Specialized algorithms |
| `ert_core.py` | ERT Python prototype |

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
