# NS-ARC Installation & Usage Guide

## 📋 Prerequisites

### System Requirements
- **OS**: Windows 10/11, Linux, macOS
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for training data (optional)
- **GPU**: CUDA-capable GPU for training (optional)

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Rust | 1.70+ | Core compression engine |
| Python | 3.8+ | Dashboard & neural features |
| Git | 2.0+ | Version control |

---

## 🔧 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ns-arc.git
cd ns-arc
```

### Step 2: Install Rust

**Windows (PowerShell):**
```powershell
winget install Rustlang.Rust.MSVC
# OR
Invoke-WebRequest -Uri https://win.rustup.rs -OutFile rustup-init.exe
./rustup-init.exe
```

**Linux/macOS:**
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

### Step 3: Install Python Dependencies

```bash
pip install streamlit torch torchvision zstandard pillow pandas
```

### Step 4: Build the Rust Core

```bash
cd ns-arc-demon
cargo build --release
```

Expected output:
```
   Compiling ns-arc-demon v0.1.0
    Finished `release` profile [optimized] target(s) in 30s
```

---

## 🚀 Usage

### Option A: Web Dashboard (Recommended)

```bash
# From project root
streamlit run ns_arc_dashboard.py
```

Open your browser to: **http://localhost:8501**

**Dashboard Modes:**
1. **Compress (Encode)** - Upload files to compress
2. **Decompress (Viewer)** - Restore compressed files
3. **Dedup Stats** - View deduplication statistics

### Option B: Command Line

```bash
# Compress a file
./ns-arc-demon/target/release/ns-arc-demon path/to/file.txt --out-dir ./dist

# Compress multiple files
./ns-arc-demon/target/release/ns-arc-demon file1.txt file2.png file3.log --out-dir ./dist
```

**Output:**
```json
{
  "file": "file.txt",
  "type": "Logs",
  "entropy": 4.23,
  "bytes_in": 102400,
  "bytes_out": 12500,
  "ratio_out_in": 0.12,
  "pct_saved": 87.8,
  "mode": "semantic_split",
  "hash_original": "a1b2c3...",
  "status": "OK"
}
```

---

## 🧠 Neural Compression (Optional)

### Train VQ-VAE for Image Compression

```bash
# Download training images (~10GB for 50,000 images)
python ns_arc_vision_train.py --download --images 50000

# Train the model (50 epochs, ~2-6 hours)
python ns_arc_vision_train.py --train --epochs 50
```

The trained model is saved to `ns_arc_vision.pt`.

---

## 📂 Output Files

| Extension | Description |
|-----------|-------------|
| `.nsarc` | NS-ARC compressed archive |
| `.nsv` | Neural-compressed visual file |
| `chunks.db` | SQLite deduplication database |

---

## ⚠️ Troubleshooting

### "Rust not found"
```bash
# Verify Rust installation
rustc --version
cargo --version

# If not found, add to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

### "Permission denied" (Linux/macOS)
```bash
chmod +x ns-arc-demon/target/release/ns-arc-demon
```

### "Python module not found"
```bash
pip install --upgrade streamlit torch torchvision zstandard pillow pandas
```

### Dashboard won't start
```bash
# Clear Streamlit cache
streamlit cache clear
streamlit run ns_arc_dashboard.py
```

---

## 🔄 Updating

```bash
git pull origin main
cd ns-arc-demon
cargo build --release
```

---

## 📊 Benchmarking

```bash
# Run on test corpus
./ns-arc-demon/target/release/ns-arc-demon ./testdata/* --out-dir ./dist

# Check performance
cat ./dist/*.json | jq '.encode_mb_s'
```

---

## 🛠️ Configuration

Create a `.nsarcrc` file in your project root:

```json
{
  "speed_mode": "balanced",
  "verify_mode": "chunk",
  "max_probes_per_file": 2,
  "min_gain_pct": 1.5,
  "zstd_level_text": 9,
  "zstd_level_binary": 5
}
```

---

## 📞 Support

- **Issues**: Open a GitHub issue
- **Discussions**: GitHub Discussions tab
- **Email**: your.email@example.com
