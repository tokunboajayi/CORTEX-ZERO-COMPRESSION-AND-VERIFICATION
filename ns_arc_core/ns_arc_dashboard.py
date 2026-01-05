import streamlit as st
import subprocess
import os
import tempfile
import time
import json
from PIL import Image, ImageOps

# Raise PIL limit for large images (but still protect against extremes)
Image.MAX_IMAGE_PIXELS = 600_000_000  # 600M pixels max

def safe_st_image(path, caption="Image", max_preview_side=2048):
    """
    Safely display an image in Streamlit.
    For huge images, creates a downscaled preview to avoid decompression bomb errors.
    """
    try:
        # Check image size first
        with Image.open(path) as im:
            width, height = im.size
            total_pixels = width * height
            
            if total_pixels > 50_000_000:  # 50M pixels = ~7000x7000
                # Create preview for huge images
                st.warning(f"Image is huge ({width}x{height} = {total_pixels:,} pixels). Showing preview.")
                im = ImageOps.exif_transpose(im)
                im.thumbnail((max_preview_side, max_preview_side))
                st.image(im, caption=f"{caption} (Preview: {max_preview_side}px max)")
            else:
                st.image(path, caption=caption)
    except Exception as e:
        st.error(f"Could not display image: {e}")
        st.info("Image saved to disk. Use the download button to view it.")


# --- CONFIG ---
DEMON_CORE_DIR = os.path.join(os.getcwd(), "ns-arc-demon")
CARGO_PATH = r"C:\Users\olato\.cargo\bin\cargo.exe"
CHUNKS_DB_PATH = os.path.join(DEMON_CORE_DIR, "dist", "chunks.db")

st.set_page_config(page_title="NS-ARC Demon Dashboard", page_icon="👹", layout="wide")

# --- HEADER ---
st.title("👹 NS-ARC: Demon Core Dashboard")
st.markdown("**Neural-Symbolic Adaptive Resonance Compressor**")

# --- SIDEBAR: STATUS ---
st.sidebar.header("System Status")
if os.path.exists(CARGO_PATH):
    st.sidebar.success("Rust Toolchain: DETECTED")
else:
    st.sidebar.error("Rust Toolchain: NOT FOUND")

if os.path.exists(os.path.join(DEMON_CORE_DIR, "src/main.rs")):
    st.sidebar.success("Demon Core: ONLINE")
else:
    st.sidebar.error("Demon Core: OFFLINE")

# Show dedup stats if database exists
if os.path.exists(CHUNKS_DB_PATH):
    try:
        import sqlite3
        conn = sqlite3.connect(CHUNKS_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COALESCE(SUM(ref_count), 0), COALESCE(SUM(size), 0) FROM chunks")
        row = cursor.fetchone()
        total_chunks, total_refs, unique_bytes = row
        dedup_ratio = total_refs / total_chunks if total_chunks > 0 else 1.0
        conn.close()
        
        st.sidebar.divider()
        st.sidebar.header("📊 Dedup Stats")
        st.sidebar.metric("Unique Chunks", f"{total_chunks:,}")
        st.sidebar.metric("Total Refs", f"{total_refs:,}")
        st.sidebar.metric("Stored Bytes", f"{unique_bytes:,} B")
        st.sidebar.metric("Dedup Ratio", f"{dedup_ratio:.2f}x")
    except Exception as e:
        st.sidebar.warning(f"Dedup DB error: {e}")

# --- Mode Selection ---
mode = st.sidebar.radio("Mode", ["Compress (Encode)", "Decompress (Viewer)", "Dedup Stats"])


# --- MAIN AREA ---
if mode == "Compress (Encode)":
    st.header("Compute Engine")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("Upload Video / Image / Code")
        uploaded_file = st.file_uploader("Drag and Drop Files", type=['mp4', 'mkv', 'png', 'jpg', 'py', 'rs', 'txt', 'log', 'jpeg', 'mpeg4'], accept_multiple_files=False)

    with col2:
        st.write("### Compression Logs")
        log_area = st.empty() # Keep for now, though the edit doesn't explicitly use it for live logs

    if uploaded_file is not None:
        # Save to temp file to pass to Rust
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        st.write(f"Processing: **{uploaded_file.name}** ({uploaded_file.size / 1024:.2f} KB)")
        
        # Progress Bar
        progress_bar = st.progress(0)
        
        # CALL RUST CORE
        try:
            # Use pre-built binary for speed (falls back to cargo run if not found)
            DEMON_EXE = os.path.join(DEMON_CORE_DIR, "target", "release", "ns-arc-demon.exe")
            
            if os.path.exists(DEMON_EXE):
                # Fast path: use pre-built binary
                cmd = [DEMON_EXE, str(tmp_path), "--out-dir", "./dist"]
            else:
                # Slow path: build and run (first time only)
                cmd = [CARGO_PATH, "run", "--release", "--", str(tmp_path), "--out-dir", "./dist"]
            
            process = subprocess.Popen(
                cmd,
                cwd=DEMON_CORE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()
            progress_bar.progress(100)
            
            # Display Output
            if process.returncode == 0:
                st.success("Analysis Complete!")
                
                # Parse JSON Output
                import json
                metrics = []
                
                logs = stdout + "\n" + stderr
                
                for line in stdout.splitlines():
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            data = json.loads(line)
                            metrics.append(data)
                        except:
                            pass
                
                if metrics:
                    for m in metrics:
                        st.divider()
                        st.write(f"### Result: {os.path.basename(m.get('file', 'Unknown'))}")
                        
                        c1, c2, c3, c4 = st.columns(4)
                        # Support both old and new field names
                        orig = m.get('bytes_in', m.get('original_size', 0))
                        comp = m.get('bytes_out', m.get('compressed_size', 0))
                        method = m.get('method', 'Unknown')
                        mode = m.get('mode', 'unknown')
                        pct_saved = m.get('pct_saved', 0)
                        entropy = m.get('entropy', 0)
                        file_type = m.get('type', m.get('type_detected', 'Unknown'))
                        
                        with c1:
                            st.metric("Original", f"{orig:,} B")
                        with c2:
                            delta = orig - comp
                            if delta == 0:
                                st.metric("Compressed", f"{comp:,} B", delta="0 B (0.00%)", delta_color="off")
                            else:
                                st.metric("Compressed", f"{comp:,} B", delta=f"{pct_saved:.1f}% saved", delta_color="normal")
                        with c3:
                            ratio = orig / comp if comp > 0 else 1.0
                            st.metric("Ratio", f"{ratio:.2f}x")
                        with c4:
                            st.metric("Entropy", f"{entropy:.2f} b/B")
                        
                        st.info(f"**Type**: `{file_type}` | **Mode**: `{mode}` | **Method**: `{method}`")
                        
                        # Show hash if available
                        if m.get('hash_original'):
                            with st.expander("Integrity Hash (SHA-256)"):
                                st.code(m.get('hash_original'), language=None)
                        
                        # Show reason if stored
                        if m.get('reason'):
                            st.warning(f"⚠️ **Decision Reason**: {m.get('reason')}")

                        # Download Button
                        # The Rust core saves to ./dist/[filename].nsarc
                        dist_filename = os.path.splitext(os.path.basename(m.get('file')))[0] + ".nsarc"
                        dist_path = os.path.join(DEMON_CORE_DIR, "dist", dist_filename)
                        
                        if os.path.exists(dist_path):
                            with open(dist_path, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download Compressed File (.nsarc)",
                                    data=f,
                                    file_name=dist_filename,
                                    mime="application/octet-stream"
                                )
                        else:
                            st.error(f"Output file not found: {dist_path}")
                else:
                     st.write("No metrics found in output.")

                with st.expander("View Raw Logic Logs"):
                    st.code(logs)
                    
            else:
                st.error("Engine Failure")
                st.code(stderr)
                
        except Exception as e:
            st.error(f"Execution Error: {e}")
            
        # Cleanup
        os.remove(tmp_path)

elif mode == "Decompress (Viewer)":
    st.header("🔓 ERT Decompressor")
    st.info("Upload a `.nsarc` file to decompress and view the original content.")
    
    uploaded_file = st.file_uploader("Upload Compressed File", type=['nsarc', 'nsv', 'nst', 'zst'])
    
    if uploaded_file:
        # Save temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nsarc") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Display file info
        file_size = len(uploaded_file.getvalue())
        st.write(f"**File**: {uploaded_file.name} ({file_size:,} bytes)")
            
        if st.button("🔓 Decompress"):
            with st.spinner("Decompressing..."):
                # Output path
                out_file = tmp_path + ".decoded"
                
                # Try zstd decompression first (most common)
                try:
                    import zstandard as zstd
                    
                    with open(tmp_path, 'rb') as f_in:
                        compressed_data = f_in.read()
                    
                    try:
                        dctx = zstd.ZstdDecompressor()
                        decompressed = dctx.decompress(compressed_data)
                        
                        with open(out_file, 'wb') as f_out:
                            f_out.write(decompressed)
                        
                        # Success metrics
                        out_size = len(decompressed)
                        ratio = file_size / out_size if out_size > 0 else 1.0
                        
                        st.success(f"✅ Decompression Successful!")
                        
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Compressed", f"{file_size:,} B")
                        with c2:
                            st.metric("Original", f"{out_size:,} B")
                        with c3:
                            st.metric("Ratio", f"{ratio:.2f}x")
                        
                        # Detect content type for preview
                        if decompressed[:8] == b'\x89PNG\r\n\x1a\n':
                            st.image(out_file, caption="Decompressed Image")
                        elif decompressed[:3] == b'\xff\xd8\xff':
                            st.image(out_file, caption="Decompressed JPEG")
                        elif len(decompressed) < 100000:
                            try:
                                text = decompressed.decode('utf-8')
                                st.text_area("Decompressed Text", text, height=300)
                            except:
                                st.info("Binary file decompressed (preview not available)")
                        else:
                            st.info(f"Large file decompressed ({out_size:,} bytes)")
                        
                        # Download button
                        with open(out_file, "rb") as f:
                            original_name = os.path.splitext(uploaded_file.name)[0]
                            st.download_button(
                                "⬇️ Download Decompressed File", 
                                f, 
                                file_name=original_name
                            )
                            
                    except zstd.ZstdError:
                        # Not zstd - check if it's a known image format (Stored as-is)
                        if compressed_data[:8] == b'\x89PNG\r\n\x1a\n':
                            # It's a PNG stored directly
                            st.success("✅ File was stored (PNG detected). Showing original.")
                            with open(out_file, 'wb') as f_out:
                                f_out.write(compressed_data)
                            safe_st_image(out_file, caption="Original Image (Stored)")
                            # Read into memory for download
                            file_data = compressed_data
                            st.download_button("⬇️ Download PNG", file_data, file_name=uploaded_file.name.replace('.nsarc', '.png'))
                        elif compressed_data[:3] == b'\xff\xd8\xff':
                            # It's a JPEG stored directly
                            st.success("✅ File was stored (JPEG detected). Showing original.")
                            with open(out_file, 'wb') as f_out:
                                f_out.write(compressed_data)
                            safe_st_image(out_file, caption="Original Image (Stored)")
                            st.download_button("⬇️ Download JPEG", compressed_data, file_name=uploaded_file.name.replace('.nsarc', '.jpg'))
                        elif compressed_data[:4] == b'RIFF' or (len(compressed_data) > 8 and compressed_data[4:8] == b'ftyp'):
                            # It's a video file stored directly (AVI/MP4)
                            st.success("✅ File was stored (Video detected). Download to view.")
                            ext = '.mp4' if compressed_data[4:8] == b'ftyp' else '.avi'
                            st.download_button("⬇️ Download Video", compressed_data, file_name=uploaded_file.name.replace('.nsarc', ext))
                        else:
                            # Try neural decoder as last resort
                            st.warning("Not a zstd archive. Trying neural decoder...")
                            vision_script = os.path.join(os.path.dirname(DEMON_CORE_DIR), "ns_arc_vision.py")
                            if os.path.exists(vision_script):
                                out_png = tmp_path + ".png"
                                cmd = ["python", vision_script, "decode", "--input", tmp_path, "--output", out_png]
                                result = subprocess.run(cmd, capture_output=True, text=True)
                                
                                if os.path.exists(out_png):
                                    st.success("Neural Reconstruction Successful!")
                                    st.image(out_png, caption="Reconstructed Image")
                                    with open(out_png, "rb") as f:
                                        st.download_button("⬇️ Download PNG", f, file_name="restored.png")
                                else:
                                    # Final fallback: return raw bytes
                                    st.info("File format not recognized. Returning raw bytes.")
                                    with open(out_file, 'wb') as f_out:
                                        f_out.write(compressed_data)
                                    with open(out_file, "rb") as f:
                                        original_name = uploaded_file.name.replace('.nsarc', '')
                                        st.download_button("⬇️ Download Raw", f, file_name=original_name)
                            else:
                                # No neural script, return raw bytes
                                st.info("File was stored (not compressed). Downloaded as-is.")
                                with open(out_file, 'wb') as f_out:
                                    f_out.write(compressed_data)
                                with open(out_file, "rb") as f:
                                    st.download_button("⬇️ Download", f, file_name=uploaded_file.name.replace('.nsarc', ''))
                            
                except ImportError:
                    st.error("zstandard library not installed. Run: pip install zstandard")
            
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(out_file):
                os.remove(out_file)

elif mode == "Dedup Stats":
    st.header("📊 Corpus Deduplication Statistics")
    
    if os.path.exists(CHUNKS_DB_PATH):
        try:
            import sqlite3
            conn = sqlite3.connect(CHUNKS_DB_PATH)
            cursor = conn.cursor()
            
            # Get summary stats
            cursor.execute("SELECT COUNT(*), COALESCE(SUM(ref_count), 0), COALESCE(SUM(size), 0) FROM chunks")
            total_chunks, total_refs, unique_bytes = cursor.fetchone()
            
            cursor.execute("SELECT COUNT(*), COALESCE(SUM(original_size), 0), COALESCE(SUM(compressed_size), 0) FROM files")
            total_files, total_original, total_compressed = cursor.fetchone()
            
            dedup_ratio = total_refs / total_chunks if total_chunks > 0 else 1.0
            space_saved = total_original - unique_bytes if total_original > 0 else 0
            
            # Display metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("📁 Files Processed", f"{total_files:,}")
            with c2:
                st.metric("🧩 Unique Chunks", f"{total_chunks:,}")
            with c3:
                st.metric("🔗 Total References", f"{total_refs:,}")
            with c4:
                st.metric("📈 Dedup Ratio", f"{dedup_ratio:.2f}x")
            
            c5, c6, c7 = st.columns(3)
            with c5:
                st.metric("💾 Original Data", f"{total_original / (1024*1024):.2f} MB")
            with c6:
                st.metric("🗜️ Stored Data", f"{unique_bytes / (1024*1024):.2f} MB")
            with c7:
                st.metric("✨ Space Saved", f"{space_saved / (1024*1024):.2f} MB")
            
            # Recent files table
            st.divider()
            st.subheader("Recent Files")
            cursor.execute("""
                SELECT original_name, original_size, compressed_size, chunk_count, created_at 
                FROM files ORDER BY id DESC LIMIT 20
            """)
            rows = cursor.fetchall()
            
            if rows:
                import pandas as pd
                df = pd.DataFrame(rows, columns=["Name", "Original", "Compressed", "Chunks", "Date"])
                df["Ratio"] = df["Original"] / df["Compressed"].replace(0, 1)
                df["Saved"] = df["Original"] - df["Compressed"]
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No files processed yet. Compress some files to see stats!")
            
            conn.close()
            
        except Exception as e:
            st.error(f"Database error: {e}")
    else:
        st.warning("No chunk database found. Compress some files first!")
        st.info(f"Expected path: `{CHUNKS_DB_PATH}`")
