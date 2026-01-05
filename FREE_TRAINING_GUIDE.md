# NS-ARC: Ultra-Detailed Free Training Guide

This guide breaks down the process of training your AI on Google Colab into tiny, specific steps.

## Phase 1: Prepare Your Files (Local Computer)

1.  Open your **File Explorer** on Windows.
2.  Navigate to this text exact location:
    `Documents` -> `Algo flow`
3.  Keep this window OPEN. You should see these 3 files:
    *   `ns_arc_1b.py`
    *   `ns_arc_tokenizer.py`
    *   `ns_arc_colab.py`

## Phase 2: Setup Colab (Web Browser)

1.  Open your web browser (Chrome/Edge).
2.  Go to this URL: [https://colab.research.google.com/](https://colab.research.google.com/)
3.  A popup called "Welcome to Colaboratory" usually appears. Click the **"New Notebook"** button (bottom right of the popup).
4.  You will see a blank code cell.

### Enable the Free GPU
1.  Look at the top menu bar. Click **Runtime**.
2.  Select **Change runtime type**.
3.  A box appears. Under **Hardware accelerator**, click "T4 GPU".
4.  Click **Save**.
5.  *Wait a moment.* In the top right corner, you should see "Connecting..." turn into "RAM" and "Disk" bars.

## Phase 3: The Upload (Important!)

1.  Look at the **Left Sidebar** of the Colab screen.
2.  Click the **Folder Icon** 📁 (it's the last icon in the list).
3.  *Wait a second.* The sidebar will expand, and you will see a folder called `sample_data`. **This means the disk is ready.**
4.  Now, switch back to your **Windows File Explorer** (Phase 1).
5.  Select these 3 files: `ns_arc_1b.py`, `ns_arc_tokenizer.py`, `ns_arc_colab.py`.
6.  **Drag and Drop** them into the **empty white space** below the `sample_data` folder.
    *   **CRITICAL**: Do NOT drop them *on top* of the `sample_data` folder. Drop them in the blank area.
7.  **Verify**: Look at the sidebar. You should see `ns_arc_colab.py` listed **separately** from `sample_data` (at the same indentation level).
    *   If you don't see them, the upload failed. Try again.
    *   If they are *inside* `sample_data`, drag them out to `..` or the top level.

## Phase 4: Run the Training

1.  Click inside the Code Cell (the box with the Play button) in the middle of the screen.
2.  Copy and paste this command:
    ```python
    !python ns_arc_colab.py
    ```
3.  Click the **Play Button** ▶️ on the left of the cell (or press Shift+Enter).
4.  Watch the output! It will say:
    *   `GPU Detected: Tesla T4`
    *   `Initializing 1.2B Parameter Demon Brain...`
    *   `Step 1: Loss ...`

## Phase 5: Download the Brain

1.  When training finishes, a new file named `ns_arc_1b_colab.pt` will appear in the specific Left Sidebar.
2.  **Right-click** on `ns_arc_1b_colab.pt`.
3.  Select **Download**.
4.  Save it to your computer.

**You have now successfully trained a byte-level compression model.**
