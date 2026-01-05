#!/usr/bin/env python3
"""
NS-ARC Dashboard Launcher

Run this to start the Streamlit dashboard for NS-ARC compression.
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    dashboard_path = os.path.join(os.path.dirname(__file__), "ns_arc_core", "ns_arc_dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path] + sys.argv[1:])
