"""
AETHEL // Clinical Readmission Risk Intelligence Platform
Root entrypoint for Streamlit Community Cloud & GitHub deployments.
"""

import sys
from pathlib import Path

# Ensure project root directory is on Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Launch Master Application
from app import app
