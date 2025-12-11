"""
config.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_ROOT = Path("data")
