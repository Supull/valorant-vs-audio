"""
config.py
---------
Loads environment variables from .env file.
Copy .env.example to .env and fill in your values.
Never commit .env to git.
"""

import os
from dotenv import load_dotenv

load_dotenv()

MAC_IP      = os.getenv("MAC_IP",      "192.168.1.34")
PORT        = int(os.getenv("PORT",        "5005"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "44100"))
CHUNK_SIZE  = int(os.getenv("CHUNK_SIZE",  "1024"))
CHANNELS    = int(os.getenv("CHANNELS",    "2"))