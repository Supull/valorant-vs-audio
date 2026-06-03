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

MAC_IP      = os.getenv("MAC_IP",      "xxx.xxx.x.xx")
PORT        = int(os.getenv("PORT",        "xxxx"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "xxxxx"))
CHUNK_SIZE  = int(os.getenv("CHUNK_SIZE",  "xxxx"))
CHANNELS    = int(os.getenv("CHANNELS",    "x"))