import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils import delete_pycache_folders

delete_pycache_folders()

