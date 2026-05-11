import sys
import os
from pathlib import Path

print(f"Current Working Directory: {os.getcwd()}")
print(f"File: {__file__}")
project_root = str(Path(__file__).parent.parent.parent)
print(f"Project Root: {project_root}")

sys.path.append(project_root)
print(f"Sys Path: {sys.path}")

try:
    import src.config.scaler as cfg
    print("Successfully imported src.config.scaler")
    print(f"SCALING_FACTOR: {cfg.SCALING_FACTOR}")
except Exception as e:
    print(f"Failed to import: {e}")
