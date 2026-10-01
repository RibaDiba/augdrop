import os
import sys

# src/ layout with no installed package yet: make `probe`, `vocab`, `hosts` importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
