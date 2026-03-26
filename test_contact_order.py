import sys
import re

# Load configs to check order
with open('configs/tinker_constraint_him_trot.py', 'r') as f:
    text = f.read()

# find class scales( ... )
match = re.search(r'class scales\(.*?\):(.*?)class', text, re.DOTALL)
if match:
    scales_text = match.group(1)
    lines = [L.strip() for L in scales_text.splitlines() if '=' in L and not L.strip().startswith('#')]
    keys = [L.split('=')[0].strip() for L in lines]
    
    air_idx = keys.index('feet_air_time') if 'feet_air_time' in keys else -1
    contact_vel_idx = keys.index('feet_contact_velocity') if 'feet_contact_velocity' in keys else -1
    
    print(f"Index of feet_air_time: {air_idx}")
    print(f"Index of feet_contact_velocity: {contact_vel_idx}")
    if contact_vel_idx > air_idx:
        print("WARNING: feet_contact_velocity is AFTER air_time, air_time will be reset before velocity parses it!")
