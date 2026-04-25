#!/usr/bin/env python
"""Remove duplicate old _on_step() method from ComprehensiveMetricsLoggingCallback"""

import sys

# Read the file
with open('tests/pipeline_carla_test.py', 'r') as f:
    content = f.read()

# Find location of "Called every training step" which appears in the old duplicate _on_step
if '"Called every training step"' in content:
    # Count occurrences
    count = content.count('"Called every training step"')
    print(f"[*] Found {count} occurrences of 'Called every training step'")
    
    if count < 2:
        print("[!] Only 1 occurrence found - no duplicate to remove")
        sys.exit(0)
    
    # Find the position of the second (old) occurrence
    pos1 = content.find('"Called every training step"')
    pos2 = content.find('"Called every training step"', pos1 + 1)
    
    if pos2 > pos1:
        # Found the duplicate - find its method start
        method_start = content.rfind('def _on_step', 0, pos2)
        header_pos = content.find('# ============', pos2)
        
        if method_start > 0 and header_pos > method_start:
            print(f"[*] Removing old duplicate code from position {method_start} to {header_pos}")
            
            # Keep content before method + newline + content from header onward
            new_content = content[:method_start] + '\n' + content[header_pos:]
            
            with open('tests/pipeline_carla_test.py', 'w') as f:
                f.write(new_content)
            
            print("[+] Successfully removed old duplicate code")
        else:
            print(f"[!] Could not find proper boundaries: method_start={method_start}, header_pos={header_pos}")
    else:
        print("[!] Could not locate second occurrence")
else:
    print("[!] String not found in file")
