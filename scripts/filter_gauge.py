#!/usr/bin/env python3
"""Filter gauge_compare log to only QR or SQ STEP lines for analyze_ip_mpf.py"""
import sys, re

gauge = sys.argv[1] if len(sys.argv) > 1 else 'QR'   # 'QR' or 'SQ'
path  = sys.argv[2] if len(sys.argv) > 2 else '-'

f = open(path) if path != '-' else sys.stdin
for line in f:
    if line.startswith('STEP') and f'gauge {gauge}' in line:
        # Strip the trailing gauge tag so analyze_ip_mpf.py parser matches
        print(re.sub(r'\s+gauge \w+.*$', '', line.rstrip()))
    elif not line.startswith('STEP'):
        print(line, end='')
