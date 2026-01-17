#!/usr/bin/env python3
import sys

current = None
total = 0

for line in sys.stdin:
    key, value = line.strip().split("\t")
    value = int(value)

    if current == key:
        total += value
    else:
        if current:
            print(f"{current}\t{total}")
        current = key
        total = value

if current:
    print(f"{current}\t{total}")
