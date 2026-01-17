#!/usr/bin/env python3
import sys
import json

for line in sys.stdin:
    try:
        record = json.loads(line)
        rating = record.get("rating")
        if rating is not None:
            print(f"{int(rating)}\t1")
    except:
        continue