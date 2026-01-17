#!/usr/bin/env python3
import sys
import re

for line in sys.stdin:
    try:
        record = line.strip()
        if not record:
            continue

        import json
        obj = json.loads(record)
        text = obj.get("text", "").lower()

        words = re.findall(r"[a-z]+", text)

        for word in words:
            print(f"{word}\t1")
    except:
        continue
