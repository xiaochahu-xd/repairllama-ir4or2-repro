import json
import sys

def norm(s):
    return "\n".join(line.rstrip() for line in s.strip().splitlines()).strip()

path = sys.argv[1]
total = 0
hit = 0

with open(path, encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        gold = norm(obj.get("gold_patch", ""))
        outputs = obj.get("output", {})
        ok = False

        for item in outputs.values():
            pred = item.get("output_patch") or item.get("output_diff") or ""
            if norm(pred) == gold:
                ok = True
                break

        total += 1
        hit += int(ok)

print({
    "file": path,
    "total": total,
    "exact_hit": hit,
    "exact_rate": hit / total if total else 0
})
