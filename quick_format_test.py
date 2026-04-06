#!/usr/bin/env python3
"""Quick test: run one task and verify [START]/[STEP]/[END] format."""

import subprocess
import sys
import re

# Run inference on easy-arbitrage only
result = subprocess.run(
    [sys.executable, "inference.py"],
    capture_output=True,
    text=True,
    env={"VPP_SERVER_URL": "http://localhost:7860", "PATH": sys.exec_prefix},
    timeout=60,
)

stdout_lines = result.stdout.strip().split('\n')
stderr_lines = result.stderr.strip().split('\n') if result.stderr else []

print("=== STDOUT (format verification) ===", file=sys.stderr)
for line in stdout_lines:
    print(f"  {line}", file=sys.stderr)

print("\n=== STDERR (diagnostics) ===", file=sys.stderr)
for line in stderr_lines[:20]:  # Show first 20 to avoid spam
    print(f"  {line}", file=sys.stderr)

# Verify format
start_count = len([l for l in stdout_lines if l.startswith("[START]")])
step_count = len([l for l in stdout_lines if l.startswith("[STEP]")])
end_count = len([l for l in stdout_lines if l.startswith("[END]")])

print(f"\n=== FORMAT CHECK ===", file=sys.stderr)
print(f"  [START] lines: {start_count} (expect ≥1)", file=sys.stderr)
print(f"  [STEP] lines:  {step_count} (expect >1)", file=sys.stderr)
print(f"  [END] lines:   {end_count} (expect ≥1)", file=sys.stderr)

# Verify field presence in [END] line
if end_count > 0:
    end_line = [l for l in stdout_lines if l.startswith("[END]")][0]
    required_fields = ["success=", "steps=", "score=", "rewards="]
    missing = [f for f in required_fields if f not in end_line]
    if missing:
        print(f"  ❌ FAIL: Missing fields in [END]: {missing}", file=sys.stderr)
    else:
        print(f"  ✅ PASS: All [END] fields present", file=sys.stderr)

sys.exit(0 if start_count >= 1 and step_count >= 1 and end_count >= 1 else 1)
