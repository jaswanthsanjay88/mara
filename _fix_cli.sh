#!/usr/bin/env bash
set -e
export PATH="$HOME/.local/bin:$PATH"
colab exec -s mara-trainer <<'PYEOF'
import subprocess
log = open('/content/prep.log', 'w')
p = subprocess.Popen(['python', '-u', '-m', 'mara.data'],
                     cwd='/content', stdout=log, stderr=subprocess.STDOUT)
print('started pid:', p.pid)
PYEOF
