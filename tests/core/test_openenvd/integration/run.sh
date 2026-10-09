#!/usr/bin/env bash
# Build the integration image and run the live-unit tests in both tiers:
#   containers: a privileged container, so openenvd can nest containers;
#   landlock:   an ordinary container running as uid 1000, like HF Spaces.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../../.." && pwd)"
docker build -f "$root/tests/core/test_openenvd/integration/Dockerfile" -t openenvd-it "$root"
docker run --rm --privileged --cgroupns=private \
    --tmpfs /var/lib/openenvd:exec --tmpfs /tmp:exec \
    openenvd-it python -m pytest -q -p no:cacheprovider /opt/tests/test_live_unit.py "$@"
docker run --rm --user 1000:1000 -e OPENENVD_IT_TIER=landlock \
    --tmpfs /var/lib/openenvd:exec,uid=1000,gid=1000 --tmpfs /tmp:exec,mode=1777 \
    openenvd-it python -m pytest -q -p no:cacheprovider /opt/tests/test_live_unit.py "$@"
