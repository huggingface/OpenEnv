#!/usr/bin/env bash
# Test Docker builds for OpenEnv environments.
#
# Builds each env image, starts the container, polls /health, then cleans up.
# Exits non-zero if any build or health check fails.
#
# Usage:
#   bash scripts/test-docker-builds.sh                              # test all envs, both platforms
#   bash scripts/test-docker-builds.sh echo-env                     # test one env, both platforms
#   bash scripts/test-docker-builds.sh "echo-env,grid-world-env"    # test multiple envs, both platforms
#   bash scripts/test-docker-builds.sh --platform=linux/arm64       # test all envs, arm64 only
#   bash scripts/test-docker-builds.sh --platform=linux/amd64,linux/arm64  # explicit both
#   bash scripts/test-docker-builds.sh --no-base                    # skip base image rebuild
#   SKIP_ENVS="chat-env finrl-env" bash scripts/test-docker-builds.sh
#
# Requirements: docker (with buildx)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="openenv-base:test"
HEALTH_TIMEOUT=120   # seconds to wait for /health to respond
HEALTH_INTERVAL=3    # seconds between polls
START_PORT=18000     # base port; each env gets its own to allow parallelism

# Default: test both architectures. Override with --platform=linux/amd64
PLATFORMS=("linux/amd64" "linux/arm64")

# ---------------------------------------------------------------------------
# Environment matrix: name | context | dockerfile | container_port (optional, default 8000)
# Format: "name:context:dockerfile" or "name:context:dockerfile:port"
# Keep this list in sync with the matrix in .github/workflows/docker-build.yml
# ---------------------------------------------------------------------------
declare -a ENVS=(
    "echo-env:envs/echo_env:envs/echo_env/server/Dockerfile"
    "chat-env:envs/chat_env:envs/chat_env/server/Dockerfile"
    "coding-env:.:envs/coding_env/server/Dockerfile"
    "connect4-env:envs/connect4_env:envs/connect4_env/server/Dockerfile"
    "chess-env:envs/chess_env:envs/chess_env/server/Dockerfile"
    "tbench2-env:envs/tbench2_env:envs/tbench2_env/server/Dockerfile"
    "textarena-env:envs/textarena_env:envs/textarena_env/server/Dockerfile"
    "maze-env:envs/maze_env:envs/maze_env/server/Dockerfile"
    "snake-env:envs/snake_env:envs/snake_env/server/Dockerfile"
    "browsergym-env:envs/browsergym_env:envs/browsergym_env/server/Dockerfile"
    "git-env:envs/git_env:envs/git_env/server/Dockerfile"
    "atari-env:envs/atari_env:envs/atari_env/server/Dockerfile"
    "sumo-rl-env:envs/sumo_rl_env:envs/sumo_rl_env/server/Dockerfile"
#   "finrl-env:envs/finrl_env:envs/finrl_env/server/Dockerfile"          # heavy deps, long build
#   "dipg-safety-env:envs/dipg_safety_env:envs/dipg_safety_env/server/Dockerfile"  # needs special runtime setup
    "unity-env:envs/unity_env:envs/unity_env/server/Dockerfile"
    "openapp-env:envs/openapp_env:envs/openapp_env/server/Dockerfile"
    "openspiel-env:envs/openspiel_env:envs/openspiel_env/server/Dockerfile"
    "grid-world-env:envs/grid_world_env:envs/grid_world_env/server/Dockerfile"
    "calendar-env:envs/calendar_env:envs/calendar_env/Dockerfile:8004"
)

# Envs that need special runtime deps or take very long — skip health check,
# only verify the build succeeds.
BUILD_ONLY_ENVS=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log()  { echo -e "${NC}[$(date +%H:%M:%S)] $*"; }
ok()   { echo -e "${GREEN}[$(date +%H:%M:%S)] ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] ⚠ $*${NC}"; }
fail() { echo -e "${RED}[$(date +%H:%M:%S)] ✗ $*${NC}"; }

cleanup_container() {
    local cid="$1"
    docker rm -f "$cid" &>/dev/null || true
}

wait_for_health() {
    local url="$1" timeout="$2" interval="$3"
    local elapsed=0
    while (( elapsed < timeout )); do
        if curl -sf "$url" &>/dev/null; then
            return 0
        fi
        sleep "$interval"
        (( elapsed += interval ))
    done
    return 1
}

is_build_only() {
    local name="$1"
    for skip in $BUILD_ONLY_ENVS; do
        [[ "$skip" == "$name" ]] && return 0
    done
    return 1
}

# Returns 0 if name matches the FILTER (or FILTER is empty)
in_filter() {
    local name="$1"
    [[ -z "$FILTER" ]] && return 0
    echo ",$FILTER," | grep -q ",${name}," && return 0
    return 1
}

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
BUILD_BASE=true
FILTER=""
for arg in "$@"; do
    case "$arg" in
        --no-base) BUILD_BASE=false ;;
        --platform=*)
            IFS=',' read -ra PLATFORMS <<< "${arg#--platform=}"
            ;;
        --*) warn "Unknown flag: $arg" ;;
        # Positional arg: single env name or comma-separated list
        *) FILTER="$arg" ;;
    esac
done

SKIP_ENVS="${SKIP_ENVS:-}"

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Build and test each env for each platform
# ---------------------------------------------------------------------------
PASSED=()
FAILED=()
SKIPPED=()

for platform in "${PLATFORMS[@]}"; do
    # linux/amd64 → linux-amd64  (safe for docker tag / result labels)
    plat_tag="${platform//\//-}"
    # Base image reuses the same local tag for each platform — platforms are built
    # sequentially so there's no collision, and plain docker build can find it locally.
    base_tag="$BASE_IMAGE"

    log ""
    log "════════════════════════════════════════════"
    log "Platform: $platform"
    log "════════════════════════════════════════════"

    if $BUILD_BASE; then
        log "Building base image for $platform: $base_tag"
        if ! docker build \
                --platform "$platform" \
                -t "$base_tag" \
                -f src/openenv/core/containers/images/Dockerfile \
                . \
                2>&1 | sed 's/^/  /'; then
            fail "Base image build failed for $platform"
            exit 1
        fi
        ok "Base image built: $base_tag"
    else
        log "Skipping base image build (--no-base)"
    fi

    port=$START_PORT

    for entry in "${ENVS[@]}"; do
        IFS=':' read -r name context dockerfile container_port <<< "$entry"
        container_port="${container_port:-8000}"

        if ! in_filter "$name"; then
            continue
        fi

        if echo "$SKIP_ENVS" | grep -qw "$name"; then
            warn "Skipping $name [$platform] (in SKIP_ENVS)"
            SKIPPED+=("$name ($plat_tag)")
            continue
        fi

        if [[ ! -f "$dockerfile" ]]; then
            warn "Skipping $name ($dockerfile not found)"
            SKIPPED+=("$name ($plat_tag)")
            continue
        fi

        image="openenv-test-${name}-${plat_tag}"
        log "─────────────────────────────────────────"
        log "Building $name [$platform]"
        log "  context:    $context"
        log "  dockerfile: $dockerfile"
        log "  port:       $container_port"

        if ! docker build \
                --platform "$platform" \
                -t "$image" \
                -f "$dockerfile" \
                --build-arg "BASE_IMAGE=$base_tag" \
                "$context" \
                2>&1 | sed 's/^/  /'; then
            fail "Build failed: $name [$platform]"
            FAILED+=("$name ($plat_tag, build)")
            continue
        fi
        ok "Build succeeded: $name [$platform]"

        # Build-only envs: don't run a container
        if is_build_only "$name"; then
            ok "Build-only check passed: $name [$platform]"
            PASSED+=("$name ($plat_tag)")
            (( port++ ))
            continue
        fi

        # Start container
        log "Starting container for $name [$platform] on port $port → container:$container_port"
        run_out=$(docker run -d \
                --platform "$platform" \
                -p "${port}:${container_port}" \
                "$image" 2>&1)
        run_exit=$?
        cid=$(echo "$run_out" | tail -1)   # last line is the container ID on success
        if [[ $run_exit -ne 0 ]] || [[ -z "$cid" ]]; then
            fail "Failed to start container: $name [$platform]"
            log "  docker run output: $run_out"
            FAILED+=("$name ($plat_tag, start)")
            (( port++ )) || true
            continue
        fi

        # Poll health endpoint
        health_url="http://localhost:${port}/health"
        log "Waiting for $health_url (up to ${HEALTH_TIMEOUT}s)"
        if wait_for_health "$health_url" "$HEALTH_TIMEOUT" "$HEALTH_INTERVAL"; then
            ok "Health check passed: $name [$platform] → $health_url"
            PASSED+=("$name ($plat_tag)")
        else
            fail "Health check timed out: $name [$platform]"
            log "Container logs:"
            docker logs "$cid" 2>&1 | tail -30 | sed 's/^/  /'
            FAILED+=("$name ($plat_tag, health)")
        fi

        cleanup_container "$cid"
        (( port++ ))
    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════"
echo " Results"
echo "═══════════════════════════════════════════"
echo ""
if (( ${#PASSED[@]} > 0 )); then
    ok "Passed (${#PASSED[@]}): ${PASSED[*]}"
fi
if (( ${#SKIPPED[@]} > 0 )); then
    warn "Skipped (${#SKIPPED[@]}): ${SKIPPED[*]}"
fi
if (( ${#FAILED[@]} > 0 )); then
    fail "Failed (${#FAILED[@]}): ${FAILED[*]}"
    echo ""
    exit 1
fi
echo ""
ok "All checks passed."
