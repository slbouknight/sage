#!/usr/bin/env bash
# Configure and build one preset, optionally running the tests or the app.
#
#   scripts/build.sh                    # clang Debug
#   scripts/build.sh gcc-debug          # gcc Debug
#   scripts/build.sh asan --test        # sanitizers, then ctest
#   scripts/build.sh debug --run        # build, then launch the app
#   scripts/build.sh relwithdebinfo     # the only build worth profiling
#
# Flags: --test, --run, --clean (wipe the preset's build dir first).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib.sh
source "$REPO_ROOT/scripts/lib.sh"
cd "$REPO_ROOT"

readonly PRESETS=(debug gcc-debug asan relwithdebinfo)

preset=debug
do_test=0
do_run=0
do_clean=0

for arg in "$@"; do
    case "$arg" in
        --test) do_test=1 ;;
        --run) do_run=1 ;;
        --clean) do_clean=1 ;;
        -h | --help)
            sage_usage "${BASH_SOURCE[0]}"
            exit 0
            ;;
        -*) sage_die "unknown flag: $arg" ;;
        *)
            # shellcheck disable=SC2076
            [[ " ${PRESETS[*]} " =~ " $arg " ]] ||
                sage_die "unknown preset: $arg (expected one of: ${PRESETS[*]})"
            preset="$arg"
            ;;
    esac
done

sage_setup_env

if ((do_clean)); then
    sage_info "Removing build/$preset"
    rm -rf "build/$preset"
fi

sage_info "Configuring ($preset)"
cmake --preset "$preset"

sage_info "Building ($preset)"
cmake --build --preset "$preset"

if ((do_test)); then
    sage_info "Testing ($preset)"
    ctest --test-dir "build/$preset" --output-on-failure
fi

if ((do_run)); then
    sage_info "Running ($preset)"
    exec "./build/$preset/src/app/sage"
fi
