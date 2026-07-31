#!/usr/bin/env bash
# Everything CI enforces, run locally before committing: formatting, then both
# compilers built and tested. Mirrors .github/workflows/ci.yml.
#
#   scripts/check.sh              # format + clang + gcc
#   scripts/check.sh --fix        # reformat in place instead of failing
#   scripts/check.sh --asan       # also build and test the sanitizer preset
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=lib.sh
source "$REPO_ROOT/scripts/lib.sh"
cd "$REPO_ROOT"

fix_format=0
with_asan=0

for arg in "$@"; do
    case "$arg" in
        --fix) fix_format=1 ;;
        --asan) with_asan=1 ;;
        -h | --help)
            sage_usage "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) sage_die "unknown flag: $arg" ;;
    esac
done

sage_setup_env

command -v clang-format >/dev/null || sage_die "clang-format not found on PATH"

mapfile -t sources < <(find src tests -type f \( -name '*.hpp' -o -name '*.cpp' \))

if ((fix_format)); then
    sage_info "Formatting ${#sources[@]} files"
    clang-format -i "${sources[@]}"
else
    sage_info "Checking formatting (${#sources[@]} files)"
    clang-format --dry-run --Werror "${sources[@]}"
fi

presets=(debug gcc-debug)
((with_asan)) && presets+=(asan)

for preset in "${presets[@]}"; do
    sage_info "Configuring ($preset)"
    cmake --preset "$preset"

    sage_info "Building ($preset)"
    cmake --build --preset "$preset"

    sage_info "Testing ($preset)"
    ctest --test-dir "build/$preset" --output-on-failure
done

printf '\n\033[32mAll checks passed.\033[0m\n'
