#!/usr/bin/env bash
# Shared environment checks. Sourced by the other scripts, not run directly.

# Minimum from CMakeLists.txt. Distro packages are often older than this and
# sit earlier on PATH, so the version is checked rather than assumed.
readonly SAGE_MIN_CMAKE=3.28

sage_die() {
    printf '\033[31merror:\033[0m %s\n' "$*" >&2
    exit 1
}

sage_info() {
    printf '\033[36m==>\033[0m %s\n' "$*"
}

# Prints the calling script's header comment block (everything after the
# shebang up to the first non-comment line) as its usage text.
sage_usage() {
    awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$1"
}

sage_require_cmake() {
    command -v cmake >/dev/null || sage_die "cmake not found on PATH"

    local version
    version="$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')"

    if [[ "$(printf '%s\n%s\n' "$SAGE_MIN_CMAKE" "$version" | sort -V | head -1)" != "$SAGE_MIN_CMAKE" ]]; then
        sage_die "cmake $version is too old (need >= $SAGE_MIN_CMAKE): $(command -v cmake)"
    fi
}

sage_require_vcpkg() {
    [[ -n "${VCPKG_ROOT:-}" ]] ||
        sage_die "VCPKG_ROOT is not set. Point it at your vcpkg checkout, e.g.
    export VCPKG_ROOT=~/vcpkg"
    [[ -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]] ||
        sage_die "VCPKG_ROOT=$VCPKG_ROOT does not look like a vcpkg checkout"
}

# find_package(Vulkan) resolves against VULKAN_SDK, and the validation layers
# are found through the layer path the same script exports. Auto-source the
# newest SDK if the caller has not already done so.
sage_require_vulkan_sdk() {
    if [[ -n "${VULKAN_SDK:-}" ]]; then
        return
    fi

    local setup_env
    setup_env="$(find "$HOME/vulkansdk" -maxdepth 2 -name setup-env.sh 2>/dev/null | sort -V | tail -1)"

    [[ -n "$setup_env" ]] ||
        sage_die "VULKAN_SDK is not set and no SDK was found under ~/vulkansdk. Source it first:
    source /path/to/vulkansdk/<version>/setup-env.sh"

    sage_info "Sourcing $setup_env"
    # shellcheck disable=SC1090
    source "$setup_env"
}

sage_setup_env() {
    sage_require_cmake
    sage_require_vcpkg
    sage_require_vulkan_sdk
}
