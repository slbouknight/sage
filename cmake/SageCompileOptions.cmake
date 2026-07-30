# sage::compile_options — the single INTERFACE target every sage target links.
# Do not set warning/debug-info flags anywhere else; add them here so every
# module gets them uniformly.

add_library(sage_compile_options INTERFACE)
add_library(sage::compile_options ALIAS sage_compile_options)

target_compile_features(sage_compile_options INTERFACE cxx_std_20)

target_compile_options(sage_compile_options INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
    -Wcast-align
    -Wold-style-cast
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -Wimplicit-fallthrough
    -Werror
)

# Linux-only, and we want the same STL ABI under both Clang and GCC so
# libstdc++ debug info and ASAN interceptors line up regardless of which
# compiler produced a given .o.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(sage_compile_options INTERFACE -stdlib=libstdc++)
    target_link_options(sage_compile_options INTERFACE -stdlib=libstdc++)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(sage_compile_options INTERFACE -fno-omit-frame-pointer)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # -fstandalone-debug: keep full type info for STL types even when a
        # TU never instantiates them, so the debugger can still show them.
        target_compile_options(sage_compile_options INTERFACE -ggdb3 -fstandalone-debug)
    else()
        target_compile_options(sage_compile_options INTERFACE -ggdb3)
    endif()
endif()

# Consumed via the `asan` preset (SAGE_SANITIZE=address,undefined).
if(SAGE_SANITIZE)
    target_compile_options(sage_compile_options INTERFACE
        -fsanitize=${SAGE_SANITIZE}
        -fno-omit-frame-pointer
    )
    target_link_options(sage_compile_options INTERFACE -fsanitize=${SAGE_SANITIZE})
endif()
