# 0003: Defer OpenUSD behind `SAGE_ENABLE_USD` (default OFF)

**Status:** Accepted

**Context:** OpenUSD ingestion is M6 scope and pulls in a heavy, slow-to-build
dependency tree that nothing before M6 needs. Every earlier milestone builds
and iterates against glTF via fastgltf instead.

**Decision:** Gate all USD code behind `option(SAGE_ENABLE_USD ... OFF)` in
CMake so it never builds — and vcpkg never fetches it — unless explicitly
turned on.

**Consequences:** CI and local dev stay fast for M0–M5; USD integration work
can't start until the flag and its build wiring exist.
