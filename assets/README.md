# Assets

Third-party test assets, vendored so a fresh clone runs without fetching
anything. Only geometry files are committed — sage defers materials and
textures, so the accompanying texture images are deliberately omitted and the
`images` entries in the glTF resolve to nothing.

Point the app at any other glTF with a path argument:

```bash
./build/debug/src/app/sage path/to/model.gltf
```

## lantern/

`Lantern.gltf`, `Lantern.bin` — from the
[Khronos glTF-Sample-Assets](https://github.com/KhronosGroup/glTF-Sample-Assets)
repository.

© 2017 Microsoft, © 2018 Frank Galligan.
Licensed under [Creative Commons Zero v1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/legalcode)
(public domain dedication).

Chosen because it is small (237 KB without textures), CC0 rather than a bespoke
license, and structurally non-trivial: a parent node carrying a rotation over
three translated children, which exercises node-hierarchy flattening rather than
just single-mesh loading.
