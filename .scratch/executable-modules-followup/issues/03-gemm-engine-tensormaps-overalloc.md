# 03: GEMMEngine over-allocates the tensormaps workspace

**What:** the `GEMMEngine` ctor
(src/turbomind/models/llama/GEMMEngine.cu) sets
`workspace_.tensormaps_size = 8192 * 128` (~1 MiB, "maximum 4096 tensor
maps") but allocates `workspace_.tensormaps` with `workspace_.partials_size`
(tens of MiB). Moved verbatim from the retired `LlamaLinear.cu`, so the
waste predates the executable-modules effort — not a regression, just free
VRAM burned on every rank at engine creation.

**Direction:** allocate with `tensormaps_size` after confirming no kernel
writes past 1 MiB of tensor-map space.

**Status:** needs-triage
