# spectralRayTracer

A C++20 CPU rendering experiment built around a concrete question: how much ray-tracing work can be avoided, grouped, or moved efficiently through the processor? It loads an STL mesh, builds a bounding-volume hierarchy, traces eight rays at a time with AVX2, and spends additional samples where an initial pass suggests they are useful.

The broader idea in the original [plan](plan) is a spectral renderer whose light transport is separate from a configurable virtual sensor. That ambition explains the eight-band ray layout, but the current executable is a narrower renderer with RGB output and hardcoded surface behavior.

## The design ideas

The plan starts with memory layout rather than a scene editor. Ray components are stored as arrays so an eight-ray packet can feed SIMD registers without loading unrelated fields. Compact BVH nodes let the renderer reject groups of triangles before testing individual intersections. SIMD masks keep hit information for each ray while processing a packet together.

The later May 2026 commits concentrate on sampling and measurement. Instead of giving every pixel the full budget, the renderer makes an initial pass, estimates tile brightness, variance and surface coverage, then chooses additional work. Diagnostic counters and a sample-count heatmap make that allocation inspectable. Commit messages report performance experiments, but they are not a portable benchmark.

## Current pipeline

```text
orc.stl → rotate mesh → build SAH BVH
        → initial ray samples → choose tile sample budgets
        → threaded AVX2 packet tracing → accumulated RGB
        → color image or sample heatmap → image.ppm
```

| Source | Responsibility |
| --- | --- |
| [main.cpp](main.cpp) | Scene constants, lighting, adaptive tiles, worker scheduling, shading and diagnostics. |
| [core_types.h](core_types.h) | Engine, geometry, ray-packet and material layouts. |
| [geometry_loader.h](geometry_loader.h) | Binary STL loading, mesh transforms and BVH construction. |
| [simd_math.h](simd_math.h) and [wavefront_pipeline.h](wavefront_pipeline.h) | Packet math and accelerated traversal. |
| [display_sensor.h](display_sensor.h) | Pixel representation and text-based P3 PPM output. |

The supplied scene uses `orc.stl`, a fixed camera and copper-colored shading. Resolution is currently **3840×2160**, with a maximum sampling budget of **128**. These are source constants, not command-line options. `RENDER_HEATMAP` selects the diagnostic image; it is currently `false`. Packet seeds are deterministic, and the makefile intentionally avoids `-ffast-math`.

## What “spectral” currently means

Eight spectrum fields exist, but the active light/output calculations use bands 0, 3 and 6. Surface response is hardcoded in the rendering loop; the allocated material table does not drive shading. The planned adjustable sensor cones and independent display thread are not implemented. `display_sensor.h` currently writes an RGB file rather than simulating a configurable sensor.

These boundaries leave a working direction for further experiments without claiming a complete spectral simulation or validated physical accuracy.

## Build and run

Use an x86 machine supporting AVX2 and FMA, `g++` with C++20 support, and GNU Make. The source uses the standard library and compiler intrinsics; no third-party package installation is specified. The [Makefile](Makefile) also uses `-march=native`, so its binary is tailored to the build machine.

From the repository directory:

```sh
make -B
./spectral_tracer
```

Keep `orc.stl` in the working directory. Running overwrites `image.ppm`, which is tracked; preserve an existing render before replacing it. There is no interactive window or scene-selection CLI.

**Use a forced rebuild after source changes:** the current makefile does not track header dependencies, so ordinary `make` can reuse stale geometry or traversal code. `make -B` avoids that limitation until dependency tracking is corrected. A source syntax check passed during review; a full build, render, timing comparison and image validation were not performed for this documentation update.
