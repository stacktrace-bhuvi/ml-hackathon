# Synthetic Dummy Data

This folder contains tiny synthetic test data for the PathSnap pipeline.

> [!WARNING]
> This data is purely synthetic for initial pipeline validation, unit testing, and team integration.
> Dummy data results must **NEVER** be used in final reports or evaluations.

The raw directory structure mirrors the actual dataset:
- `data/dummy/train/00000001/00000001.{track,nodes,arcs,route}`
- `data/dummy/train/00000002/00000002.{track,nodes,arcs,route}`
- `data/dummy/student_test/00000003/00000003.{track,nodes,arcs}`

In addition, synthetic parquets matching the exact schema contract are stored here:
- `data/dummy/dummy_points.parquet`
- `data/dummy/dummy_candidates.parquet`
