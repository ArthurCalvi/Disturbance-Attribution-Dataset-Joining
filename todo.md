TODO — Google Satellite Embeddings (DeepMind) for Disturbance Classes

Goal
- Use Google’s annual Satellite Embedding (DeepMind) to: (1) characterize embedding signatures for each disturbance class (Fire, Storm, Anthropogenic, Biotic, Drought, Unknown), (2) benchmark class separability vs. existing features, and (3) explore change detection between years.

Scope and Fit in This Repo
- Inputs: attributed events in `outputs/attribution/` and tiles in `results/datasets/tiles_2_5_km_final.parquet`.
- New data: per‑tile annual embeddings (64 bands) exported from Earth Engine.
- Processing: sample embeddings under attributed event polygons; aggregate by class; analyze separability and temporal stability.
- Outputs: QC plots + metrics in `outputs/qc/embeddings/` and optional per‑event embedding parquet in `results/datasets/`.

References
- Earth Engine dataset: `GOOGLE/SATELLITE/EMBEDDING/V1/ANNUAL` (unit‑length 64‑D vectors)
- Data catalog: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL

Directory & Naming Conventions
- Raw embeddings (GeoTIFFs): `data/embeddings/annual/{year}/emb_tile_{tile_id}_{year}.tif`
- VRT index (optional): `data/embeddings/annual/{year}/features.vrt`
- Per‑event samples: `results/datasets/embeddings_events_{year}.parquet`
- QC outputs: `outputs/qc/embeddings/{year}/...`

Phase 0 — Decisions & Config
- [ ] Choose analysis year(s) (match attribution year coverage).
- [ ] Ensure tile parquet exists: `results/datasets/tiles_2_5_km_final.parquet` with `tile_id`.
- [ ] Add constants to `src/config/constants.py`:
  - `EMBEDDING_BANDS = [f"emb_{i:03d}" for i in range(64)]`
  - `EMBEDDING_CRS = 'EPSG:2154'`, `EMBEDDING_RES = 10`
  - `EMBEDDING_DIR = 'data/embeddings/annual'`

Phase 1 — Export Embeddings from Earth Engine
- For each target year and tile, export the 64‑band image in `EPSG:2154` at 10 m.
- Example (Python API):
  ```python
  import ee, time
  ee.Initialize()
  year = 2023
  start = ee.Date.fromYMD(year, 1, 1)
  img = (ee.ImageCollection('GOOGLE/SATELLITE/EMBEDDING/V1/ANNUAL')
           .filterDate(start, start.advance(1,'year')).mosaic())
  new_names = ee.List.sequence(0, 63).map(lambda i: ee.String('emb_').cat(ee.Number(i).format('%03d')))
  img = img.select(img.bandNames(), new_names)
  # Load your tiles as an EE asset with `tile_id`
  tiles = ee.FeatureCollection('users/<account>/tiles_2_5_km_final')
  tile = tiles.filter(ee.Filter.eq('tile_id', 'TILE_###')).geometry()
  task = ee.batch.Export.image.toDrive(
      image=img.clip(tile), description=f'emb_TILE_###_{year}',
      fileNamePrefix=f'emb_TILE_###_{year}', folder='EarthEngineExports',
      region=tile, crs='EPSG:2154', scale=10, maxPixels=1e13, fileFormat='GeoTIFF')
  task.start();
  while task.active():
      print('Exporting...', task.status()); time.sleep(30)
  print('Done:', task.status())
  ```
- Post‑export: move files into `data/embeddings/annual/{year}/` and ensure bands are `emb_000..emb_063`.

Phase 2 — Build VRT (optional)
- [ ] Create a VRT to index per‑tile rasters (fast IO, single path):
  - `gdalbuildvrt data/embeddings/annual/2023/features.vrt data/embeddings/annual/2023/emb_tile_*.tif`
- [ ] Validate: CRS `EPSG:2154`, 10 m pixel size, 64 bands.

Phase 3 — Sample Embeddings Under Events
- Input events: `outputs/attribution/*` as GeoParquet with polygons and `final_class` labels.
- Implement a small utility in `src/embeddings/sample_events.py` to:
  - [ ] Load embedding raster(s) for the year (VRT or directory of tiles).
  - [ ] Overlay event polygons; sample per‑polygon stats of each band (e.g., mean). Result: a 64‑D vector per event.
  - [ ] Handle large polygons via grid tiling or rasterization at 10 m before zonal stats.
  - [ ] Output parquet: columns `event_id`, `final_class`, `year`, `emb_000..emb_063`, geometry WKT/centroid (optional).
- Technical notes:
  - Prefer `rasterio` + `rasterstats`/`rioxarray` for zonal statistics.
  - Use cosine metrics; embeddings are unit‑length, so dot product equals cosine similarity.

Phase 4 — Class Embedding Analysis (QC)
- Implement `src/qc/embeddings_analysis.py` with CLI to read `embeddings_events_{year}.parquet` and produce:
  - [ ] Per‑class centroid (64‑D mean unit vector) and within‑class variance (cosine/angle spread).
  - [ ] Between‑class cosine distance matrix; separability summary (e.g., min inter‑class vs max intra‑class angle).
  - [ ] 2‑D UMAP/t‑SNE of embeddings colored by `final_class` and sized by event area.
  - [ ] Class‑conditional radial plots (spider) of top‑variance dimensions (optional).
  - [ ] Save plots + CSV metrics to `outputs/qc/embeddings/{year}/`.

Phase 5 — Benchmark vs Existing Features (optional)
- If you have per‑event harmonics or other features, run side‑by‑side classification:
  - [ ] Train a simple RF/LightGBM using embeddings only vs harmonics only vs both.
  - [ ] Use eco‑region or spatial folds to avoid leakage.
  - [ ] Report macro‑F1/ROC‑AUC per class; calibration; feature importances.
  - [ ] Save confusion matrices and PR curves to `outputs/qc/embeddings/{year}/`.

Phase 6 — Year‑to‑Year Change (Angle)
- For two years (Y1,Y2), compute per‑event change:
  - [ ] Either re‑sample both years under the same event geometries, or sample the same pixel sets inside events.
  - [ ] Compute dot product and angle between per‑event mean vectors: `angle = arccos(dot)`.
  - [ ] Correlate large angles with known disturbance timing; visualize distributions.

Phase 7 — Pipeline Integration
- [ ] Add a post‑attribution step in `src/inference/` to optionally trigger embedding sampling and QC when embeddings are present.
- [ ] Minimal config switches: `use_embeddings`, `embedding_year`, `embedding_path`.
- [ ] Register color map from `DISTURBANCE_CLASS_COLORS` for consistent plots.

Deliverables & Acceptance Criteria
- [ ] `results/datasets/embeddings_events_{year}.parquet` with 64‑D vectors per event and labels.
- [ ] QC pack in `outputs/qc/embeddings/{year}/`: centroid CSV, distance matrix, UMAP/t‑SNE, separability metrics.
- [ ] (Optional) Benchmark report comparing embeddings vs existing features.

Potential Pitfalls
- CRS/pixel alignment: always export as `EPSG:2154`, `scale=10` in GEE.
- Large polygons: use tiling or rasterization to control memory during zonal stats.
- Earth Engine quotas: batch exports; monitor tasks.
- Class imbalance: use stratified sampling or class weighting for benchmarks.

Quick Commands (once exports are downloaded)
- Build VRT:
  - `gdalbuildvrt data/embeddings/annual/2023/features.vrt data/embeddings/annual/2023/emb_tile_*.tif`
- Sample events (proposed CLI):
  - `python -m src.embeddings.sample_events --events outputs/attribution/events_2023.parquet --embeddings data/embeddings/annual/2023/features.vrt --year 2023 --out results/datasets/embeddings_events_2023.parquet --loglevel INFO`
- Run QC analysis (proposed CLI):
  - `python -m src.qc.embeddings_analysis --input results/datasets/embeddings_events_2023.parquet --outdir outputs/qc/embeddings/2023 --loglevel INFO`

Next Steps (minimal code to implement)
- [ ] Add constants in `src/config/constants.py`.
- [ ] Create `src/embeddings/sample_events.py` (zonal stats and parquet write).
- [ ] Create `src/qc/embeddings_analysis.py` (centroids, distances, UMAP/t‑SNE, plots).

