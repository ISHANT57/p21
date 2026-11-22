# Source of the other pipeline's numbers

Summaries of `ayush_pipeline/results/runs_rgb.csv` and `runs_ms_a.csv` to `runs_ms_d.csv` (pipeline by Ayush Kumar,
a member of this team). Rebuild with `python scripts/crosscheck_pipelines.py --other-repo ayush_pipeline`.
Protocol of that pipeline: stratified 80/10/10 split (2,700 test images), 3 seeds. The result files also hold
macro-F1; the cross-check uses test accuracy only.
Band groups of the ablation (BAND_GROUPS in ayush_pipeline/src/data.py, storage order with B8A last):
rgb_ms = B04,B03,B02 from the GeoTIFF files; rgb+nir = + B08; rgb+nir+rededge = + B08, B05, B06, B07, B8A;
rgb+nir+swir = + B08, B11, B12; all13 = every band. The red-edge and SWIR groups are cumulative: both contain B08 as well.
Mean and sample std over the seeds. Not merged with our own results: the protocols differ.
