# Data storage format

The data (and metadata for partitions) of the data should be kept in this directory. There are multiple APIs for storing
data, here we decribe the storage method with csv files specifying partitions.

## CSV dataset format

- Keep all `*.exe, *.dll` files in a subdirectory (e.g. `binaries/` is used in the sample config files).
- For each partition of the data, keep a `partition.csv`, containing 4 columns (with header)
  - `path`: A path relative to the root directory specified in yaml files, specifying where to locate the binary executables.
  - `metadata_path`: A path relative to the root directory specified in yaml files, specifying where to locate the corresponding metadata files. This can be obtained by executing the `src/preprocess_pe.py` (see `-h` of `process_pe.py`)
  - `target`: Any string representing a specific class (e.g. `benign`, `malicious`).
  - `class`: Integer corresponding to the `target` specified above (e.g. `0`, `1`).
- To run the sample scripts, `train.csv`, `valid.csv` and `test.csv` need to be provided for training, validation, calibration and certification of RS-Del
