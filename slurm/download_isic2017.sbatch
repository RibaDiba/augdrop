#!/bin/bash

#SBATCH --job-name=dropaug-download-isic2017
#SBATCH --nodes=1

## CPU only: download ISIC 2017 (Kaggle moon1570/isic-2017-train-val-test-images-and-masks).
#SBATCH --time=02:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --mem=8gb
#SBATCH --ntasks=1

#SBATCH --output=/fs/gamma-projects/pruneaug/slurm/logs/%x.%j.out
#SBATCH --error=/fs/gamma-projects/pruneaug/slurm/logs/%x.%j.err

#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -eo pipefail

# sbatch slurm/download_isic2017.sbatch
# kagglehub into data/raw/isic2017/ (KAGGLEHUB_CACHE), then lists what it got.
REPO_DIR="${REPO_DIR:-/fs/gamma-projects/pruneaug}"
source "${REPO_DIR}/slurm/common.sh"
group_logs dropaug-download-isic2017
cd "${REPO_DIR}"
activate_env || exit 1

export KAGGLEHUB_CACHE="${REPO_DIR}/data/raw/isic2017"
mkdir -p "${KAGGLEHUB_CACHE}"
python -c "import kagglehub" 2>/dev/null || pip install --user kagglehub
python - <<'PY'
import kagglehub
path = kagglehub.dataset_download("moon1570/isic-2017-train-val-test-images-and-masks")
print("Path to dataset files:", path)
PY
find "${KAGGLEHUB_CACHE}" -maxdepth 6 -type d | head -40
for d in $(find "${KAGGLEHUB_CACHE}" -mindepth 5 -maxdepth 7 -type d); do echo "$d: $(ls "$d" | wc -l) entries, e.g. $(ls "$d" | head -2 | tr '\n' ' ')"; done
