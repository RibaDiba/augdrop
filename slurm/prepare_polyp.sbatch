#!/bin/bash

#SBATCH --job-name=dropaug-prepare-polyp
#SBATCH --nodes=1

## CPU only: download PraNet's TrainDataset / TestDataset and write data/polyp.
#SBATCH --time=02:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --mem=16gb
#SBATCH --ntasks=2

#SBATCH --output=/fs/gamma-projects/pruneaug/slurm/logs/%x.%j.out
#SBATCH --error=/fs/gamma-projects/pruneaug/slurm/logs/%x.%j.err

#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -eo pipefail

# ==========================================================================
# prepare_polyp.sbatch -- the PraNet polyp benchmark, ready for configs/dataset/polyp.yaml.
#
#   sbatch slurm/prepare_polyp.sbatch
#
# 1. downloads the two Google Drive files from the PraNet README into
#    data/raw/polyp_pranet/ (skipped when the zip is already there) and logs their sha256;
# 2. unzips them;
# 3. scripts/prepare_polyp.py -> data/polyp/{img_dir,ann_dir}/{train,val} and
#    data/polyp_zeroshot/<colondb|etis|cvc300>/..., asserting 1450/162/380/196/60.
#
# Env knobs: FORCE=1 replaces an existing data/polyp (prepare_polyp.py --force).
# ==========================================================================
REPO_DIR="${REPO_DIR:-/fs/gamma-projects/pruneaug}"
source "${REPO_DIR}/slurm/common.sh"
group_logs dropaug-prepare-polyp
cd "${REPO_DIR}"
activate_env || exit 1

RAW="${REPO_DIR}/data/raw/polyp_pranet"
mkdir -p "${RAW}"

# PraNet README: "Google Drive Link (399.5MB)" and "(327.2MB)"
TRAIN_ID=1YiGHLw4iTvKdvbT6MgwO9zcCv8zJ_Bnb
TEST_ID=1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao

python -c "import gdown" 2>/dev/null || pip install --user gdown
for spec in "TrainDataset:${TRAIN_ID}" "TestDataset:${TEST_ID}"; do
    name="${spec%%:*}"; id="${spec##*:}"
    if [ ! -f "${RAW}/${name}.zip" ]; then
        python -m gdown "https://drive.google.com/uc?id=${id}" -O "${RAW}/${name}.zip"
    fi
    sha256sum "${RAW}/${name}.zip" | tee -a "${RAW}/SHA256SUMS.txt"
    [ -d "${RAW}/${name}" ] || unzip -q -o "${RAW}/${name}.zip" -d "${RAW}"
done

python scripts/prepare_polyp.py --raw "${RAW}" ${FORCE:+--force}
