source $(conda info --base)/etc/profile.d/conda.sh
conda activate swm

python eval.py --config-name=tworoom.yaml policy=tworoom/lewm