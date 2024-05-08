if [ "$1" == "" ]; then
    echo "Usage: ./run.sh <nlp|mm>"
    exit 1
fi
conda activate deep
tensorboard --logdir:runs & python -m main.py --type $1