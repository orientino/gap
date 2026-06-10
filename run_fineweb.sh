#!/bin/bash
#SBATCH -J fineweb
#SBATCH --array=1-5
#SBATCH --qos=default
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out

echo -e "--------------------------------"
echo -e "Start:\t $(date)"
echo -e "JobID:\t ${SLURM_JOBID}"
echo -e "Node:\t ${SLURM_NODELIST}"
echo -e "--------------------------------\n"

eval "$(micromamba shell hook --shell bash)"
micromamba activate gap
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

SEED=42
WD=0
BS=1
EP=1
DATA=fineweb
SEQ_LEN=256
ACC_STEPS=1

DIR_DATA=./data
DIR_OUTPUT=./output/${DATA}_d6

MOMS=(0.9 0.95 0.99 0.999)
# OPTS=(adam)
# LRS=(1e-4 1e-3 1e-2 1e-1)
OPTS=(sgd)
LRS=(1e-2 3e-2 1e-1 3e-1 1e-0)

id=$((SLURM_ARRAY_TASK_ID - 1))
LR=${LRS[$id]}

GPU=0
for OPT in "${OPTS[@]}"; do
    for MOM in "${MOMS[@]}"; do
        NAME=${OPT}_lr${LR}_bs$((BS*ACC_STEPS))_mom${MOM}_ep${EP}_seed${SEED}

        CUDA_VISIBLE_DEVICES=$GPU python -m language.train \
            --seed $SEED \
            --bs $BS \
            --lr $LR \
            --wd $WD \
            --mom $MOM \
            --opt $OPT \
            --data $DATA \
            --seq_len $SEQ_LEN \
            --epochs $EP \
            --accum_steps $ACC_STEPS \
            --dir_data $DIR_DATA \
            --dir_output $DIR_OUTPUT \
            --run_name $NAME \
            --n_workers 32 \
            --log_interval 10 \
            --eval_interval 40000 &

        GPU=$((GPU + 1))
        if [ $GPU -eq 4 ]; then
            wait
            GPU=0
        fi
    done
done
wait
