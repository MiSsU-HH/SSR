CONFIG=$1

WORK_DIR=${WORK_DIR:-"./work_logs"}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29700}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS=${GPUS:-4}
VISIBLE_GPUS=${VISIBLE_GPUS:-"4,5,6,7"}

export CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/eval.py \
    --config $CONFIG \
    --work-dir $WORK_DIR \
    --launcher pytorch \
    ${@:4}