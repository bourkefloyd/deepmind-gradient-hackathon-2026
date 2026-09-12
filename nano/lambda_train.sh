#!/usr/bin/env bash
# Train nano variants on a Lambda GPU box (hackathon requirement). Run from the repo root on the Mac:
#   nano/lambda_train.sh up   ubuntu@<IP>     # ship nano/ + word lists, generate 1M-board data, launch 4 trainers
#   nano/lambda_train.sh log  ubuntu@<IP>     # tail the four logs
#   nano/lambda_train.sh pull ubuntu@<IP>     # copy checkpoints to runs/lambda/<name>/
# Instance used on 2026-09-12: gpu_1x_a100_sxm4 (1x A100 40GB, us-west-2, $1.99/h), Lambda Stack torch.
set -euo pipefail
cmd=${1:?up|log|pull}; host=${2:?user@ip}
SSH="ssh -i $HOME/.ssh/lambda_nano -o StrictHostKeyChecking=no -o ConnectTimeout=15"
BOARDS=${BOARDS:-1000000}
BUDGET=${BUDGET:-33}
BATCH=${BATCH:-1024}

case "$cmd" in
  up)
    tar czf /tmp/nano_ship.tgz nano/*.py data/enable1.txt data/common-30k.txt
    $SSH "$host" 'mkdir -p ~/wh && cd ~/wh && mkdir -p data runs'
    scp -i "$HOME/.ssh/lambda_nano" -o StrictHostKeyChecking=no /tmp/nano_ship.tgz "$host":~/wh/
    $SSH "$host" bash -s <<EOF
set -e
cd ~/wh && tar xzf nano_ship.tgz
python3 -c "import torch, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# data: same recipe as the Mac run (frequency-weighted) at ${BOARDS} boards, plus a len-bonus 1.6 curriculum set
nohup python3 -m nano.data --boards ${BOARDS} --paths 6 --workers 28 --out data/wh_base.npz > runs/data_base.log 2>&1
nohup python3 -m nano.data --boards ${BOARDS} --paths 6 --workers 28 --len-bonus 1.6 --out data/wh_len16.npz > runs/data_len16.log 2>&1
cat runs/data_base.log runs/data_len16.log
# four trainers share the A100 (bf16 autocast + torch.compile: 43k samples/s alone for d6 at batch 1024; fp32 sgemm was 7k); budget-tied cosine so they all finish together
export SWM_ALLOW_CPU=0
nohup python3 -m nano.train --data data/wh_base.npz  --depth 6 --steps 400000 --budget-min ${BUDGET} --batch-size ${BATCH} --val-every 2000 --amp --compile --out runs/a_d6_base  > runs/a_d6_base.log  2>&1 &
nohup python3 -m nano.train --data data/wh_base.npz  --depth 8 --steps 400000 --budget-min ${BUDGET} --batch-size ${BATCH} --val-every 2000 --amp --compile --out runs/b_d8_base  > runs/b_d8_base.log  2>&1 &
nohup python3 -m nano.train --data data/wh_len16.npz --depth 6 --steps 400000 --budget-min ${BUDGET} --batch-size ${BATCH} --val-every 2000 --amp --compile --out runs/c_d6_len16 > runs/c_d6_len16.log 2>&1 &
nohup python3 -m nano.train --data data/wh_len16.npz --depth 8 --steps 400000 --budget-min ${BUDGET} --batch-size ${BATCH} --val-every 2000 --amp --compile --out runs/d_d8_len16 > runs/d_d8_len16.log 2>&1 &
sleep 90
for f in runs/[abcd]_*.log; do echo "== \$f"; grep -v Warning \$f | head -4; grep -v Warning \$f | tail -1; done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
EOF
    ;;
  log)
    $SSH "$host" 'cd ~/wh && for f in runs/[abcd]_*.log; do echo "== $f"; grep -E "^step|val@|budget|saved" $f | tail -3; done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
    ;;
  pull)
    mkdir -p runs/lambda
    for r in a_d6_base b_d8_base c_d6_len16 d_d8_len16; do
      mkdir -p "runs/lambda/$r"
      scp -q -i "$HOME/.ssh/lambda_nano" -o StrictHostKeyChecking=no "$host":~/wh/runs/$r/model.pt "$host":~/wh/runs/$r/train.json "$host":~/wh/runs/$r.log "runs/lambda/$r/" || echo "missing $r"
    done
    ls -la runs/lambda/*/model.pt
    ;;
esac
