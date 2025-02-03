export data='metamath'
# export data='gsm8k'
export WANDB_PROJECT='math_partial_data'
wandb online

export bsd='checkpoints/data_ablate'

############### Data: Total 350k ###############
# ep=1 bash run.sh
# ep=3 dp=33 bash run.sh
# ep=2 dp=50 bash run.sh
# ep=2 dp=75 bash run.sh

############### Data: Total 175k ###############
set -e
ep=1 dp=33 bash run.sh
ep=3 dp=11 bash run.sh
ep=2 dp=17 bash run.sh


############### Layers ###############
export ep=3
export dp=11

l=0-10 bash run.sh
l=0-15 bash run.sh
l=0-20 bash run.sh

for i in {0..31..4}; do
    l=$i-$(($i+3)) bash run.sh
done