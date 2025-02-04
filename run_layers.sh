# export data='metamath'
# export data='gsm8k'
export data='med'
export eval_only=0
export WANDB_PROJECT='med_tune_layers'
wandb online

# export ep=1
# export dp=33

############### Layers ###############
bash run.sh
# l=0-20 bash run.sh
# l=0-10 bash run.sh
# l=0-15 bash run.sh
# l=0-25 bash run.sh
# l=0-30 bash run.sh
# l=10-31 bash run.sh

# tune the continuous layers, 8 layers as a block
for i in {0..31..8}; do
    l=0-$(($i+7)) bash run.sh
    l=31-$((31-i+1)) bash run.sh
done

l=8-23 bash run.sh

# tune every 4 layers
for i in {0..31..4}; do
    l=$i-$(($i+3)) bash run.sh
done

# tune every 8 layers
for i in {0..31..8}; do
    l=$i-$(($i+7)) bash run.sh
done
