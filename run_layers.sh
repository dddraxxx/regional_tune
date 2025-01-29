export data='metamath'
# export data='gsm8k'

############### Layers ###############
bash run.sh
l=0-20 bash run.sh
l=0-10 bash run.sh
l=0-15 bash run.sh
# l=0-25 bash run.sh
# l=0-30 bash run.sh
# l=10-31 bash run.sh

# tune every 4 layers
# for i in {0..31..4}; do
#     l=$i-$(($i+3)) bash run.sh
# done

# tune every 8 layers
# for i in {0..31..8}; do
#     l=$i-$(($i+7)) bash run.sh
# done
