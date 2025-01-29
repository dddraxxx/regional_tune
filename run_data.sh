export data='metamath'
# export data='gsm8k'

############### Data ###############
export bsd='checkpoints/data_ablate'
ep=1 bash run.sh
ep=3 part=33 bash run.sh
ep=2 part=50 bash run.sh
ep=2 part=75 bash run.sh