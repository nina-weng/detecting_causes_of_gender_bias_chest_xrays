#!/bin/sh
### General options

### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J NIH
### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
###BSUB -R "select[gpu32gb]"
###BSUB -R "select[sxm2]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 23:50
# request 5GB of system-memory
#BSUB -R "rusage[mem=20GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u ninwe@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/12.0
source ../../../../chexpert_env/bin/activate

#/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery
python3 ./disease_prediction.py -s NIH -d Pneumothorax -f 0 50 100 -n 1 -r 0-10

