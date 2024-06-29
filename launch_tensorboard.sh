#!/use/bin/bash

# ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9 
tensorboard --logdir=/root/autodl-tmp --port=6007