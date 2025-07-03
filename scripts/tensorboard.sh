# this script is copied from the Generative Deep Learning 2nd Edition repository 
# at (https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/scripts/tensorboard.sh)

# The original code is available [here](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition) and is licensed under the Apache License 2.0.
# This implementation is distributed under the Apache License 2.0. See the LICENSE file for details._

CHAPTER=$1
EXAMPLE=$2

tensorboard --logdir=./notebooks/$CHAPTER/$EXAMPLE/log --port=6007
