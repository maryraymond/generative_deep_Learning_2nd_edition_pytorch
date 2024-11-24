CHAPTER=$1
EXAMPLE=$2

tensorboard --logdir=./notebooks/$CHAPTER/$EXAMPLE/log --port=6007
