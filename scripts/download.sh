# this script is copied from the Generative Deep Learning 2nd Edition repository 
# at (https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/scripts/download.sh)

# The original code is available [here](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition) and is licensed under the Apache License 2.0.
# This implementation is distributed under the Apache License 2.0. See the LICENSE file for details._


DATASET=$1

if [ $DATASET = "faces" ]
then
source ./scripts/downloaders/download_kaggle_data.sh jessicali9530 celeba-dataset
elif [ $DATASET = "bricks" ]
then
source scripts/downloaders/download_kaggle_data.sh joosthazelzet lego-brick-images 
elif [ $DATASET = "recipes" ]
then
source scripts/downloaders/download_kaggle_data.sh hugodarwood epirecipes
elif [ $DATASET = "flowers" ]
then
source scripts/downloaders/download_kaggle_data.sh nunenuh pytorch-challange-flower-dataset
elif [ $DATASET = "wines" ]
then
source scripts/downloaders/download_kaggle_data.sh zynicide wine-reviews
elif [ $DATASET = "cellosuites" ]
then
source scripts/downloaders/download_bach_cello_data.sh
elif [ $DATASET = "chorales" ]
then
source scripts/downloaders/download_bach_chorale_data.sh
else
echo "Invalid dataset name - please choose from: faces, bricks, recipes, flowers, wines, cellosuites, chorales"
fi



