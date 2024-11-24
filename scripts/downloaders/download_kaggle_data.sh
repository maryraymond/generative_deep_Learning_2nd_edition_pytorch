USER=$1
DATASET=$2

set -x  # Enable debugging

cd ./data/ && \
kaggle datasets download -d $USER/$DATASET && \
echo "Unzipping..." \
&& unzip -q -o $DATASET.zip -d ./$DATASET && rm ./$DATASET.zip && \
echo "🚀 Done!"

