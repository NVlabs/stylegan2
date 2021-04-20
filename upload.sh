# PROJECT_ID='for-nikita'
# BUCKET_NAME='for-nikita'

# gcloud auth login
# gcloud config set project $PROJECT_ID

# MODEL_PATH = ''
DATETIME=`date +"%Y-%m-%d-%H:%M:%S"`

# gsutil -m cp -r $MODEL_PATH gs://for-nikita/style-GAN2/models/{model_name}

SAVES=./saves/

# for entry in `ls $SAVES`; do
#   echo "$entry"
# done

cd $SAVES
files=(*)

MODEL_FOLDER=${files[-1]}

# cd $MODEL_FOLDER
# files=(*.pkl)
# MODEL=${files[-1]}

gsutil -m cp -r $MODEL_FOLDER/ gs://for-nikita/style-GAN2/models/$MODEL_FOLDER'-'$DATETIME