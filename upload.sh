DATETIME=`date +"%Y-%m-%d-%H:%M:%S"`

SAVES=./saves/

cd $SAVES
files=(*)

MODEL_FOLDER=${files[-1]}

gsutil -m cp -r $MODEL_FOLDER/ gs://for-nikita/style-GAN2/models/$MODEL_FOLDER'-'$DATETIME