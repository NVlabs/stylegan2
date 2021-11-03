#----------------------------------SETTINGS------------------------------------#

DATASET='custom' #Каталог с датасетом
DATA_DIR='./datasets' #Путь до каталога с датасетом
CONFIG_ID='config-f' #Конфигурация модели. По умолчанию config-e - большая сеть 
NUM_GPUS=1 #Кол-во gpu
TOTAL_KIMG=100000 #Общая продолжительность обучения, измеряемая тысячами 
                  #реальных изображений на один цикл.
MIRROR_AUGMENT=false #Зеркальная аугментация
MINIBATCH_SIZE=8 #Размер мини-пакета. По умолчанию 32
RESOLUTION=256 #Разрешение изображений
RESULT_DIR='./saves/' #Каталог для сохранения результатов оубчения
PATH_TO_MODEL='for-nikita/style-GAN2/models/network-snapshot-015056.pkl' #Путь до модели на GS
PRETRAINED=true #Если TRUE, то модель продолжает обучаться
TRAIN=true #TRUE-обучение

#------------------------------------------------------------------------------#

IFS="/" read -a SPLIT_PATH <<< $PATH_TO_MODEL

if [ $PRETRAINED = true ]; then
mkdir ./model/
	gsutil -m cp -r gs://$PATH_TO_MODEL ./model/
  RESUME_PKL="./model/${SPLIT_PATH[-1]}"
else
  RESUME_PKL=' '
fi



if [ $TRAIN = true ]; then
  cd ./stylegan2
  python3 run_training.py --dataset $DATASET\
                          --data_dir $DATA_DIR\
                          --config_id $CONFIG_ID\
                          --num_gpus $NUM_GPUS\
                          --total_kimg $TOTAL_KIMG\
                          --mirror_augment $MIRROR_AUGMENT\
                          --minibatch_size_base $MINIBATCH_SIZE\
                          --resolution $RESOLUTION\
                          --result_dir $RESULT_DIR\
                          --resume_pkl $RESUME_PKL
fi

