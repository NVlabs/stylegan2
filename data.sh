#----------------------------------SETTINGS------------------------------------#

PROJECT_ID='for-nikita' #ID проекта
BUCKET_NAME='for-nikita' #Имя Bucket
LOAD='DATASETS' #DATASETS (загрузка датасета из GS) \
                #или IMAGES (создание датасета и его загрузка на GS).
PATH_TO_DATASETS='style-GAN2/datasets/256-gray' #Путь до датасета на GS
DATASET_FOLDER="img-gray-256" # Каталог для сохранения датасета на GS
RESOLUTION=256 #Разрешение img 
MODE="gray" #gray or rgb
COUNT=5000 #Кол-во изображений для тестового датасета. 
           #Не передавать аргумент для использования всех изображений

#------------------------------------------------------------------------------#

gcloud auth login
gcloud config set project $PROJECT_ID

cd ./stylegan2/

if [ $LOAD = "IMAGES" ]; then
	gsutil cp gs://$BUCKET_NAME/images.zip ./images.zip;
  unzip -q ./images.zip -d ./;
fi

echo $0
 
full_path=$(realpath $0)
dir_path=$(dirname $full_path)
echo $dir_path

PATH_TO_IMG="${dir_path}/images/"

if [ $LOAD = "IMAGES" ]; then
  echo Обработка изображений
  mkdir -p ./images/custom/;
  python3 preprocessing.py --resolution $RESOLUTION\
                           --mode $MODE\
                           --count $COUNT\
                           --path $PATH_TO_IMG
fi

if [ $LOAD = "IMAGES" ]; then
  echo Создание датасета
  python3 dataset_tool.py create_from_images ./datasets/custom ./images/custom
  rm -rf ./images
  echo Копирование датасета на GS 
  gsutil -m cp -r ./datasets/custom/*.tfrecords gs://for-nikita/style-GAN2/datasets/$DATASET_FOLDER/
fi

if [ $LOAD = "DATASETS" ]; then
  echo Копирование датасета из GS 
	mkdir -p ./datasets/custom/;
  gsutil -m cp -r gs://${BUCKET_NAME}/${PATH_TO_DATASETS}/*.tfrecords ./datasets/custom;
fi







