_Thanks to Steve Olsen for writing this up!_ 

## TRAINING

Video demo for training steps on [YouTube](https://www.youtube.com/watch?v=69YOjyAxum0)

### Upload dataset:
1. zip folder and upload to google drive
2. get shareable link -> advanced - > On - Public on the web
3. copy link [id#]
4. link (id# is string between https://drive.google.com/file/d/ and /view?usp=sharing)
5. $ `cd stylegan2`
6. $ `mkdir raw_datasets`
7. $ `pip install gdown`
8. $ `cd raw_datasets`
9. $ `gdown —id [id#]`
10. $ `unzip dataset_name.zip`

### Create custom dataset

in stylegan2 folder:
$ `python dataset_tool.py create_from_images ~/stylegan2/datasets/dataset_name ./raw_datasets/dataset_name`

### Run training

1. In stylegan2 folder: $ `python run_training.py --num-gpus=1 --data-dir=./datasets --config=config-f --dataset=dataset_name --mirror-augment=False --metrics=None`
2. Run once to check if working
3. ctrl+c to stop training
4. Press up to get same command and add nohup to the beginning
 $ `nohup python run_training.py --num-gpus=1 --data-dir=./datasets --config=config-f --dataset=dataset_name --mirror-augment=False --metrics=None`
nohup keeps process running in background

### To Terminate:

1. run $ `nvidia-smi`
2. you will see a list of processes, you want to kill the PID # (column 2) of the one taking up the most GPU (far right)
3. run $ `kill -9 [PID #]` (for example $ `kill -9 4817`)
4. run $ `nvidia-smi` again to confirm it stopped running

## TESTING

Video demo for testing (in Colab) on [YouTube](https://www.youtube.com/watch?v=-6p0zwHc5-8)
