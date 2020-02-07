_Thanks to Steve Olsen for writing this up!_ 

##TRAINING
github.com/dvschultz/stylegan2

*Upload dataset:*
- zip folder and upload to google drive
- get shareable link -> advanced - > On - Public on the web
- copy link [id#]
- link (id# is string between https://drive.google.com/file/d/ and /view?usp=sharing) (mine was: 1kfAKPrHuGSQb93_NIcxWUbaN_cqgfrYx)
- $ `cd stylegan2`
- $ `mkdir raw_datasets`
- $ `pip install gdown`
- $ `cd raw_datasets`
- $ `gdown —id [id#]`
- $ `unzip dataset_name.zip`

*Create custom dataset*
in stylegan2 folder:
$ `python dataset_tool.py create_from_images ~/stylegan2/datasets/dataset_name ./raw_datasets/dataset_name`

*Run training*
In stylegan2 folder:
$ `python run_training.py --num-gpus=1 --data-dir=./datasets --config=config-f --dataset=dataset_name --mirror-augment=False --metrics=None

Run once to check if working

ctrl+c to stop training

Press up to get same command and add nohup to the beginning
 $ `nohup python run_training.py --num-gpus=1 --data-dir=./datasets --config=config-f --dataset=dataset_name --mirror-augment=False --metrics=None`
nohup keeps process running in background

*To Terminate:*
run $ `nvidia-smi`
you will see a list of processes, you want to kill the PID # (column 2) of the one taking up the most GPU (far right)
run $ `kill -9 [PID #]` 
(for example $ `kill -9 4817`)
Run $ `nvidia-smi` again to confirm it stopped running
