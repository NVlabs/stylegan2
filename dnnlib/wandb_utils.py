import imghdr

import wandb
from wandb.integration.tensorboard import tf_summary_to_dict

class WandbLogger():
    def __init__(self, project, name, config, job_type=None):
        self.run = wandb.init(project=project, name=name, config=config, job_type=job_type) if not wandb.run else wandb.run
        self.log_dict = {}
            
    def log_tf_summary(self, summary):
        tf_log_dict = wandb.integration.tensorboard.tf_summary_to_dict(summary)
        if tf_log_dict:
            for key, value in tf_log_dict.items():
                self.log_dict[key] = value
                
    def log_model_artifact(self, path, step):
        model_artifact = wandb.Artifact('run_'+wandb.run.id+'_checkpoints', type='model', metadata={'cur_nimg': step})
        model_artifact.add_file(path, name='network-snapshot-%06d.pkl' % (step))
        wandb.log_artifact(model_artifact)

    def flush(self):
        wandb.log(self.log_dict)
        self.log_dict = {}
    
    def log(self, log_dict, flush=False):
        for key, value in log_dict.items():
            if imghdr.what(value): # Check if the value is an image
                self.log_dict[key] = wandb.Image(value)
            else:
                self.log_dict[key] = value
        if flush:
            self.flush()
            


