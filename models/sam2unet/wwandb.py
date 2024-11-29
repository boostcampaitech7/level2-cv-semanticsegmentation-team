import wandb

def set_wandb(configs):
    if not isinstance(configs, dict):
        configs = vars(configs)
    wandb.login(key=configs['api_key'])
    wandb.init(
        entity=configs['team_name'],
        project=configs['project_name'],
        name=configs['experiment_detail'],
        config={
        #         'model': configs['model_name'],
        #         'resize': configs['image_size'],
                'batch_size': configs['batch_size'],
        #         'loss_name': configs['loss_name'],
        #         'scheduler_name': configs['scheduler_name'],
                'learning_rate': configs['lr'],
                'epoch': configs['epoch']
            }
    )