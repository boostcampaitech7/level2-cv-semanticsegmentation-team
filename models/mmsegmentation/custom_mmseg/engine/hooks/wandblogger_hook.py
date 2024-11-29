from mmengine.hooks import Hook
from mmseg.registry import HOOKS
import wandb

@HOOKS.register_module()
class WandBLoggerHook(Hook):
    """Custom Hook for logging metrics and other information to WandB."""
    def __init__(self, interval=10):
        """
        Args:
            interval (int): Frequency (in iterations) to log metrics.
        """
        self.interval = interval

    def before_train(self, runner):
        # Initialize WandB project if not already initialized
        if not wandb.run:
            # Check if it's a Sweep run
            if wandb.run:
                wandb.init()  # Reuse existing Sweep configuration
            else:
                # Regular initialization
                wandb.init(
                    project="semantic_segmentation_mmseg",
                    config=runner.cfg,
                    name=runner.cfg.work_dir.split('/')[-1]  # Use work_dir name as run name
                )
        
        # Update config with runner's configuration for easy traceability
        wandb.config.update(runner.cfg)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.iter % self.interval == 0:
            # Log training metrics
            metrics = {
                "train/loss": outputs['loss'].item(),
                "train/lr": runner.optim_wrapper.param_groups[0]['lr'],
                "train/iter": runner.iter,
                "train/epoch": runner.epoch,
            }
            wandb.log(metrics, step=runner.iter)

    def after_val_epoch(self, runner, metrics):
        if not metrics:
            print("No validation metrics available.")
            return

        # Log validation metrics to WandB
        try:
            wandb.log({"val/mDice": metrics["mDice"]})
            print(f"Logged to WandB: {metrics}")
        except Exception as e:
            print(f"Failed to log to WandB: {e}")

    def after_run(self, runner):
        # Finalize WandB run after training
        wandb.finish()
