
class CyclicPlateauScheduler():

    def __init__(self, steps_per_epoch, optimizer, initial_lr=0.001, min_lr=1e-8, min_improve_factor=0.999, lr_patience=0, lr_reduce_factor=0.5, lr_reduce_metric='val_loss'):
        super().__init__()
        self.lr = initial_lr
        self.min_lr = min_lr
        self.min_improve_factor = min_improve_factor
        self.lr_patience = lr_patience
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_reduce_metric = lr_reduce_metric
        self.steps_per_epoch = steps_per_epoch
        self.optimizer = optimizer
        self.best_lr_metric_val = float('inf')
        self.reduce_metric_too_high_count = 0

    def training_step(self, step_index): # cyclic scheduler
        half_steps = self.steps_per_epoch // 2
        min_lr = 1/4 * self.lr
        if step_index < half_steps:
            c = step_index / half_steps
            lr = min_lr + c * (self.lr - min_lr)
        else:
            c = (step_index - half_steps) / half_steps
            lr = self.lr - c * (self.lr - min_lr)
        self.optimizer.param_groups[0]['lr'] = lr

    def validation_epoch_end(self, metrics): # plateau scheduler
        reduce_metric_val = metrics[self.lr_reduce_metric]
        if reduce_metric_val > self.best_lr_metric_val * self.min_improve_factor:
            self.reduce_metric_too_high_count += 1
        if reduce_metric_val < self.best_lr_metric_val:
            self.best_lr_metric_val = reduce_metric_val
        if self.reduce_metric_too_high_count > self.lr_patience:
            self.lr = max(self.lr * self.lr_reduce_factor, self.min_lr)
            self.reduce_metric_too_high_count = 0