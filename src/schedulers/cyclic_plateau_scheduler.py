
class CyclicPlateauScheduler():

    def __init__(self, initial_lr, min_lr, lr_patience, lr_reduce_factor, lr_reduce_metric, steps_per_epoch, optimizer):
        super().__init__()
        self.lr = initial_lr
        self.min_lr = min_lr
        self.lr_patience = lr_patience
        self.lr_reduce_factor = lr_reduce_factor
        self.lr_reduce_metric = lr_reduce_metric
        self.steps_per_epoch = steps_per_epoch
        self.optimizer = optimizer
        self.last_lr_metric_val = float('inf')
        self.reduce_metric_too_high_count = 0

    def step(self, step_index): # cyclic scheduler
        half_steps = self.steps_per_epoch // 2
        min_lr = 1/10 * self.lr
        if step_index < half_steps:
            c = step_index / half_steps
            lr = min_lr + c * (self.lr - min_lr)
        else:
            c = (step_index - half_steps) / half_steps
            lr = self.lr - c * (self.lr - min_lr)
        self.optimizer.param_groups[0]['lr'] = lr

    def validation_epoch_end(self, val_step_outputs): # plateau scheduler
        reduce_metric_val = sum([output[self.lr_reduce_metric] for output in val_step_outputs]) / len(val_step_outputs)
        if reduce_metric_val > self.last_lr_metric_val * 0.99:
            self.reduce_metric_too_high_count += 1
        if self.reduce_metric_too_high_count > self.lr_patience:
            self.lr = max(self.lr * self.lr_reduce_factor, self.min_lr)
            self.reduce_metric_too_high_count = 0
        self.last_lr_metric_val = reduce_metric_val