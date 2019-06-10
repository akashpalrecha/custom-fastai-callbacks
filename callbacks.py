from fastai.basic_train import LearnerCallback
from fastai.basic_train import Learner

class SaveEveryNIterations(LearnerCallback):
    """Saves model after every N iterations
    We save all models with the same name as otherwise rather heavy models can quickly gobble up
    all available disk space.

    Usage:
    saver_callback = partial(SaveEveryNIterations, num_iterations=100,
                                              save_name="saved_every_100_iterations")
    learn = create_cnn(data, models.resnet18, callback_fns = [saver_callback])
    """
    def __init__(self, learn: Learner, num_iterations: int = 100, save_name=None, disable_callback:bool=False):
        """
        :param num_iterations: Saves model after every `num_iterations` iterations
        :param save_name: [optional] Filename to save model with
        """
        super().__init__(learn)
        self.num_iterations = num_iterations
        self.save_name = save_name
        if save_name is None:
            self.save_name = f'saved_every_{self.num_iterations}_iterations'

    def on_batch_end(self, iteration, **kwargs) -> None:
        if iteration % self.num_iterations == 0 and iteration != 0:
            #TODO : Report to Fastai : param names are different for saving model in language_model_learner
            self.learn.save(self.save_name)
            print(f"Model saved as {self.save_name} | Iteration : {iteration}")


class StopAfterNIterations(LearnerCallback):
    """Stops model after N iterations.
    Usage:
    stopper = partial(StopAfterNIterations, num_iterations = 17)
    learn = create_cnn(data, models.resnet18, callback_fns = [stopper])
    """
    def __init__(self, learn: Learner, num_iterations:int=100, disable_callback:bool=False):
        """
        :param num_iterations: Stops model after every `num_iterations` iterations
        """
        super().__init__(learn)
        self.num_iterations = num_iterations
        self.stop_training = False

    def on_batch_end(self, iteration, **kwargs) -> None:
        if iteration == self.num_iterations:
            print(f"Iteration {iteration} reached. Stopping Training")
            self.stop_training = True
            return {'stop_training': self.stop_training}

    def on_epoch_end(self, **kwargs) ->bool:
        if self.stop_training:
            print('Run learn.validate(learn.data.valid_dl) to see results')
            return {'stop_training': self.stop_training}


class GradientAccumulator(LearnerCallback):
    """Accumulates gradients over N iterations
    Usage:
    accumulator = partial(GradientAccumulator, num_iterations=100)
    learn = create_cnn(data, models.resnet18, callback_fns = [accumulator])
    """
    def __init__(self, learn: Learner, num_iterations: int = 4, disable_callback: bool = False):
        """
        :param num_iterations: Accumulate gradients over `num_iterations` iterations before taking an optimizer step
        """
        super().__init__(learn)
        self.num_iterations = num_iterations
        self.skipped_last_backprop = False

    def on_backward_end(self, iteration, **kwargs) -> None:
        if (iteration % self.num_iterations != 0) or (iteration == 0):
            self.skipped_last_backprop = True
            return {'skip_step': True, 'skip_zero': True}
        else:
            self.skipped_last_backprop = False
            return False

    def on_step_end(self, **kwargs):
        if self.skipped_last_backprop:
            return {'skip_zero': True}
        else:
            return False

    def on_epoch_end(self, **kwargs) ->bool:
        """Deals with the edge case of an epoch ending"""
        if self.skipped_last_backprop:
            self.learn.opt.step()
            self.learn.opt.zero_grad()


class ShowResutsEveryNIterations(LearnerCallback):
    """Shows model results after every N iterations

    Usage:
    results_callback = partial(SaveEveryNIterations, num_iterations=100)
    learn = create_cnn(data, models.resnet18, callback_fns = [results_callback])
    """
    def __init__(self, learn: Learner, num_iterations: int = 100, save_name=None, disable_callback:bool=False):
        """
        :param num_iterations: Show model resuts after every `num_iterations` iterations
        """
        super().__init__(learn)
        self.num_iterations = num_iterations

    def on_batch_end(self, iteration, **kwargs) -> None:
        if iteration % self.num_iterations == 0 and iteration != 0:
            self.learn.show_results()
            self.learn.model.train()