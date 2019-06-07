from fastai import LearnerCallback
from fastai import Learner
from fastai import Any


class SaveEveryNIterations(LearnerCallback):
    """
    :param num_iterations: Saves model after every `num_iterations` iterations
    :param save_name: [optional] Filename to save model with
    :param disable_callback : set to True to disable callback functionality

    Saves model after every N iterations
    This is useful when training models that take hours to train for just 1 epoch
    We save all models with the same name as otherwise rather heavy models can quickly gobble up
    all available disk space.

    Usage:
    saver_callback = saver_callback = partial(SaveEveryNIterations, num_iterations=100,
                                              save_name="saved_every_100_iterations")
    learn = create_cnn(data, models.resnet18, callback_fns = [saver_callback])

    # To change number of iterations:
    learn.stop_after_n_iterations.num_iterations = new_value

    # To disable callback functionality:
    learn.stop_after_n_iterations.disable_callback = True
    """
    def __init__(self, learn: Learner, num_iterations: int = 100, save_name=None, disable_callback:bool=False):
        super().__init__(learn)
        self._num_iterations = num_iterations
        self.save_name = save_name
        self._disable_callback = disable_callback
        if save_name is None:
            self.save_name = f'saved_every_{self._num_iterations}_iterations'

    def on_batch_end(self, iteration, **kwargs) -> None:
        if self._disable_callback: return False
        if iteration % self._num_iterations == 0 and iteration != 0:
            self.learn.save(name=self.save_name)
            print(f"Model saved as {self.save_name} | Iteration : {iteration}")

    @property
    def num_iterations(self):
        return self._num_iterations

    @property
    def disable_callback(self):
        return self._disable_callback

    @num_iterations.setter
    def num_iterations(self, new_val):
        self.num_iterations = new_val

    @disable_callback.setter
    def disable_callback(self, new_val):
        self.disable_callback = new_val


class StopAfterNIterations(LearnerCallback):
    """
    :param num_iterations: Saves model after every `num_iterations` iterations
    :param disable_callback : set to True to disable callback functionality

    Stops model after N iterations
    This is useful when training models that take hours to train for just 1 epoch

    Usage:
    stopper = partial(StopAfterNIterations, num_iterations = 17)
    learn = create_cnn(data, models.resnet18, callback_fns = [stopper])

    # To change number of iterations:
    learn.stop_after_n_iterations.num_iterations = new_value
    # To disable callback functionality:
    learn.stop_after_n_iterations.disable_callback = True
    """
    def __init__(self, learn: Learner, num_iterations:int=100, disable_callback:bool=False):
        super().__init__(learn)
        self._num_iterations = num_iterations
        self.stop_training = False
        self._disable_callback = disable_callback

    def on_batch_end(self, iteration, **kwargs) -> None:
        if self._disable_callback: return False
        if iteration == self._num_iterations:
            print(f"Iteration {iteration} reached. Stopping Training")
            self.stop_training = True
            return {'stop_training': self.stop_training}

    def on_epoch_end(self, **kwargs) ->bool:
        if self._disable_callback: return False
        if self.stop_training:
            print('Run learn.validate(learn.data.valid_dl) to see results')
            return {'stop_training': self.stop_training}

    @property
    def num_iterations(self):
        return self._num_iterations

    @property
    def disable_callback(self):
        return self._disable_callback

    @num_iterations.setter
    def num_iterations(self, new_val):
        self.num_iterations = new_val

    @disable_callback.setter
    def disable_callback(self, new_val):
        self.disable_callback = new_val


class GradientAccumulator(LearnerCallback):
    """
    :param num_iterations: Accumulate gradients over `num_iterations` iterations before backpropagating
    :param disable_callback : set to True to disable callback functionality

    Accumulates gradients over N iterations
    This is useful when training models where it isn't possible to increase batch size above 1 or 2
    Accumulating gradients solves the issue of unstable gradients
    Usage:
    accumulator = partial(GradientAccumulator, num_iterations=100)
    learn = create_cnn(data, models.resnet18, callback_fns = [accumulator])

    # To change number of iterations:
    learn.stop_after_n_iterations.num_iterations = new_value

    # To disable callback functionality:
    learn.stop_after_n_iterations.disable_callback = True
    """
    def __init__(self, learn: Learner, num_iterations: int = 4, disable_callback: bool = False):
        super().__init__(learn)
        self._num_iterations = num_iterations
        self._disable_callback = disable_callback
        self.skipped_last_backprop = False

    def on_backward_end(self, iteration, **kwargs) -> None:
        if self._disable_callback: return False
        if (iteration % self._num_iterations != 0) or (iteration == 0):
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


    @property
    def num_iterations(self):
        return self._num_iterations

    @property
    def disable_callback(self):
        return self._disable_callback

    @num_iterations.setter
    def num_iterations(self, new_val):
        self.num_iterations = new_val

    @disable_callback.setter
    def disable_callback(self, new_val):
        self.disable_callback = new_val