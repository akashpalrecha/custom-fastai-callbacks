from fastai import LearnerCallback
from fastai import Learner
from fastai import Any

class SaveEveryNIterations(LearnerCallback):
    """
    Saves model after every N iterations
    This is useful when training models that take hours for just 1 epoch
    We save all models with the same name as otherwise rather heavy models can quickly gobble up
    all available disk space.
    :param num_iterations: Saves model after every `num_iterations` iterations
    :param save_name: [optional] Filename to save model with
    """
    def __init__(self, learn: Learner, num_iterations: int = 100, save_name=None):
        super().__init__(learn)
        self.num_iterations = num_iterations
        self.save_name = save_name
        if save_name is None:
            self.save_name = f'saved_every_{self.num_iterations}_iterations'

    def on_batch_end(self, iteration, **kwargs) -> None:
        if iteration % self.num_iterations == 0 and iteration != 0:
            self.learn.save(name=self.save_name)
            print(f"Model saved as {self.save_name} | Iteration : {iteration}")