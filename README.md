# custom-fastai-callbacks
Custom Callbacks to extend the Fastai library's functionality
> WORK IN PROGRESS <br>
I'm still learning and will keep updating this repo as I create more callbacks.
<hr>
<hr>

```
class SaveEveryNIterations(LearnerCallback):
    """
    :param num_iterations: Saves model after every `num_iterations` iterations
    :param save_name: [optional] Filename to save model with
    
    Saves model after every N iterations.
    This is useful when training models that take hours to train for just 1 epoch
    We save all models with the same name as otherwise rather heavy models can quickly gobble up
    all available disk space.
    
    Usage: 
    saver_callback = saver_callback = partial(SaveEveryNIterations, num_iterations=100, 
                                              save_name="saved_every_100_iterations")
    learn = create_cnn(data, models.resnet18, callback_fns = [saver_callback])
    """
```
```
class StopAfterNIterations(LearnerCallback):
    """
    :param num_iterations: Saves model after every `num_iterations` iterations
    
    Stops model after N iterations
    This is useful when training models that take hours to train for just 1 epoch
    
    Usage:
    stopper = partial(StopAfterNIterations, num_iterations = 17)
    learn = create_cnn(data, models.resnet18, callback_fns = [stopper])
    """
```