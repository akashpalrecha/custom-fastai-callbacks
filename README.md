# Fast-Callbacks
Custom Callbacks to extend the Fastai library's functionality <br>
(*Feel free to contribute to this project in whatever ways you like !!*)
<hr>
This package contains generalized callbacks that be used by anyone using the fastai library (V1).<br>
Currently there are 3 callbacks here that I found useful when training models over huge datasets or when the 
models are too big to increase the batch size over a small number.
I plan to keep adding callbacks to this repository as and when I make them for my own use. 
I'm still and always learning and welcome any changes and feedback to my project.
<hr>
1. GradientAccumulator: Accumulates gradients over N iterations before performing an optimizer step.<br>
    This is useful when training models where it isn't possible to increase batch size above 1 or 2.
    This implementation does not solve the subtle issue of BatchNorm layers during gradient accumulation.
    We simply skip over optimizer steps for N iterations and accumulate gradients before doing the actual step.<br>
    Usage:
    
```python
    from callbacks import GradientAccumulator
    accumulator = partial(GradientAccumulator, num_iterations=4)
    learn = create_cnn(data, models.resnet18, callback_fns = [accumulator])
```
2. SaveEveryNIterations: Saves model after every N iterations<br>
    This is useful when training models that take hours to train for just 1 epoch
    We save all models with the same name as otherwise the models could use up too much memory.
    Also, there is no option to save the best model as that would require validating multiple times
    in the middle of training and also because the losses won't be stable over fractions of a an epoch. <br>
    Usage:
```python
    from callbacks import SaveEveryNIterations
    saver_callback = saver_callback = partial(SaveEveryNIterations, num_iterations=100, 
                                              save_name="saved_every_100_iterations")
    learn = create_cnn(data, models.resnet18, callback_fns = [saver_callback])
```

3. StopAfterNIterations : Stops model after N iterations.<br>
    This is useful when training models that take hours to train for just 1 epoch. We may want to 
    train the model only for a few iterations to test out a few things rather than training for the whole
    epoch.<br>
    Usage:
```python
    from callbacks import StopAfterNIterations
    stopper = partial(StopAfterNIterations, num_iterations = 17)
    learn = create_cnn(data, models.resnet18, callback_fns = [stopper])
```