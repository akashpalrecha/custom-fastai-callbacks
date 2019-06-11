# Fast-Callbacks
Custom Callbacks to extend the Fastai library's functionality <br>
(**Feel free to contribute to this project in whatever ways you like !!**)
<hr>
This package contains generalized callbacks that be used by anyone using the fastai library (V1).<br>
Currently there are 3 callbacks here that I found useful when training models over huge datasets or when the 
models are too big to increase the batch size over a small number.
I plan to keep adding callbacks to this repository as and when I make them for my own use. 
I'm still and always learning and welcome any changes and feedback to my project.
<hr>

1. **GradientAccumulator**: Accumulates gradients over N iterations before performing an optimizer step.<br>
    This implementation does not solve the subtle issue of BatchNorm layers during gradient accumulation.
    We simply skip over optimizer steps for N iterations and accumulate gradients before doing the actual step.<br>
    Usage:
    
```python
    from callbacks import GradientAccumulator
    accumulator = partial(GradientAccumulator, num_iterations=4)
    learn = create_cnn(data, models.resnet18, callback_fns = [accumulator])
```
2. **SaveEveryNIterations**: Saves model after every N iterations<br>
    We save all models with the same name as otherwise the models could use up too much memory.
    Usage:
```python
    from callbacks import SaveEveryNIterations
    saver_callback = partial(SaveEveryNIterations, num_iterations=100, 
                                              save_name="saved_every_100_iterations")
    learn = create_cnn(data, models.resnet18, callback_fns = [saver_callback])
```

3. **StopAfterNIterations** : Stops model after N iterations.<br>
    Usage:
    
```python
    from callbacks import StopAfterNIterations
    stopper = partial(StopAfterNIterations, num_iterations = 17)
    learn = create_cnn(data, models.resnet18, callback_fns = [stopper])
```

4. **ShowResultsEveryNIterations**: Shows model results after every N iterations<br>
    Usage:
    
```python
    from callbacks import ShowResutsEveryNIterations
    results_callback = partial(ShowResutsEveryNIterations, num_iterations=100)
    learn = create_cnn(data, models.resnet18, callback_fns = [results_callback])
```

5. **SkipNIterations**: Skips first N iterations while training<br>
    Usage:
    
```python
    from callbacks import SkipNIterations
    skipper = partial(SkipNIterations, num_iterations=100)
    learn = create_cnn(data, models.resnet18, callback_fns = [skipper])
```