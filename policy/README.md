# Signal Controller (Actor)

## Requirement for signal controller classes

The signal controller has to be a class that follows the following structure:

```python
class ControllerName(object):
    def __init__(self):
        pass
    
    def forward(self, observation):
        ...
        return phase_list
```

There must be a ```forward()``` function of the controller that return the phase id of the intersections. 

## Default signal controller

Here is the list of default controller:

### Actuate control (SUMO default model)

### Fixed-time control (pre-timed)

### Max-pressure control

### Max-pressure control trained by RL
