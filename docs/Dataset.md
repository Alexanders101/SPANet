# Dataset Structure

We use HDF5 files in order to store all of the training and testing data
for SPANet. The structure of these files is closely related with your
[event definition](./EventInfo.md).

## HDF5 Structure

We multi-level HDF5 structure within each file which conforms to the following tree structure. 
Next to each key is the expected shape and data type for each array in the file.
Special keys which must be exactly as shown will be in `CAPITALCASE`.
Custom keys which may modified for your event will be in `lower_case_with_underscores`

```
- INPUTS
---- sequential_input_1:
------- MASK      : [NUM_EVENTS, MAX_ELEMENTS_1]   : bool
------- feature_1 : [NUM_EVENTS, MAX_ELEMENTS_1]   : float
------- feature_2 : [NUM_EVENTS, MAX_ELEMENTS_1]   : float
------- feature_3 : [NUM_EVENTS, MAX_ELEMENTS_1]   : float

---- sequential_input_2:
------- MASK      : [NUM_EVENTS, MAX_ELEMENTS_2]   : bool
------- feature_1 : [NUM_EVENTS, MAX_ELEMENTS_2]   : float
------- feature_2 : [NUM_EVENTS, MAX_ELEMENTS_2]   : float
------- feature_3 : [NUM_EVENTS, MAX_ELEMENTS_2]   : float

---- global_input_1:
------- feature_1 : [NUM_EVENTS, ]   : float
------- feature_2 : [NUM_EVENTS, ]   : float
------- feature_3 : [NUM_EVENTS, ]   : float
------- ...

- TARGETS
---- event_particle_1:
------- decay_product_1 : [NUM_EVENTS, ]   : int or long
------- decay_product_2 : [NUM_EVENTS, ]   : int or long
------- ...
        
---- event_particle_2:
------- decay_product_1 : [NUM_EVENTS, ]   : int or long
------- decay_product_2 : [NUM_EVENTS, ]   : int or long
------- ...

- REGRESSIONS
... (Explained Later)

- CLASSIFICATIONS
... (Explained Later)

```

## Feature, Particle, and Jet Names
The `lower_case_with_underscores` values in the structure above 
refer to symbols which you define as part of your event `.yaml` file. 
These names must match exactly to those defined in the `.yaml` file in 
order for the dataset to be parsed correctly. The ordering does not 
matter in either the event file or the dataset, but the network outputs 
will match the ordering in the event file rather than the HDF5 file.


## Sequential Masks
```
- INPUTS
---- sequential_input_1:
------- MASK      : [NUM_EVENTS, MAX_ELEMENTS]   : bool
```

The source mask array is necessary because our network expected padded
events on the input. That is, regardless of the number of jets in each 
event, you must store all events to have `MAX_ELEMENTS` different values
and then mark each input as either a real jet with a `True` value in the 
mask array or mark it as a padding input with a `False` value in the mask array.

## Sequential Features
`------- feature_1 : [NUM_EVENTS, MAX_ELEMENTS]   : float`

The feature arrays simply contain the value for each feature and each
jet in your dataset. Padded jets can have any value for their features, 
but typically you just store a 0 for any padded values.
the maximum number of elements in every event **must** remain the
same for each sequential input. However, sequential inputs
may have a different number elements between them.

## Global Features
```
---- global_input_1:
------- feature_1 : [NUM_EVENTS, ]   : float
```

Global features have very similar structure to sequential features.
The only real difference is that they are missing the element dimension.
Additionally, we assume that global features are always present for every event
and so we do not have a mask for global inputs.

## Assignment Targets
```
- TARGETS
---- event_particle_1:
------- decay_product_1 : [NUM_EVENTS, ]   : int or long
```

The assignment target arrays contain the indicies of each assignment. 
Only sequential inputs may be assignment targets.
Sequential inputs are concatenated in the order they are defined, and the
assignment target should be the index of sequential input corresponding to that target.

For example if `decay_product_1` of `event_particle_1` is the 
second element of `sequential_input_1` in the first event, then 
`file["TARGETS/event_particle_1/decay_product_1"][0] = 1` because the index of the second  is 1.

If `decay_product_3` of `event_particle_2` is the 
**third** element of `sequential_input_2` in the **second** event, then 
`file["TARGETS/event_particle_2/decay_product_3"][1] = MAX_ELEMENTS_1 + 2`.
Notice that we have to add the maximum possible elements in the first sequential input.
This is becuase all sequential inputs are concatenated.

Any targets which are missing within an event should be marked with `-1`.

## Regressions
Regression targets follow the same structure specified in the event file.
Every regression target will contain a single real value for every event.
You may mask regression targets with a `nan` value. Targets which are `nan` will
be ignored in the regression loss.
```
- REGRESSIONS
---- EVENT:
------- event_regression_target_1 : [NUM_EVENTS, ] : float
------- event_regression_target_2 : [NUM_EVENTS, ] : float
------- ...
---- event_particle_1:
------- PARTICLE:
---------- particle_regression_target_1 : [NUM_EVENTS, ] : float
---------- particle_regression_target_2 : [NUM_EVENTS, ] : float
---------- ...
------- decay_product_1:
---------- particle_regression_target_1 : [NUM_EVENTS, ] : float
---------- particle_regression_target_2 : [NUM_EVENTS, ] : float
---------- ...
------- decay_product_2:
---------- particle_regression_target_1 : [NUM_EVENTS, ] : float
---------- particle_regression_target_2 : [NUM_EVENTS, ] : float
---------- ...
------- ...
---- ...
```

## Classifications
Classification targets will follow the same structure as the regression
data and classification event specification. Every classification target 
will contain a single integer for every event indicating the assigned class.
You may mask classification targets with a `-1` value.

```
- CLASSIFICATIONS
---- EVENT:
------- event_classification_target_1 : [NUM_EVENTS, ] : int or long
------- event_classification_target_2 : [NUM_EVENTS, ] : int or long
------- ...
---- event_particle_1:
------- PARTICLE:
---------- particle_classification_target_1 : [NUM_EVENTS, ] : int or long
---------- particle_classification_target_2 : [NUM_EVENTS, ] : int or long
---------- ...
------- decay_product_1:
---------- particle_classification_target_1 : [NUM_EVENTS, ] : int or long
---------- particle_classification_target_2 : [NUM_EVENTS, ] : int or long
---------- ...
------- decay_product_2:
---------- particle_classification_target_1 : [NUM_EVENTS, ] : int or long
---------- particle_classification_target_2 : [NUM_EVENTS, ] : int or long
---------- ...
------- ...
---- ...
```

## Example
Refer to the Example Dataset section of the [`ttbar` Example Guide](TTBar.md) for a
description of the `ttbar` example HDF5 layout.
