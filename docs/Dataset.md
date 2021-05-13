# Dataset Structure

We use HDF5 files in order to store all of the training and testing data
for SPANet. The structure of these files is closely related with your
[event definition](./EventInfo.md).

## HDF5 Structure

We use a two-level HDF5 structure within each file which conforms to the following tree structure. 
Next to each key is the expected shape and data type for each array in the file. 

```
- source
---- mask      : [NUM_EVENTS, MAX_JETS]   : bool
---- FEATURE_1 : [NUM_EVENTS, MAX_JETS]   : float or double
---- FEATURE_2 : [NUM_EVENTS, MAX_JETS]   : float or double
---- FEATURE_3 : [NUM_EVENTS, MAX_JETS]   : float or double
---- ...

- PARTICLE_1
---- mask      : [NUM_EVENTS, ]           : bool
---- JET_1     : [NUM_EVENTS, ]           : int or long
---- JET_2     : [NUM_EVENTS, ]           : int or long
---- ...

- PARTICLE_2
---- mask      : [NUM_EVENTS, ]           : bool
---- JET_1     : [NUM_EVENTS, ]           : int or long
---- JET_2     : [NUM_EVENTS, ]           : int or long
---- ...

- ...
```

## Feature, Particle, and Jet Names
The ALL_CAPS values in the structure above refer to symbols which you define
as part of your event `.ini` file. These names must match exactly to those
defined in the `.ini` file in order for the dataset to be parsed correctly.
The ordering does not matter in either the event file or the dataset, but
the network outputs will match the ordering in the event file rather than 
the HDF5 file.


## Source Masks
```
- source
---- mask      : [NUM_EVENTS, MAX_JETS]   : bool
```

The source mask array is necessary because our network expected padded
events on the input. That is, regardless of the number of jets in each 
event, you must store all events to have `MAX_JETS` different values
and then mark each jet as either a real jet with a `True` value in the 
mask array or mark it as a padding jet with a `False` value in the mask array.

## Source Features
`---- FEATURE_1 : [NUM_EVENTS, MAX_JETS]   : float or double`

The feature arrays simply contain the value for each feature and each
jet in your dataset. Padded jets can have any value for their features, 
but typically you just store a 0 for any padded values.

## Particle Masks
```
- PARTICLE_1
---- mask      : [NUM_EVENTS, ]           : bool
```

The particle masks indicate whether a given particle is present, or 
is fully reconstructable, in a given event. In order for this mask value to be
`True`, all of the jets associated with the particle must be present in the event.
This value should be `False` otherwise.

## Particle Jets
`---- JET_1     : [NUM_EVENTS, ]           : int or long`

The jet arrays contain the indicies of each jet as determined by your feature
ordering. So if `JET_1` of `PARTICLE_1` is the second jet event 0, then 
`file["PARTICLE_1/JET_1"][0] = 1` because the index of the second jet is 1.

The value of jets not present in the event should `-1`.

## Example
Refer to the Example Dataset section of the [`ttbar` Example Guide](TTBar.md) for a
description of the `ttbar` example HDF5 layout.
