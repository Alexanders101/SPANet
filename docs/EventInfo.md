# Event Specification Format

The first step to training SPANets is to define the 
topology of your target event. To do this, `SPANet` 
uses a definition `.ini` file which contains the features
and jet information for your event.

The structure of the `.ini` file always follows this format:
```
[SOURCE]
FEATURE_1 = FEATURE_OPTION
FEATURE_2 = FEATURE_OPTION
FEATURE_3 = FEATURE_OPTION
...

[EVENT]
particles = (PARTICLE_1, PARTICLE_2, ...)
permutations = EVENT_SYMMETRY_GROUP

[PARTICLE_1]
jets = (JET_1, JET_2, ...)
permutations = JET_SYMMETRY_GROUP

[PARTICLE_2]
jets = (JET_1, JET_2, ...)
permutations = JET_SYMMETRY_GROUP

...
```

We will now go over each of the capitalized options to explain what
they can represent. You may also view an example of a complete ini file
in [`ttbar.ini`](../event_files/old/ttbar.ini).

## SOURCE
One of the two **required** headers is `[SOURCE]` which will contain
all the features that you want to use during training.
Simply give each feature a unique name, since they will later be
used in how you define your dataset. The exact names are not important
as long as they are unique.

the `FEATURE_OPTION` defines how you want `SPANet` to pre-process the
given features. Valid options include:

| FEATURE_OPTION     | Description |
| :--------------:   | ----------- |
| `none`             | No pre-processing applied            |
| `log`              | Scale the feature on a `log` scale | 
| `normalize`        | Normalize the feature based on training dataset statistics |
| `log_normalize`    | First apply a `log` scale and then normalize |

## EVENT
The second **required** header. This will contain all the event
particle and symmetry information. We define "particles" to be
the collections of jets that you are interested in predicting.

--------------------------------------------
`particles = (PARTICLE_1, PARTICLE_2, ...)`

This line defines the set of particles that will be present in your event.
Like with the features, the names you give these particles is arbitrary,
so just make sure they are unique.

-----------------------------------
`permutations = EVENT_SYMMETRY_GROUP`

See [Symmetry Group Description](#Symmetry-Group-Description).

----------------------------

## Particle Definitions
The remaining headers in the ini file must be named on of the particles
that you defined in the `[EVENT]` section. Each of these sections will
then contain the jet names and their respective jet symmetry groups.

--------------
`[PARTICLE_1]`

The header with a matching name to one of the particles defined in `[EVENT]`

----------------------------
`jets = (JET_1, JET_2, ...)`

A set of jet names assigned to this particle. The names have to be unique
within a given particle, but they do not need to be unique
across different particles.

----------------------------
`permutations = JET_SYMMETRY_GROUP`

Again, see [Symmetry Group Description](#Symmetry-Group-Description).

----------------------------


## Symmetry Group Description

For now, we only support describing permutation groups as products
of complete symmetry goups G = S_1 x S_2 x ...

In order to define these permutation groups, you simply describe which
particles or jets belong to each of the fully symmetric groups. This is
expressed as a list of tuples each of which contain the names of the
connected particles or jets

For example
` [(PARTICLE_1, PARTICLE_2), (PARTICLE_3, PARTICLE_4)] ` will define an
event symmetric group where the first two particles are interchangeable, 
and the last two particles are interchangeable.

Any groupings of three or more particles will mean that **ALL** of the
particles are symmetric with each other. Any elements not present in the 
permtuations description will be assumed to be invariant with only iteself.

For example, using the same four particles as above
` [(PARTICLE_1, PARTICLE_2, PARTICLE_3)] `
defines a group with the first three particles are completely invariant
with respect to each other but the final particle `PARTICLE_4` is 
invariant with nothing.

## Example
Refer to the Event File section of the [`ttbar` Example Guide](TTBar.md) for a
description of the `ttbar.ini` example event file.