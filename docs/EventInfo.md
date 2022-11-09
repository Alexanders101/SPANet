# Event Specification Format

The first step to training SPANets is to define the topology of your target event. To do this, `SPANet`uses a definition `.yaml` file which contains the features and jet information for your event. This will describe both the inputs and outputs for your model, along with any related symmetries.

The structure of the `.yaml` file will follows a standard format. Special keys which must be exactly as shown will be in `CAPITALCASE`. Custom keys which may modified for your event will be in `lower_case_with_underscores`
```yaml
INPUTS:
    SEQUENTIAL:
        sequential_input_1:
            feature_1: feature_option_1
            feature_2: feature_option_2
            feature_3: feature_option_3
            ...
            
        sequential_input_2:
            feature_1: feature_option_1
            feature_2: feature_option_2
            feature_3: feature_option_3
            ...
        ...
        
    GLOBAL:
        global_input_1:
            feature_1: feature_option_1
            feature_2: feature_option_2
            feature_3: feature_option_3
            ...
        ...
        
EVENT:
    event_particle_1:
        - decay_product_1: sequential_input_1
        - decay_product_2: sequential_input_1
        ...
        
    event_particle_2:
        - decay_product_1: sequential_input_1
        - decay_product_2: sequential_input_1
        ...
    
    ...

PERMUTATIONS:
    EVENT:
        - [ event_particle, event_particle ]
        ...
        
    event_particle_1:
        - [ decay_product, decay_product ]
        ...
        
    event_particle_2:
        - [ decay_product, decay_product ]
        ...
    ...
   
REGRESSIONS:
    ... (WIP Explained Later)

CLASSIFICATIONS:
    ... (WIP Explained Later)
```

We will now go over each of the sections to explain what must be included. 
You may also view an example of a complete event file
in [`ttbar.yaml`](../event_files/full_hadronic_ttbar.yaml).

## `INPUTS`
The first **reuquired** section. Inputs will contain a description of the features that will be fed in as input to SPANet. Each input should have a unique name. They will later be used in how you define your dataset. The exact names are not important
as long as they are unique.

There are two types of inputs:
- **SEQUENTIAL** inputs represent variable length inputs for each event.
  These may include objects such as hadronic jets, leptons, neutrinos, etc. We require that the event contains at least one sequential input. These inputs will also define the potential reconstruction targets.
- **GLOBAL** inputs represent features which exist for the entire event.
  There exists only a single instance of the inputs for every event.
  Examples include neutrino missing energy.

Each input contains one or more *features*. These are the observable values associated with each input. Each feature is also given a unique name which will be used when creating the dataset. Each feature can have association several options which define how `SPANet` will pre-process the feature. Valid options include:

| FEATURE_OPTION     | Description |
| :--------------:   | ----------- |
| `none`             | No pre-processing applied            |
| `log`              | Scale the feature on a `log` scale | 
| `normalize`        | Normalize the feature based on training dataset statistics |
| `log_normalize`    | First apply a `log` scale and then normalize |

## `EVENT`
The second **required** section. This will contain a simplified Feynman diagram of your event. We require that events processed by SPANet follow a particular two-level structure. We split the event into
1. **Event Particles:** The first level of the Feynmen Diagram. These are typically non-observable particles which we are interested in studying. These particles are required to decay into other particles.
2. **Decay Products:** The second level will contain observable decay products. These are required to be particles which will have reconstruction targets associated with them. Decay products may correspond to a specific sequential input. You can define the correspondence by adding the name of the sequential input after the decay product `decay_product: sequential_input`.

We describe this Feynman diagram structure with a simple two layer tree. Give each event particle a unique name. Decay particles may repeat names as long as they belong to different event particles.



## `PERMUTATIONS`
Describe the symmetries allowed in during assignment. You may specify an event-level symmetry group over event particles with the special keyword.
```yaml
EVENT:
    - [ event_particle, event_particle ]
    ...
```
Decay product symmetry groups may be specicied with their associated event particle name.
```yaml
event_particle_1:
    - [ decay_product, decay_product ]
    ...
```

SPANet supports describing permutation groups as products of complete symmetry goups G = S_1 x S_2 x ... or using an explicit collection of generating cycles.

### Complete Symmetric Groups
In order to define symmetric permutation groups, you simply describe which particles or jets belong to each of the fully symmetric groups. This is expressed as a list of lists each of which contain the names of the connected particles or jets

For example
```yaml
EVENT:
    - [ event_particle_1, event_particle_2 ]
    - [ event_particle_3, event_particle_4 ]
```
will define an event symmetric group where the first two particles are interchangeable, and the last two particles are interchangeable.

Any groupings of three or more particles will mean that **ALL** of the particles are symmetric with each other. Any elements not present in the permutations description will be assumed to be invariant with only itself.

For example, using the same four particles as above ` [event_particle_1, event_particle_2, event_particle_3] ` defines a group with the first three particles are completely invariant with respect to each other but the final particle `event_particle_4` is invariant with nothing.

### Explicit Permutation groups
You can also define custom permutation groups using explicit cycles. These cycles will be used as the generators for a general permutation group. Cycles are defined using nested lists instead of the simple lists above. Each list of lists defines a disjoint cycle so `[[1, 2, 3], [4, 5]] = (1,2,3)(4,5)`. See the following article for more information on the disjoint cycle notation for permutations. [https://groupprops.subwiki.org/wiki/Cycle_decomposition_for_permutations](https://groupprops.subwiki.org/wiki/Cycle_decomposition_for_permutations)

For example
```yaml
EVENT:
    - [ [p1, p2], [p3, p4] ]
```
Will define an event permutation group where `p1` may only be swapped with `p2` if `p3` is simultaneously swapped with `p4`. This defines a different group than 
```yaml
EVENT:
    - [[p1, p2]]
    - [[p3, p4]]
```
Where the two pairs may be swapped independently of each other. 

## `REGRESSIONS` & `CLASSIFICATIONS`
These sections define the additional regression and classification targets associated to any part of the event tree. These are optional and may be left out if you do not wish to train a model to perform these tasks. These two sections have identical structure. We will use `REGRESSIONS` for these descriptions. 

```yaml
REGRESSIONS:
    EVENT: 
        - event_regression_1
        - event_regression_2
    event_particle_1:
        - PARTICLE:
            - event_particle_1_regression_1
            - event_particle_1_regression_2
        - decay_product_1:
            - decay_product_1_regression_1
            - decay_product_1_regression_2
        - decay_product_2:
            - decay_product_2_regression_1
            - decay_product_2_regression_2
        ...
    ...
```

You may add additional regression targets to any point in the Feynman diagram. The `EVENT` targets will be defined for the entire event and can include things such as system invariant mass. You can also add regression targets to individual event particles with the special `PARTICLE` tag. Finally, you can add regression targets to each individual decay products within each event particle. Any symmetric event particles or decay products must define the same regression targets.

**This feature is currently considered a work in progress and may not be fully compatible with the symmetric structures defined above. Only the `EVENT` regression and classifications are currently fully supported.**

## Example
Refer to the Event File section of the [`ttbar` Example Guide](TTBar.md) for a description of the `ttbar.yaml` example event file.