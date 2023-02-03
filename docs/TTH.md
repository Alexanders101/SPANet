# Semi-Leptonic `ttH` Example Guide

Semi-leptonic `ttH` presents an opportunity to display a more complicated event topology, with multiple types of inputs and outputs. This is guide builds upon the `ttbar` example by displaying some more complex features.

## Event `.yaml` File

The yaml file describing the event may be found at [`event_files/semi_leptonic_ttH.yaml`](../event_files/semi_leptonic_ttH.yaml). We recreate it here.

```yaml
INPUTS:
  SEQUENTIAL:
    Momenta:
      mass: log_normalize
      pt: log_normalize
      eta: normalize

      sin_phi: normalize
      cos_phi: normalize

      qtag: none
      btag: none
      etag: none
      utag: none

  GLOBAL:
    Met:
      met: log_normalize
      sin_phi: normalize
      cos_phi: normalize


EVENT:
  lt:
    - b

  ht:
    - b
    - q1
    - q2
  h:
    - b1
    - b2

PERMUTATIONS:
    ht:
      - [ q1, q2 ]
    h:
      - [ b1, b2 ]

REGRESSIONS:
  EVENT:
    - neutrino_eta
    - neutrino_px
    - neutrino_py
    - neutrino_pz
    - log_invariant_mass

CLASSIFICATIONS:
  EVENT:
    - signal
```

### Global MET
Notice that we now have several additional sections when compared to `ttbar`. We define a new input which will keep track of the missing neutrino energy. This behaves similarly to our sequential inputs, except that there is exactly one value present for every event. This presents a way of adding extra event-level information that is not assigned to any particular jet or lepton. 

```yaml
GLOBAL:
    Met:
      met: log_normalize
      sin_phi: normalize
      cos_phi: normalize
```

### Neutrino Regression Reconstruction
We additionally want to estimate some of the lost information about the neutrino which the detector was not able to capture. We extract these values from the simulator and store them. SPANet is able to include additional real-valued regression outputs to estimate these measurements.

```yaml
REGRESSIONS:
  EVENT:
    - neutrino_eta
    - neutrino_px
    - neutrino_py
    - neutrino_pz
    - log_invariant_mass
```

### Signal-Background Separation
We additionally include some 4 bjet `ttbar` events as background to the `ttH` signal. We may train SPANet to distinguish between the signal and background events by including an additional classification output. Each of these outputs is a multi-class classification output, although in this case we only have two classes.
```yaml
CLASSIFICATIONS:
  EVENT:
    - signal
```

## Combined Training
If you specify multiple outputs for SPANet, then all of the outputs will be trained simultaneously. You can control the strength of each target using the following hyperparameters.
```json
// From `options_files/semi_leptonic_ttH/example.json`
"assignment_loss_scale": 1.0,
"detection_loss_scale": 1.0,
"kl_loss_scale": 0.0,
"regression_loss_scale": 1.0,
"classification_loss_scale": 1.0,
```

These will control the absolute weight assigned to every loss term in the total loss function. Note that masking out one of the losses for an event will still allow other objectives to be trained. For example, in our `ttH` example, notice that all of the background events lack a reconstruction target. Therefore, the reconstruction heads will ignore background events and the background will only be used to train the regression and classification outputs. Although note that we will still ignore these events and losses if `"partial_events": false` since this depends only on the reconstruction targets.

## Dataset
We mirror the structure defined in the event info file in the dataset. We include a small example datasets with the correct strcuture to assist in making your own complex events. The HDF5 structure for this dataset is copied below.

Notice a couple of details:
- `Met` inputs are one-dimensional compared to `Momenta` because we defined the MET to be a global variable in the event info. Note that we also don't need to include a `MASK` for the event variables.
- `CLASSIFICATIONS` inputs are one-dimensional `int64` arrays which will define the class of every event. The total number of classes is inferred from this array. A value of `-1` may be used to indicate no class for a given event, although we do not have such instances in the example.
- `REGRESSIONS` inputs are simple one-dimensional `float32` arrays storing the regression target for every event. A value of float `NaN` may be used to indicate no value for the event, although we do not have such instances in the example.

```
============================================================
| Structure for data/semi_leptonic_ttH/example.h5 
============================================================

|-CLASSIFICATIONS               
|---EVENT                       
|-----signal                     :: int64    : (1000,)
|-INPUTS                        
|---Met
|-----met                        :: float32  : (1000,)
|-----sumet                      :: float32  : (1000,)
|-----cos_phi                    :: float32  : (1000,)
|-----sin_phi                    :: float32  : (1000,)
|---Momenta
|-----MASK                       :: bool     : (1000, 19)
|-----cos_phi                    :: float32  : (1000, 19)
|-----sin_phi                    :: float32  : (1000, 19)
|-----eta                        :: float32  : (1000, 19)
|-----mass                       :: float32  : (1000, 19)
|-----pt                         :: float32  : (1000, 19)
|-----btag                       :: float32  : (1000, 19)
|-----qtag                       :: float32  : (1000, 19)
|-----etag                       :: float32  : (1000, 19)
|-----utag                       :: float32  : (1000, 19)
|-REGRESSIONS                   
|---EVENT                       
|-----log_invariant_mass         :: float32  : (1000,)
|-----neutrino_eta               :: float32  : (1000,)
|-----neutrino_px                :: float32  : (1000,)
|-----neutrino_py                :: float32  : (1000,)
|-----neutrino_pz                :: float32  : (1000,)
|-TARGETS                       
|---h                           
|-----b1                         :: int64    : (1000,)
|-----b2                         :: int64    : (1000,)
|---ht                          
|-----b                          :: int64    : (1000,)
|-----q1                         :: int64    : (1000,)
|-----q2                         :: int64    : (1000,)
|---lt                          
|-----b                          :: int64    : (1000,)
|-----l                          :: int64    : (1000,)
```