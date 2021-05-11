import numpy as np

import numba
from numba import njit
from numba.extending import overload

from warnings import filterwarnings
filterwarnings("ignore", category=numba.NumbaPendingDeprecationWarning)


# Create a dummy function to overload for numba
def mask(data, index, value):
    pass


@njit("void(float32[::1], int64, float32)")
def mask_1(data, index, value):
    data[index] = value


@njit("void(float32[:, ::1], int64, float32)")
def mask_2(data, index, value):
    data[index, :] = value
    data[:, index] = value


@njit("void(float32[:, :, ::1], int64, float32)")
def mask_3(data, index, value):
    data[index, :, :] = value
    data[:, index, :] = value
    data[:, :, index] = value


@njit("void(float32[:, :, :, ::1], int64, float32)")
def mask_4(data, index, value):
    data[index, :, :, :] = value
    data[:, index, :, :] = value
    data[:, :, index, :] = value
    data[:, :, :, index] = value


@njit("void(float32[:, :, :, :, ::1], int64, float32)")
def mask_5(data, index, value):
    data[index, :, :, :, :] = value
    data[:, index, :, :, :] = value
    data[:, :, index, :, :] = value
    data[:, :, :, index, :] = value
    data[:, :, :, :, index] = value


@njit("void(float32[:, :, :, :, :, ::1], int64, float32)")
def mask_6(data, index, value):
    data[index, :, :, :, :, :] = value
    data[:, index, :, :, :, :] = value
    data[:, :, index, :, :, :] = value
    data[:, :, :, index, :, :] = value
    data[:, :, :, :, index, :] = value
    data[:, :, :, :, :, index] = value


@njit("void(float32[:, :, :, :, :, :, ::1], int64, float32)")
def mask_7(data, index, value):
    data[index, :, :, :, :, :, :] = value
    data[:, index, :, :, :, :, :] = value
    data[:, :, index, :, :, :, :] = value
    data[:, :, :, index, :, :, :] = value
    data[:, :, :, :, index, :, :] = value
    data[:, :, :, :, :, index, :] = value
    data[:, :, :, :, :, :, index] = value


@overload(mask)
def mask_impl(data, index, value):
    if data.ndim == 1:
        def impl(data, index, value):
            return mask_1(data, index, value)

        return impl
    if data.ndim == 2:
        def impl(data, index, value):
            return mask_2(data, index, value)

        return impl
    if data.ndim == 3:
        def impl(data, index, value):
            return mask_3(data, index, value)

        return impl
    if data.ndim == 4:
        def impl(data, index, value):
            return mask_4(data, index, value)

        return impl
    if data.ndim == 5:
        def impl(data, index, value):
            return mask_5(data, index, value)

        return impl
    if data.ndim == 6:
        def impl(data, index, value):
            return mask_6(data, index, value)

        return impl
    if data.ndim == 7:
        def impl(data, index, value):
            return mask_7(data, index, value)

        return impl


@njit
def unravel_index(index, shape):
    sizes = np.zeros(len(shape), dtype=np.int64)
    result = np.zeros(len(shape), dtype=np.int64)
    sizes[-1] = 1
    for i in range(len(shape) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * shape[i + 1]
    remainder = index
    for i in range(len(shape)):
        result[i] = remainder // sizes[i]
        remainder %= sizes[i]
    return result


@njit
def extract_prediction(input_predictions, max_dimensions=0):
    float_inf = np.float32(np.inf)
    num_targets = len(input_predictions)

    # Create copies of predictions for safety and calculate the output shapes
    predictions = []
    shapes = []

    for prediction in input_predictions:
        prediction = prediction.copy()
        predictions.append(prediction)
        shapes.append(prediction.shape)

        if prediction.ndim > max_dimensions:
            max_dimensions = prediction.ndim

    # Fill up the prediction matrix
    # -2 : Not yet assigned
    # -1 : Masked value
    # else : The actual index value
    results = np.zeros((num_targets, max_dimensions), np.int64) - 2

    while np.any(results < -1):
        best_index = np.argmax(np.array([np.max(pp) for pp in predictions]))
        best_jets = unravel_index(np.argmax(predictions[best_index].ravel()), shapes[best_index])

        results[best_index] = -1
        for i in range(predictions[best_index].ndim):
            results[best_index, i] = best_jets[i]

        predictions[best_index] -= float_inf
        for prediction in predictions:
            for jet in best_jets:
                mask(prediction, jet, -float_inf)

    return results


@numba.njit(parallel=True)
def extract_predictions(predictions):
    max_dimensions = 0
    batch_size = 0
    num_targets = len(predictions)

    for p in predictions:
        current_batch_size = p.shape[0]
        current_dimensions = p.ndim - 1

        if current_batch_size > batch_size:
            batch_size = current_batch_size

        if current_dimensions > max_dimensions:
            max_dimensions = current_dimensions

    output = np.zeros((batch_size, num_targets, max_dimensions), np.int64)
    for batch in numba.prange(batch_size):
        current_prediction = [p[batch] for p in predictions]
        output[batch, :, :] = extract_prediction(current_prediction, max_dimensions)

    return output.transpose(1, 0, 2)
