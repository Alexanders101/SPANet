from functools import reduce
from itertools import permutations, product
import warnings

import numpy as np

from spanet.dataset.event_info import EventInfo


class SymmetricEvaluator:
    def __init__(self, event_info: EventInfo):
        self.event_info = event_info
        self.event_group = event_info.event_symbolic_group
        self.target_groups = event_info.product_symbolic_groups

        # Gather all of the Similar particles together based on the permutation groups
        clusters = []
        cluster_groups = []

        for orbit in self.event_group.orbits():
            orbit = tuple(sorted(orbit))
            names = [event_info.event_particles[i] for i in orbit]

            cluster_name = map(dict.fromkeys, names)
            cluster_name = map(lambda x: x.keys(), cluster_name)
            cluster_name = ''.join(reduce(lambda x, y: x & y, cluster_name))
            clusters.append((cluster_name, names, orbit))

            cluster_group = self.target_groups[names[0]]
            for name in names:
                assert (
                    self.target_groups[name] == cluster_group,
                    "Invalid Symmetry Group. Invariant targets have different structures."
                )

            cluster_groups.append((cluster_name, names, cluster_group))

        self.clusters = clusters
        self.cluster_groups = cluster_groups

    @staticmethod
    def permute_arrays(array_list, permutation):
        return [array_list[index] for index in permutation]

    def sort_outputs(self, predictions, target_jets, target_masks):
        predictions = [np.copy(p) for p in predictions]
        target_jets = [np.copy(p) for p in target_jets]

        # Sort all of the targets and predictions to avoid any intra-particle symmetries
        for i, (_, particle_group) in enumerate(self.target_groups.items()):
            for orbit in particle_group.orbits():
                orbit = tuple(sorted(orbit))

                target_jets[i][:, orbit] = np.sort(target_jets[i][:, orbit], axis=1)
                predictions[i][:, orbit] = np.sort(predictions[i][:, orbit], axis=1)

        return predictions, target_jets, target_masks

    def particle_count_info(self, target_masks):
        target_masks = np.array(target_masks)

        # Count the total number of particles for simple filtering
        total_particle_counts = target_masks.sum(0)

        # Count the number of particles present in each cluster
        particle_counts = [target_masks[list(cluster_indices)].sum(0)
                           for _, _, cluster_indices in self.clusters]

        # Find the maximum number of particles in each cluster
        particle_max = [len(cluster_indices) for _, _, cluster_indices in self.clusters]

        return total_particle_counts, particle_counts, particle_max

    def cluster_purity(self, predictions, target_jets, target_masks):
        results = []

        for cluster_name, cluster_particles, cluster_indices in self.clusters:
            # Extract jet information for the current cluster
            cluster_target_masks = np.stack([target_masks[i] for i in cluster_indices])
            cluster_target_jets = np.stack([target_jets[i] for i in cluster_indices])
            cluster_predictions = np.stack([predictions[i] for i in cluster_indices])

            # Keep track of the best accuracy achieved for each event
            best_accuracy = np.zeros(cluster_target_masks.shape[1], dtype=np.int64)

            for target_permutation in permutations(range(len(cluster_indices))):
                target_permutation = list(target_permutation)

                accuracy = cluster_predictions == cluster_target_jets[target_permutation]
                accuracy = accuracy.all(-1) * cluster_target_masks[target_permutation]
                accuracy = accuracy.sum(0)

                best_accuracy = np.maximum(accuracy, best_accuracy)

            # Get rid of pesky warnings
            total_particles = cluster_target_masks.sum()
            if total_particles > 0:
                cluster_accuracy = best_accuracy.sum() / cluster_target_masks.sum()
            else:
                cluster_accuracy = np.nan

            results.append((cluster_name, cluster_particles, cluster_accuracy))

        return results

    def event_purity(self, predictions, target_jets, target_masks):
        target_masks = np.stack(target_masks)

        # Keep track of the best accuracy achieved for each event
        best_accuracy = np.zeros(target_masks.shape[1], dtype=np.int64)

        for target_permutation in self.event_info.event_permutation_group:
            permuted_targets = self.permute_arrays(target_jets, target_permutation)
            permuted_mask = self.permute_arrays(target_masks, target_permutation)
            accuracy = np.array([(p == t).all(-1) * m
                                 for p, t, m
                                 in zip(predictions, permuted_targets, permuted_mask)])
            accuracy = accuracy.sum(0)

            best_accuracy = np.maximum(accuracy, best_accuracy)

        # Event accuracy is defined as getting all possible particles in event
        num_particles_in_event = target_masks.sum(0)
        accurate_event = best_accuracy == num_particles_in_event

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return accurate_event.mean()

    def full_report(self, predictions, target_jets, target_masks):
        predictions, target_jets, target_masks = self.sort_outputs(predictions, target_jets, target_masks)

        total_particle_counts, particle_counts, particle_max = self.particle_count_info(target_masks)
        particle_ranges = [list(range(-1, pmax + 1)) for pmax in particle_max]

        full_results = []

        for event_counts in product(*particle_ranges):
            # Filter all events to make sure they at least have a particle there
            event_mask = total_particle_counts >= 0

            # Filter to have the correct cluster counts
            for particle_count, event_count in zip(particle_counts, event_counts):
                if event_count >= 0:
                    event_mask = event_mask & (particle_count == event_count)

                # During wildcard events, make sure we have at least one particle in the event.
                if event_count < 0:
                    event_mask = event_mask & (total_particle_counts > 0)

            # Filter event information according to computed mask
            masked_predictions = [p[event_mask] for p in predictions]
            masked_target_jets = [p[event_mask] for p in target_jets]
            masked_target_masks = [p[event_mask] for p in target_masks]

            # Compute purity values
            masked_event_purity = self.event_purity(masked_predictions, masked_target_jets, masked_target_masks)
            masked_cluster_purity = self.cluster_purity(masked_predictions, masked_target_jets, masked_target_masks)

            mask_proportion = event_mask.mean()

            full_results.append((event_counts, mask_proportion, masked_event_purity, masked_cluster_purity))

        return full_results

    def full_report_string(self, predictions, target_jets, target_masks, prefix: str = ""):
        full_purities = {}

        report = self.full_report(predictions, target_jets, target_masks)
        for event_mask, mask_proportion, event_purity, particle_purity in report:

            event_mask_name = ""
            purity = {
                "{}{}/event_purity": event_purity,
                "{}{}/event_proportion": mask_proportion
            }

            for mask_count, (cluster_name, _, cluster_purity) in zip(event_mask, particle_purity):
                mask_count = "*" if mask_count < 0 else str(mask_count)
                event_mask_name = event_mask_name + mask_count + cluster_name
                purity["{}{}/{}_purity".format("{}", "{}", cluster_name)] = cluster_purity

            purity = {
                key.format(prefix, event_mask_name): val for key, val in purity.items()
            }

            full_purities.update(purity)

        return full_purities
