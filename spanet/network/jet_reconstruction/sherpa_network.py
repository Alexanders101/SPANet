from typing import Optional
from sherpa import Client, Trial

from spanet.options import Options
from spanet.network.jet_reconstruction.jet_reconstruction_training import JetReconstructionTraining
from spanet.network.jet_reconstruction.jet_reconstruction_validation import JetReconstructionValidation


class SherpaNetwork(JetReconstructionTraining, JetReconstructionValidation):
    def __init__(self, options: Options, client: Client, trial: Trial):
        super(SherpaNetwork, self).__init__(options)

        self.trial = trial
        self.client = client
        self.sherpa_iteration = 0

    def commit_sherpa(self, objective, context: Optional[dict] = None):
        if context is None:
            context = {}

        if self.client:
            self.sherpa_iteration += 1
            self.client.send_metrics(trial=self.trial,
                                     iteration=self.sherpa_iteration,
                                     objective=objective.item(),
                                     context={key: val.item() for key, val in context.items()})

    def validation_epoch_end(self, outputs):
        # Optionally use this accuracy score for something like hyperparameter search
        validation_accuracy = sum(x['validation_accuracy'] for x in outputs) / len(outputs)
        self.commit_sherpa(validation_accuracy)
