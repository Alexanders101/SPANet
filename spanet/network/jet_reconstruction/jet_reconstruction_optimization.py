from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.autograd.functional import _construct_standard_basis_for, _autograd_grad, _grad_postprocess, \
    _tuple_postprocess, _as_tuple, _check_requires_grad

from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.options import Options


class JetReconstructionOptimization(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionOptimization, self).__init__(options, torch_script)

        self.num_losses = (
            (self.options.assignment_loss_scale > 0) * len(self.training_dataset.assignments) +
            (self.options.detection_loss_scale > 0) * len(self.training_dataset.assignments) +
            (self.options.regression_loss_scale > 0) * len(self.training_dataset.regressions) +
            (self.options.classification_loss_scale > 0) * len(self.training_dataset.classifications) +
            (self.options.kl_loss_scale > 0)
        )

        self.loss_weight_logits = torch.nn.Parameter(torch.zeros(self.num_losses), requires_grad=True)
        self.loss_weight_alpha = 0.0

    @staticmethod
    def jacobian_script(outputs: Tensor, inputs: Tuple[Tensor], create_graph: bool = False):
        is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jacobian")
        _check_requires_grad(outputs, "outputs", strict=False)

        output_numels = tuple(output.numel() for output in outputs)
        grad_outputs = _construct_standard_basis_for(outputs, output_numels)
        flat_outputs = tuple(output.reshape(-1) for output in outputs)

        def vjp(grad_output):
            vj = list(
                _autograd_grad(flat_outputs, inputs, grad_output, create_graph=create_graph, is_grads_batched=True))
            for el_idx, vj_el in enumerate(vj):
                if vj_el is not None:
                    continue
                vj[el_idx] = torch.zeros_like(inputs[el_idx]).expand((sum(output_numels),) + inputs[el_idx].shape)
            return tuple(vj)

        jacobians_of_flat_output = vjp(grad_outputs)

        jacobian_input_output = []
        for jac, input_i in zip(jacobians_of_flat_output, inputs):
            jacobian_input_i_output = []
            for jac, output_j in zip(jac.split(output_numels, dim=0), outputs):
                jacobian_input_i_output_j = jac.view(output_j.shape + input_i.shape)
                jacobian_input_i_output.append(jacobian_input_i_output_j)
            jacobian_input_output.append(jacobian_input_i_output)

        jacobian_output_input = tuple(zip(*jacobian_input_output))
        jacobian_output_input = _grad_postprocess(jacobian_output_input, create_graph)
        return _tuple_postprocess(jacobian_output_input, (is_outputs_tuple, True))

    @staticmethod
    def jacobian_loop(outputs: Tensor, inputs: Tuple[Tensor], create_graph: bool = False, strict: bool = False):
        is_outputs_tuple, outputs = _as_tuple(outputs, "outputs of the user-provided function", "jacobian")
        _check_requires_grad(outputs, "outputs", strict=strict)

        jacobian: Tuple[torch.Tensor, ...] = tuple()
        for i, out in enumerate(outputs):
            jac_i: Tuple[List[torch.Tensor]] = tuple([] for _ in range(len(inputs)))

            for j in range(out.nelement()):
                vj = _autograd_grad((out.reshape(-1)[j],), inputs, retain_graph=True, create_graph=create_graph)

                for el_idx, (jac_i_el, vj_el, inp_el) in enumerate(zip(jac_i, vj, inputs)):
                    if vj_el is not None:
                        if strict and create_graph and not vj_el.requires_grad:
                            msg = ("The jacobian of the user-provided function is "
                                   "independent of input {}. This is not allowed in "
                                   "strict mode when create_graph=True.".format(i))
                            raise RuntimeError(msg)
                        jac_i_el.append(vj_el)
                    else:
                        if strict:
                            msg = ("Output {} of the user-provided function is "
                                   "independent of input {}. This is not allowed in "
                                   "strict mode.".format(i, el_idx))
                            raise RuntimeError(msg)
                        jac_i_el.append(torch.zeros_like(inp_el))

            jacobian += (tuple(torch.stack(jac_i_el, dim=0).view(out.size()
                                                                 + inputs[el_idx].size()) for (el_idx, jac_i_el) in
                               enumerate(jac_i)),)

        jacobian = _grad_postprocess(jacobian, create_graph)

        return _tuple_postprocess(jacobian, (is_outputs_tuple, True))

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=self.options.balance_losses)

    def backward(self, loss: Tensor, optimizer, optimizer_idx, *args, **kwargs) -> None:
        if not self.options.balance_losses:
            loss.sum().backward(*args, **kwargs)
            return

        loss_weights = torch.softmax(self.loss_weight_logits, 0)
        free_weights = loss_weights.detach()

        parameters = tuple(p for p in self.parameters() if p.requires_grad)
        jacobians = self.jacobian_loop(loss, parameters)
        GW = []

        for parameter, jacobian in zip(parameters, jacobians):
            weights = free_weights.reshape((-1, ) + (1,) * (jacobian.ndim - 1))
            parameter.grad = (weights * jacobian).sum(0)
            GW.append(jacobian.view(self.num_losses, -1))

        GW = torch.cat(GW, -1)
        GW = loss_weights.unsqueeze(-1) * GW
        GW = torch.sqrt(GW.square().sum(-1))

        GW_bar = GW.detach().mean()

        r = (loss.detach() / loss.detach().mean()) ** self.loss_weight_alpha
        L_grad = torch.abs((GW - GW_bar * r)).sum()
        self.loss_weight_logits.grad, = torch.autograd.grad(L_grad, self.loss_weight_logits)

        for i, a in enumerate(free_weights):
            self.log(f"weights/{i}", a)
