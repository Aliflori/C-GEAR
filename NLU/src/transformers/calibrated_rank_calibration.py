# coding=utf-8
"""Deterministic training-only virtual calibration for dynamic LoRA ranks.

This module deliberately has no dependency on :class:`Trainer` or the rank
allocator.  The calibrated allocator supplies already-collated *training*
batch pairs and a loss callback; this module only performs a reversible,
stateless AdamW-like virtual update of the proposed rank components.
"""

import copy
import hashlib
import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from .optimization import AdamW


@dataclass
class RNGState:
    """Complete process RNG state used by calibration forwards."""

    python_state: object
    numpy_state: tuple
    torch_cpu_state: torch.Tensor
    torch_cuda_states: object


def capture_rng_state():
    """Capture Python, NumPy, CPU Torch, and initialized CUDA RNG states."""

    cuda_states = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return RNGState(
        python_state=copy.deepcopy(random.getstate()),
        numpy_state=copy.deepcopy(np.random.get_state()),
        torch_cpu_state=torch.get_rng_state().clone(),
        torch_cuda_states=cuda_states,
    )


def restore_rng_state(state):
    """Restore a state produced by :func:`capture_rng_state`."""

    if not isinstance(state, RNGState):
        raise TypeError("state must be an RNGState instance.")
    random.setstate(copy.deepcopy(state.python_state))
    np.random.set_state(copy.deepcopy(state.numpy_state))
    torch.set_rng_state(state.torch_cpu_state.clone())
    if state.torch_cuda_states is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore CUDA RNG state because CUDA is unavailable.")
        torch.cuda.set_rng_state_all(
            [cuda_state.clone() for cuda_state in state.torch_cuda_states]
        )


def _clone_value(value):
    """Clone nested model inputs so loss callbacks may mutate their argument."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return copy.deepcopy(value)


def _clone_score(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


def _copy_score(module, value):
    current = getattr(module, "score", None)
    if isinstance(current, torch.Tensor) and isinstance(value, torch.Tensor):
        if current.shape == value.shape and current.dtype == value.dtype:
            with torch.no_grad():
                current.copy_(value)
            return
    setattr(module, "score", _clone_score(value))


class VirtualCalibrationSnapshot:
    """Exact restorable state around virtual candidate evaluation.

    Only candidate parameter *data* can be changed by the stateless virtual
    update, so copying the full frozen backbone is intentionally avoided.  All
    gradients and trainability flags are nevertheless captured because the
    real training gradient already exists at the allocation hook.
    """

    def __init__(self, model, optimizer, scheduler, affected_parameters):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        unique_affected = []
        seen = set()
        for parameter in affected_parameters:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                unique_affected.append(parameter)
        self.parameter_data = [
            (parameter, parameter.detach().clone())
            for parameter in unique_affected
        ]
        self.gradients = [
            (
                parameter,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for parameter in model.parameters()
        ]
        self.requires_grad = [
            (parameter, bool(parameter.requires_grad))
            for parameter in model.parameters()
        ]
        self.buffers = [
            (buffer, buffer.detach().clone()) for buffer in model.buffers()
        ]
        self.training_modes = [
            (module, bool(module.training)) for module in model.modules()
        ]
        self.scores = [
            (module, _clone_score(module.score))
            for module in model.modules()
            if hasattr(module, "score")
        ]
        self.optimizer_state = (
            copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
        )
        self.scheduler_state = (
            copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
        )
        self.rng_state = capture_rng_state()

    def restore(self):
        """Restore all captured state without invoking model.train()/eval()."""

        with torch.no_grad():
            for parameter, value in self.parameter_data:
                parameter.copy_(value)
            for buffer, value in self.buffers:
                buffer.copy_(value)

        for parameter, requires_grad in self.requires_grad:
            parameter.requires_grad_(requires_grad)
        for parameter, gradient in self.gradients:
            if gradient is None:
                parameter.grad = None
            else:
                parameter.grad = gradient.detach().clone()

        # Direct assignment avoids SVDLinear.train/eval merge side effects.
        for module, training in self.training_modes:
            module.training = training
        for module, score in self.scores:
            _copy_score(module, score)

        if self.optimizer is not None and self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)
        if self.scheduler is not None and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
        restore_rng_state(self.rng_state)


def _parameter_chunks(parameters, rank_dimension):
    cursor = 0
    chunks = []
    for parameter in parameters:
        width = (
            int(parameter.numel())
            if rank_dimension is None
            else int(parameter.size(rank_dimension))
        )
        chunks.append((cursor, cursor + width, parameter))
        cursor += width
    return chunks, cursor


def _parameters_in_rank_interval(parameters, rank_dimension, start, stop, label):
    chunks, capacity = _parameter_chunks(parameters, rank_dimension)
    if start < 0 or stop <= start or stop > capacity:
        raise ValueError(
            "Candidate rank interval [{}, {}) is invalid for {} capacity {}.".format(
                start, stop, label, capacity
            )
        )
    selected = []
    covered = start
    for chunk_start, chunk_stop, parameter in chunks:
        overlaps = chunk_start < stop and chunk_stop > start
        if not overlaps:
            continue
        if chunk_start < start or chunk_stop > stop:
            raise ValueError(
                "Candidate rank interval cuts through parameter chunk {} [{}, {}).".format(
                    label, chunk_start, chunk_stop
                )
            )
        if chunk_start != covered:
            raise ValueError("Candidate rank components are not contiguous for {}.".format(label))
        selected.append(parameter)
        covered = chunk_stop
    if covered != stop:
        raise ValueError("Candidate rank interval is incomplete for {}.".format(label))
    return selected


def get_candidate_rank_parameters(module, rank_increment):
    """Return the next whole A/E/B parameter chunks for one dynamic module."""

    increment = int(rank_increment)
    if increment <= 0:
        raise ValueError("rank_increment must be positive.")
    required = ("lora_A", "lora_E", "lora_B", "ranknum")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise TypeError("Candidate module lacks dynamic LoRA fields: {}.".format(missing))

    rank_value = float(module.ranknum.detach().item())
    active_rank = int(round(rank_value))
    if not math.isfinite(rank_value) or abs(rank_value - active_rank) > 1e-6:
        raise ValueError("Candidate module has a non-integral active rank.")
    stop = active_rank + increment
    a_parameters = _parameters_in_rank_interval(
        module.lora_A, 0, active_rank, stop, "lora_A"
    )
    e_parameters = _parameters_in_rank_interval(
        module.lora_E, None, active_rank, stop, "lora_E"
    )
    b_parameters = _parameters_in_rank_interval(
        module.lora_B, 1, active_rank, stop, "lora_B"
    )
    return {
        "active_rank": active_rank,
        "new_active_rank": stop,
        "lora_A": a_parameters,
        "lora_E": e_parameters,
        "lora_B": b_parameters,
        "parameters": a_parameters + e_parameters + b_parameters,
        "ranknum": module.ranknum,
    }


def _activate_candidate(candidate_specs):
    with torch.no_grad():
        for module, spec in candidate_specs:
            module.ranknum.fill_(float(spec["new_active_rank"]))
            for parameter in spec["parameters"]:
                parameter.requires_grad_(True)


def _optimizer_group_for_parameter(optimizer, parameter):
    for group in optimizer.param_groups:
        if any(group_parameter is parameter for group_parameter in group["params"]):
            return group
    raise ValueError(
        "Every proposed rank parameter must already belong to the optimizer; "
        "the calibrated integration must register reserves before scoring."
    )


def _state_step_as_int(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("AdamW optimizer step must be scalar.")
        value = value.detach().item()
    return int(value)


def stateless_adamw_update(parameters, optimizer, update_scale=1.0):
    """Apply a virtual local-AdamW update without mutating optimizer moments.

    The arithmetic follows ``transformers.optimization.AdamW.step`` in this
    repository: moments are advanced, bias correction is applied to the step
    size, the Adam update is applied, and decoupled weight decay is applied to
    the resulting parameter using the actual parameter-group learning rate.
    """

    if optimizer is None:
        raise ValueError("An optimizer is required for virtual AdamW updates.")
    if not isinstance(optimizer, AdamW):
        raise TypeError(
            "Calibrated virtual updates require the repository's "
            "transformers.optimization.AdamW optimizer."
        )
    update_scale = float(update_scale)
    if not math.isfinite(update_scale) or not 0.0 < update_scale <= 1.0:
        raise ValueError("update_scale must be finite, greater than zero, and at most one.")

    updated = 0
    with torch.no_grad():
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            gradient = gradient.detach()
            if gradient.is_sparse:
                raise RuntimeError("AdamW virtual calibration does not support sparse gradients.")
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("Candidate gradient contains non-finite values.")

            group = _optimizer_group_for_parameter(optimizer, parameter)
            parameter_before = (
                parameter.detach().clone() if update_scale < 1.0 else None
            )
            beta1, beta2 = group.get("betas", optimizer.defaults.get("betas", (0.9, 0.999)))
            epsilon = float(group.get("eps", optimizer.defaults.get("eps", 1e-8)))
            learning_rate = float(group.get("lr", optimizer.defaults.get("lr", 0.0)))
            weight_decay = float(
                group.get("weight_decay", optimizer.defaults.get("weight_decay", 0.0))
            )
            correct_bias = bool(
                group.get("correct_bias", optimizer.defaults.get("correct_bias", True))
            )

            state = optimizer.state.get(parameter, {})
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg is None:
                exp_avg = torch.zeros_like(parameter)
            else:
                exp_avg = exp_avg.detach().clone()
            if exp_avg_sq is None:
                exp_avg_sq = torch.zeros_like(parameter)
            else:
                exp_avg_sq = exp_avg_sq.detach().clone()
            if exp_avg.shape != parameter.shape or exp_avg_sq.shape != parameter.shape:
                raise ValueError("AdamW moment shape does not match candidate parameter.")

            step = _state_step_as_int(state.get("step", 0)) + 1
            exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                gradient, gradient, value=1.0 - beta2
            )
            denominator = exp_avg_sq.sqrt().add_(epsilon)
            step_size = learning_rate
            if correct_bias:
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size *= math.sqrt(bias_correction2) / bias_correction1

            parameter.addcdiv_(exp_avg, denominator, value=-step_size)
            if weight_decay > 0.0:
                parameter.add_(parameter, alpha=-learning_rate * weight_decay)
            if parameter_before is not None:
                # Real newly activated parameters use delta warmup after the
                # ordinary AdamW step. Mirror that first-step effective LR
                # without advancing any real optimizer state.
                parameter.copy_(
                    parameter_before
                    + (parameter.detach() - parameter_before) * update_scale
                )
            updated += 1
    return updated


def _loss_tensor(loss):
    if isinstance(loss, (tuple, list)):
        loss = loss[0]
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise TypeError("loss_fn must return a scalar Tensor or a tuple beginning with one.")
    return loss


def _fold_seed(base_rng_state, fold_index, stream):
    digest = hashlib.sha256()
    digest.update(base_rng_state.torch_cpu_state.cpu().numpy().tobytes())
    digest.update(repr(base_rng_state.python_state).encode("utf-8"))
    digest.update(str(int(fold_index)).encode("ascii"))
    digest.update(stream.encode("ascii"))
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)


def _seed_fold_rng(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def _candidate_specs(candidate_module_names, module_map, rank_increment):
    if len(set(candidate_module_names)) != len(candidate_module_names):
        raise ValueError("candidate_module_names must be unique.")
    specs = []
    for name in candidate_module_names:
        if name not in module_map:
            raise ValueError("Unknown candidate LoRA module: {}.".format(name))
        module = module_map[name]
        specs.append((module, get_candidate_rank_parameters(module, rank_increment)))
    return specs


def score_virtual_candidate(
    model,
    optimizer,
    scheduler,
    candidate_module_names,
    module_map,
    rank_increment,
    calibration_batch_pairs,
    loss_fn,
    beta,
    max_grad_norm=None,
    loss_scale=1.0,
    virtual_update_scale=1.0,
):
    """Score one variable-size candidate with reversible virtual AdamW updates.

    ``calibration_batch_pairs`` must contain ``(batch_A, batch_B)`` pairs drawn
    by the caller exclusively from the training set.  For each fold, batch A
    supplies gradients and a different batch B supplies the before/after loss.
    The before loss is measured on the current pre-activation model; the after
    loss is measured after candidate activation and its virtual update.  The
    identical B-side RNG state is used for both, so dropout noise cannot
    masquerade as a candidate gain.
    """

    started = time.perf_counter()
    candidate_names = tuple(candidate_module_names)
    lcb_beta = float(beta)
    if not math.isfinite(lcb_beta) or lcb_beta < 0.0:
        raise ValueError("beta must be finite and nonnegative.")
    loss_scale = float(loss_scale)
    if not math.isfinite(loss_scale) or loss_scale <= 0.0:
        raise ValueError("loss_scale must be finite and positive.")
    virtual_update_scale = float(virtual_update_scale)
    if (
        not math.isfinite(virtual_update_scale)
        or virtual_update_scale <= 0.0
        or virtual_update_scale > 1.0
    ):
        raise ValueError(
            "virtual_update_scale must be finite, greater than zero, and at most one."
        )
    pairs = list(calibration_batch_pairs)
    if not pairs:
        raise ValueError("At least one calibration batch pair is required.")
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in pairs):
        raise ValueError("Each calibration fold must be a (batch_A, batch_B) pair.")

    # k=0 is a valid no-growth candidate and performs no model forwards.
    if not candidate_names:
        runtime = time.perf_counter() - started
        return {
            "candidate_modules": [],
            "candidate_size": 0,
            "candidate_cost": 0,
            "fold_gains": [0.0 for _ in pairs],
            "fold_details": [
                {"fold": index, "loss_before": None, "loss_after": None, "gain": 0.0}
                for index in range(len(pairs))
            ],
            "calibration_gain_mean": 0.0,
            "calibration_gain_std": 0.0,
            "calibration_gain_lcb": 0.0,
            "calibration_gain_per_parameter": 0.0,
            "calibration_valid": True,
            "loss_scale": loss_scale,
            "virtual_update_scale": virtual_update_scale,
            "calibration_runtime_seconds": runtime,
        }

    specs = _candidate_specs(candidate_names, module_map, rank_increment)
    candidate_parameters = []
    affected_parameters = []
    for _, spec in specs:
        candidate_parameters.extend(spec["parameters"])
        affected_parameters.extend(spec["parameters"])
        affected_parameters.append(spec["ranknum"])
    if len({id(parameter) for parameter in candidate_parameters}) != len(candidate_parameters):
        raise ValueError("Candidate modules share proposed rank parameters unexpectedly.")
    candidate_cost = int(sum(parameter.numel() for parameter in candidate_parameters))
    if candidate_cost <= 0:
        raise ValueError("A positive-rank candidate must have positive parameter cost.")

    snapshot = VirtualCalibrationSnapshot(
        model, optimizer, scheduler, affected_parameters
    )
    fold_details = []
    try:
        for fold_index, (batch_a, batch_b) in enumerate(pairs):
            # Every fold starts from the exact real-trajectory state.
            snapshot.restore()
            before_seed = _fold_seed(snapshot.rng_state, fold_index, "batch_b")
            _seed_fold_rng(before_seed)
            with torch.no_grad():
                # The common baseline is the current, pre-activation model.
                loss_before = _loss_tensor(loss_fn(model, _clone_value(batch_b)))
                loss_before_value = float(loss_before.detach().float().item())
            if not math.isfinite(loss_before_value):
                fold_details.append(
                    {
                        "fold": fold_index,
                        "loss_before": loss_before_value,
                        "loss_after": float("nan"),
                        "gain": float("nan"),
                    }
                )
                break

            # Discard any forward-mutated buffers before the candidate branch.
            snapshot.restore()
            _activate_candidate(specs)
            for parameter in model.parameters():
                parameter.grad = None
            gradient_seed = _fold_seed(snapshot.rng_state, fold_index, "batch_a")
            _seed_fold_rng(gradient_seed)
            loss_a = _loss_tensor(loss_fn(model, _clone_value(batch_a)))
            if not bool(torch.isfinite(loss_a.detach()).all()):
                fold_details.append(
                    {
                        "fold": fold_index,
                        "loss_before": loss_before_value,
                        "loss_after": float("nan"),
                        "gain": float("nan"),
                    }
                )
                break
            # Mirror native AMP loss scaling without touching the real
            # GradScaler's growth tracker or found-inf state. The Trainer passes
            # its current scale, and we unscale every model gradient before the
            # same global clipping rule used by the real step.
            (loss_a * loss_scale).backward()
            if loss_scale != 1.0:
                with torch.no_grad():
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(loss_scale)
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                fold_details.append(
                    {
                        "fold": fold_index,
                        "loss_before": loss_before_value,
                        "loss_after": float("nan"),
                        "gain": float("nan"),
                    }
                )
                break
            if max_grad_norm is not None and float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(max_grad_norm),
                    error_if_nonfinite=False,
                )
            try:
                stateless_adamw_update(
                    candidate_parameters,
                    optimizer,
                    update_scale=virtual_update_scale,
                )
            except FloatingPointError:
                fold_details.append(
                    {
                        "fold": fold_index,
                        "loss_before": loss_before_value,
                        "loss_after": float("nan"),
                        "gain": float("nan"),
                    }
                )
                break

            # Reuse exactly the pre-update B-side dropout/randomness stream.
            _seed_fold_rng(before_seed)
            with torch.no_grad():
                loss_after = _loss_tensor(loss_fn(model, _clone_value(batch_b)))
                loss_after_value = float(loss_after.detach().float().item())
            gain = loss_before_value - loss_after_value
            fold_details.append(
                {
                    "fold": fold_index,
                    "loss_before": loss_before_value,
                    "loss_after": loss_after_value,
                    "gain": gain,
                }
            )
    finally:
        snapshot.restore()

    fold_gains = [detail["gain"] for detail in fold_details]
    valid = bool(fold_gains) and all(math.isfinite(value) for value in fold_gains)
    if valid:
        mean = sum(fold_gains) / len(fold_gains)
        variance = sum((value - mean) ** 2 for value in fold_gains) / len(fold_gains)
        std = math.sqrt(max(0.0, variance))
        lcb = mean - lcb_beta * std
        gain_per_parameter = mean / max(candidate_cost, 1)
    else:
        mean = float("nan")
        std = float("nan")
        lcb = float("nan")
        gain_per_parameter = float("nan")

    return {
        "candidate_modules": list(candidate_names),
        "candidate_size": len(candidate_names) * int(rank_increment),
        "candidate_cost": candidate_cost,
        "fold_gains": fold_gains,
        "fold_details": fold_details,
        "calibration_gain_mean": mean,
        "calibration_gain_std": std,
        "calibration_gain_lcb": lcb,
        "calibration_gain_per_parameter": gain_per_parameter,
        "calibration_valid": valid,
        "loss_scale": loss_scale,
        "virtual_update_scale": virtual_update_scale,
        "calibration_runtime_seconds": time.perf_counter() - started,
    }


__all__ = [
    "RNGState",
    "VirtualCalibrationSnapshot",
    "capture_rng_state",
    "get_candidate_rank_parameters",
    "restore_rng_state",
    "score_virtual_candidate",
    "stateless_adamw_update",
]
