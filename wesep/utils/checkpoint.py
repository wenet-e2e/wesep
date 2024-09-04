from typing import List, Optional

import torch

from wesep.utils.schedulers import BaseClass


def load_pretrained_model(model: torch.nn.Module,
                          path: str,
                          type: str = "generator"):
    assert type in ["generator", "discriminator"]
    states = torch.load(
        path,
        map_location="cpu",
    )
    if type == "generator":
        state = states["models"][0]
    else:
        assert len(states["models"]) == 2
        state = states["models"][1]

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(state)
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)


def load_checkpoint(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[BaseClass],
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str,
    only_model: bool = False,
    mode: str = "all",
):
    assert mode in ["all", "generator", "discriminator"]
    states = torch.load(
        path,
        map_location="cpu",
    )
    if mode == "generator":
        model_state, optimizer_state, scheduler_state = (
            [states["models"][0]],
            [states["optimizers"][0]],
            [states["schedulers"][0]],
        )
    elif mode == "discriminator":
        model_state, optimizer_state, scheduler_state = (
            [states["models"][1]],
            [states["optimizers"][1]],
            [states["schedulers"][1]],
        )
    else:
        model_state, optimizer_state, scheduler_state = (
            states["models"],
            states["optimizers"],
            states["schedulers"],
        )

    for model, state in zip(models, model_state):
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state, strict=False)
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
    if not only_model:
        for optimizer, state in zip(optimizers, optimizer_state):
            optimizer.load_state_dict(state)
        for scheduler, state in zip(schedulers, scheduler_state):
            if scheduler is not None:
                scheduler.load_state_dict(state)
        if scaler is not None:
            if states["scaler"] is not None:
                scaler.load_state_dict(states["scaler"])


def save_checkpoint(
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    schedulers: List[BaseClass],
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str,
):
    if isinstance(models[0], torch.nn.DataParallel):
        state_dict = [model.module.state_dict() for model in models]
    elif isinstance(models[0], torch.nn.parallel.DistributedDataParallel):
        state_dict = [model.module.state_dict() for model in models]
    else:
        state_dict = [model.state_dict() for model in models]
    torch.save(
        {
            "models":
            state_dict,
            "optimizers": [o.state_dict() for o in optimizers],
            "schedulers":
            [s.state_dict() if s is not None else None for s in schedulers],
            "scaler":
            scaler.state_dict() if scaler is not None else None,
        },
        path,
    )
