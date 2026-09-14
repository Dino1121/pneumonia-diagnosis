import torch


def create_optimizer(
    model,
    optimizer_name,
    lr,
    weight_decay=0.0
):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )

    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    return optimizer