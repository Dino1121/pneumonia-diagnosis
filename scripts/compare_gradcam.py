"""Compare representative models with Grad-CAM++ on the same test images."""

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASSES = ("NORMAL", "PNEUMONIA")


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="data/splits/group_split.csv",
    )
    parser.add_argument(
        "--scratch_checkpoint",
        default="experiments/scratch/exp_008/best_model.pth",
    )
    parser.add_argument(
        "--transfer_checkpoint",
        default="experiments/transfer/exp_003/best_model.pth",
    )
    parser.add_argument(
        "--filename",
        help="Exact filename or CSV filepath of ONE test image",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output_dir",
        default="results/gradcam",
    )

    args = parser.parse_args()

    if args.num_images < 1:
        parser.error("num_images must be >= 1")

    return args


def select_rows(csv_path, args):
    with csv_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)

        required = {
            "filepath",
            "label",
            "split",
        }

        if not required.issubset(
            reader.fieldnames or []
        ):
            raise ValueError(
                f"CSV must contain {sorted(required)}"
            )

        rows = [
            row
            for row in reader
            if row["split"] == "test"
        ]

    if args.filename:
        rows = [
            row
            for row in rows
            if args.filename in (
                row["filepath"],
                Path(row["filepath"]).name,
            )
        ]

        if len(rows) != 1:
            raise ValueError(
                f"Expected exactly one test image for "
                f"{args.filename!r}; found {len(rows)}. "
                f"Use its full CSV filepath if ambiguous."
            )

    else:
        normal_rows = [
            row
            for row in rows
            if row["label"] == "NORMAL"
        ]

        pneumonia_rows = [
            row
            for row in rows
            if row["label"] == "PNEUMONIA"
        ]

        rng = random.Random(args.seed)

        rng.shuffle(normal_rows)
        rng.shuffle(pneumonia_rows)

        n_normal = args.num_images // 2
        n_pneumonia = args.num_images - n_normal

        if n_normal > len(normal_rows):
            raise ValueError(
                f"Requested {n_normal} NORMAL images, "
                f"but only {len(normal_rows)} are available "
                f"in the test split."
            )

        if n_pneumonia > len(pneumonia_rows):
            raise ValueError(
                f"Requested {n_pneumonia} PNEUMONIA images, "
                f"but only {len(pneumonia_rows)} are available "
                f"in the test split."
            )

        rows = (
            normal_rows[:n_normal]
            + pneumonia_rows[:n_pneumonia]
        )

        rng.shuffle(rows)

    if not rows:
        raise ValueError(
            "No test images selected. "
            "Check filename and CSV split values."
        )

    for row in rows:
        if row["label"] not in CLASSES:
            raise ValueError(
                f"Unknown label: {row['label']}"
            )

        image_path = project_path(
            row["filepath"]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

    return rows


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    args = parse_args()

    csv_path = project_path(
        args.csv
    )

    rows = select_rows(
        csv_path,
        args,
    )

    checkpoints = [
        project_path(
            args.scratch_checkpoint
        ),
        project_path(
            args.transfer_checkpoint
        ),
    ]

    for path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

    import importlib.metadata

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from PIL import Image
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import (
        ClassifierOutputTarget,
    )

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

    from src.data.transforms import (
        eval_transform,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )

    from src.models.resnet import (
        create_resnet50,
    )

    if args.device == "auto":
        device_name = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device_name = args.device

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is unavailable. "
            "Use --device cpu."
        )

    device = torch.device(
        device_name
    )

    output_dir = (
        project_path(
            args.output_dir
        )
        / datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    models = []

    for path in checkpoints:
        model = create_resnet50(
            model_type="scratch",
            num_classes=2,
        )

        state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model = (
            model
            .to(device)
            .eval()
        )

        models.append(
            model
        )

    mean = np.asarray(
        IMAGENET_MEAN,
        dtype=np.float32,
    )

    std = np.asarray(
        IMAGENET_STD,
        dtype=np.float32,
    )

    records = []

    names = [
        f"Scratch / {checkpoints[0].parent.name}",
        f"TL / {checkpoints[1].parent.name}",
    ]

    with GradCAMPlusPlus(
        model=models[0],
        target_layers=[
            models[0].layer4[-1]
        ],
    ) as scratch_cam, GradCAMPlusPlus(
        model=models[1],
        target_layers=[
            models[1].layer4[-1]
        ],
    ) as transfer_cam:

        for index, row in enumerate(
            rows
        ):
            image_path = project_path(
                row["filepath"]
            )

            with Image.open(
                image_path
            ) as image:
                tensor = (
                    eval_transform(
                        image
                    )
                    .unsqueeze(0)
                    .to(device)
                )

            rgb = np.clip(
                (
                    tensor[0]
                    .detach()
                    .cpu()
                    .permute(
                        1,
                        2,
                        0,
                    )
                    .numpy()
                    * std
                    + mean
                ),
                0,
                1,
            )

            maps = []
            outcomes = []

            for model, cam in zip(
                models,
                (
                    scratch_cam,
                    transfer_cam,
                ),
            ):
                with torch.no_grad():
                    logits = model(
                        tensor
                    )

                    probabilities = (
                        torch.softmax(
                            logits,
                            dim=1,
                        )[0]
                        .cpu()
                        .numpy()
                    )

                predicted_index = int(
                    probabilities.argmax()
                )

                prediction = CLASSES[
                    predicted_index
                ]

                predicted_probability = float(
                    probabilities[
                        predicted_index
                    ]
                )

                with torch.enable_grad():
                    heatmap = cam(
                        input_tensor=tensor,
                        targets=[
                            ClassifierOutputTarget(
                                predicted_index
                            )
                        ],
                        aug_smooth=False,
                        eigen_smooth=False,
                    )[0]

                if not np.isfinite(
                    heatmap
                ).all():
                    raise ValueError(
                        f"Non-finite CAM: "
                        f"{row['filepath']}"
                    )

                outcomes.append(
                    {
                        "prediction":
                            prediction,
                        "predicted_class_index":
                            predicted_index,
                        "predicted_probability":
                            predicted_probability,
                        "p_normal":
                            float(
                                probabilities[0]
                            ),
                        "p_pneumonia":
                            float(
                                probabilities[1]
                            ),
                        "cam_target":
                            prediction,
                        "correct":
                            (
                                prediction
                                == row["label"]
                            ),
                        "cam_all_zero":
                            bool(
                                np.max(
                                    heatmap
                                )
                                == 0
                            ),
                    }
                )

                maps.append(
                    heatmap
                )

            correct = tuple(
                result["correct"]
                for result in outcomes
            )

            category = {
                (
                    True,
                    True,
                ):
                    "both_correct",
                (
                    True,
                    False,
                ):
                    "scratch_only_correct",
                (
                    False,
                    True,
                ):
                    "transfer_only_correct",
                (
                    False,
                    False,
                ):
                    "both_wrong",
            }[correct]

            fig, axes = plt.subplots(
                1,
                3,
                figsize=(
                    13,
                    5,
                ),
            )

            axes[0].imshow(
                rgb
            )

            axes[0].set_title(
                f"Input (resize + padding)\n"
                f"True: {row['label']}"
            )

            for (
                ax,
                name,
                result,
                heatmap,
            ) in zip(
                axes[1:],
                names,
                outcomes,
                maps,
            ):
                ax.imshow(
                    rgb
                )

                if not result[
                    "cam_all_zero"
                ]:
                    ax.imshow(
                        heatmap,
                        cmap="jet",
                        vmin=0,
                        vmax=1,
                        alpha=0.45,
                    )

                if result[
                    "cam_all_zero"
                ]:
                    suffix = (
                        "\nNo positive CAM signal"
                    )
                else:
                    suffix = ""

                predicted_class = (
                    result[
                        "prediction"
                    ]
                )

                predicted_probability = (
                    result[
                        "predicted_probability"
                    ]
                )

                ax.set_title(
                    f"{name}\n"
                    f"Pred: {predicted_class} | "
                    f"P({predicted_class}): "
                    f"{predicted_probability:.3f}\n"
                    f"CAM target: "
                    f"{result['cam_target']}"
                    f"{suffix}"
                )

            for ax in axes:
                ax.axis(
                    "off"
                )

            fig.suptitle(
                f"{Path(row['filepath']).name}\n"
                f"Grad-CAM++ predicted-class attribution | "
                f"{category}",
                fontsize=11,
            )

            fig.text(
                0.5,
                0.025,
                (
                    "Each CAM targets the class predicted "
                    "by that model. "
                    "CAMs are independently normalized to [0, 1]. "
                    "Color intensity is not confidence "
                    "or lesion severity."
                ),
                ha="center",
                fontsize=9,
            )

            fig.tight_layout(
                rect=(
                    0,
                    0.065,
                    1,
                    0.89,
                )
            )

            stem = (
                f"{index:04d}_"
                f"{Path(row['filepath']).stem}"
            )

            png_path = (
                output_dir
                / f"{stem}.png"
            )

            fig.savefig(
                png_path,
                dpi=160,
            )

            plt.close(
                fig
            )

            np.savez_compressed(
                (
                    output_dir
                    / f"{stem}_cams.npz"
                ),
                scratch=maps[0],
                transfer=maps[1],
            )

            records.append(
                {
                    "filepath":
                        row["filepath"],
                    "true_label":
                        row["label"],
                    "category":
                        category,
                    "figure":
                        png_path.name,
                    "scratch":
                        outcomes[0],
                    "transfer":
                        outcomes[1],
                }
            )

            print(
                f"Saved: "
                f"{png_path}"
            )

    manifest = {
        "method":
            "GradCAMPlusPlus",
        "target_mode":
            "predicted_class",
        "target_score":
            "pre-softmax logit",
        "target_layer":
            "layer4[-1]",
        "normalization":
            (
                "independent per image "
                "and model, [0,1]"
            ),
        "overlay_alpha":
            0.45,
        "colormap":
            "jet",
        "device":
            str(device),
        "split":
            "test",
        "csv":
            str(csv_path),
        "csv_sha256":
            sha256(
                csv_path
            ),
        "selection": {
            "filename":
                args.filename,
            "num_images":
                args.num_images,
            "seed":
                args.seed,
            "sampling":
                "class-balanced random",
        },
        "checkpoints": [
            {
                "path":
                    str(path),
                "sha256":
                    sha256(
                        path
                    ),
            }
            for path in checkpoints
        ],
        "versions": {
            name:
                importlib.metadata.version(
                    name
                )
            for name in (
                "torch",
                "torchvision",
                "grad-cam",
                "numpy",
                "matplotlib",
            )
        },
        "samples":
            records,
    }

    manifest_path = (
        output_dir
        / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Finished: "
        f"{len(records)} image(s). "
        f"Metadata: "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()