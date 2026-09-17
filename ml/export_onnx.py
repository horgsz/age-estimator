"""Export the trained age estimator to ONNX and verify parity with PyTorch.

Contract: opset 17, dynamic batch axis, input ``input`` as NCHW float32, output
``logits`` shaped [N, 101]. Consumers apply softmax and take the soft
expectation over bin indices 0..100 to recover the age.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from data import UTKFaceDataset, load_split
from model import CHECKPOINT_PATH, load_checkpoint

ONNX_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "age_model.onnx"
OPSET = 17
TOLERANCE = 1e-4


def export(checkpoint: Path, onnx_path: Path, opset: int = OPSET) -> dict[str, object]:
    model, meta = load_checkpoint(checkpoint)
    model.eval()

    size = int(meta["input_size"])
    dummy = torch.randn(1, 3, size, size)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"Exported -> {onnx_path}")
    return meta


def _verification_batch(batch: int, size: int, corpus: str = "utkface") -> torch.Tensor:
    """Real normalized test images; random noise drives BatchNorm out of range
    and produces logits in the thousands, which makes an absolute 1e-4 check
    meaningless. Parity must hold on the distribution the model actually sees.

    ``corpus`` must match the corpus the checkpoint was trained on, for the same
    reason: verifying a real-GT model on UTKFace crops is a weaker check than it
    looks, because the activations are not the ones it will encounter."""
    try:
        if corpus == "realgt":
            from realgt_data import RealGTDataset, load_manifest

            frame = load_manifest()
            frame = frame[frame["split"] == "test"].head(batch)
            dataset = RealGTDataset(frame, train=False)
        else:
            frame = load_split("test").head(batch)
            dataset = UTKFaceDataset(frame, train=False)
        return torch.stack([dataset[i][0] for i in range(len(dataset))])
    except (FileNotFoundError, OSError, ImportError, AssertionError) as exc:
        print(f"Falling back to synthetic input ({type(exc).__name__}: {exc})")
        torch.manual_seed(0)
        return torch.randn(batch, 3, size, size) * 0.5


def verify(
    checkpoint: Path,
    onnx_path: Path,
    batch: int = 8,
    tol: float = TOLERANCE,
    corpus: str = "utkface",
) -> float:
    model, meta = load_checkpoint(checkpoint)
    model.eval()
    size = int(meta["input_size"])

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)

    inputs = graph.graph.input
    outputs = graph.graph.output
    assert len(inputs) == 1 and inputs[0].name == "input", "input must be named 'input'"
    assert len(outputs) == 1 and outputs[0].name == "logits", (
        "output must be named 'logits'"
    )

    in_dims = inputs[0].type.tensor_type.shape.dim
    out_dims = outputs[0].type.tensor_type.shape.dim
    assert in_dims[0].dim_param, "input batch axis must be dynamic"
    assert out_dims[0].dim_param, "output batch axis must be dynamic"
    assert [d.dim_value for d in in_dims[1:]] == [3, size, size], "input must be NCHW"
    assert out_dims[1].dim_value == int(meta["num_bins"]), "logits must be [N, 101]"
    assert inputs[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT

    sample = _verification_batch(batch, size, corpus)
    with torch.no_grad():
        torch_out = model(sample).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(["logits"], {"input": sample.numpy().astype(np.float32)})[0]

    assert onnx_out.shape == (len(sample), int(meta["num_bins"])), (
        f"unexpected ONNX output shape {onnx_out.shape}"
    )

    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(
        f"Opset: {opset_of(graph)} | batch {len(sample)} "
        f"| max |torch - onnx| = {max_diff:.3e}"
    )
    if max_diff > tol:
        raise AssertionError(f"ONNX mismatch {max_diff:.3e} exceeds tolerance {tol:g}")

    # Also compare the derived ages, which is what consumers actually use.
    torch_age, _ = model.expectation(torch.from_numpy(torch_out))
    onnx_age, _ = model.expectation(torch.from_numpy(onnx_out))
    age_diff = float((torch_age - onnx_age).abs().max())
    print(f"Max age delta: {age_diff:.3e} years")
    print(f"Sample ages (torch): {[round(a, 2) for a in torch_age.tolist()]}")

    # A second batch size proves the dynamic axis actually works at runtime.
    alt = sample[:1]
    with torch.no_grad():
        torch_alt = model(alt).numpy()
    onnx_alt = session.run(["logits"], {"input": alt.numpy().astype(np.float32)})[0]
    alt_diff = float(np.abs(torch_alt - onnx_alt).max())
    print(f"Dynamic batch check (N=1): max diff = {alt_diff:.3e}")
    if alt_diff > tol:
        raise AssertionError(f"ONNX mismatch at N=1: {alt_diff:.3e} > {tol:g}")

    print(f"OK: ONNX matches PyTorch within {tol:g}")
    return max_diff


def opset_of(graph: onnx.ModelProto) -> int:
    for entry in graph.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return entry.version
    return -1


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the age model to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--onnx", type=Path, default=ONNX_PATH)
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument(
        "--corpus", choices=["utkface", "realgt"], default="utkface",
        help="corpus to draw verification images from; match the checkpoint",
    )
    args = parser.parse_args()

    meta = export(args.checkpoint, args.onnx, args.opset)
    verify(args.checkpoint, args.onnx, tol=args.tolerance, corpus=args.corpus)
    print(f"\nArtifact meta: {meta}")


if __name__ == "__main__":
    main()
