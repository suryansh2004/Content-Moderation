from __future__ import annotations

import argparse
from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import AutoOptimizationConfig
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--optimization-level", choices=["O1", "O2", "O3", "O4"], default="O2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ORTModelForSequenceClassification.from_pretrained(model_dir, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    optimizer = ORTOptimizer.from_pretrained(output_dir)
    optimization_config = getattr(AutoOptimizationConfig, args.optimization_level)()
    optimizer.optimize(save_dir=output_dir, optimization_config=optimization_config)

    print(f"ONNX model exported and optimized in {output_dir}")


if __name__ == "__main__":
    main()
