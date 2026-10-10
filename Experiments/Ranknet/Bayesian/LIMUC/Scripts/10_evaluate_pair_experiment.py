"""Save unseen-patient performance against cumulative annotation budget."""

import argparse
from pathlib import Path

import pandas as pd
from pair_evaluation import evaluate_predictions


def main():
    """Evaluate one experiment round and update its comparison CSV."""
    parser = argparse.ArgumentParser(
        description="Evaluate unseen-patient rankings for a pair-acquisition round."
    )
    parser.add_argument("--prediction-summary-csv", required=True, type=Path)
    parser.add_argument("--cumulative-pair-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--round", dest="round_number", required=True, type=int)
    args = parser.parse_args()

    prediction_data = pd.read_csv(args.prediction_summary_csv.expanduser())
    cumulative_pairs = pd.read_csv(args.cumulative_pair_csv.expanduser())
    metrics = evaluate_predictions(prediction_data)
    row = {
        "experiment_id": args.experiment_id,
        "method": args.method,
        "fold": args.fold,
        "seed": args.seed,
        "round": args.round_number,
        "cumulative_pair_count": len(cumulative_pairs),
        **metrics,
    }

    output_path = args.output_csv.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output = pd.read_csv(output_path)
        identity = ["experiment_id", "method", "fold", "seed", "round"]
        matching = pd.Series(True, index=output.index)
        for column in identity:
            matching &= output[column].astype(str) == str(row[column])
        output = output.loc[~matching]
        output = pd.concat([output, pd.DataFrame([row])], ignore_index=True)
    else:
        output = pd.DataFrame([row])
    output = output.sort_values(
        ["method", "seed", "fold", "cumulative_pair_count"]
    ).reset_index(drop=True)
    output.to_csv(output_path, index=False)
    print(f"Saved performance comparison: {output_path}")


if __name__ == "__main__":
    main()
