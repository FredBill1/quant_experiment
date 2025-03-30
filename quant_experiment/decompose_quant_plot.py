import pandas as pd
import plotly.express as px

from .config import MODEL_PATH

SAVED_MODEL_DIR = MODEL_PATH.with_name("decompose_quant")
RESULTS_PATH = SAVED_MODEL_DIR / "tboard/results.csv"


def main() -> None:
    df = pd.read_csv(RESULTS_PATH)

    df["decompose_method_and_do_finetune"] = df["decompose_method"].astype(str) + df["do_finetune"].map(lambda x: "-finetune" if x else "")
    df.loc[df["decompose_method"].isna(), "decompose_method_and_do_finetune"] = "baseline"

    fig = px.scatter(
        df,
        x="model_size",
        y="test_acc",
        color="decompose_method_and_do_finetune",
        hover_data=["decompose_method", "decompose_factor", "do_finetune", "quant_weight", "quant_act"],
        title="Model Size vs Test Accuracy",
        log_x=True,
    )

    fig.update_layout(
        title="Model Size vs Test Accuracy",
        xaxis_title="Model Size/bytes (log scale)",
        yaxis_title="Test Accuracy",
        legend_title="Decompose Method",
    )
    fig.show()


if __name__ == "__main__":
    main()
