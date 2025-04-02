from pathlib import Path

import pandas as pd
import plotly.express as px
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .config import MODEL_PATH

PATH = MODEL_PATH.parent / "decompose_quant/tboard"


def main():
    epochs = []
    accs = []
    methods = []
    factors = []
    runs = list(filter(Path.is_dir, PATH.iterdir()))
    for run in runs:
        method, factor = run.name.split("_")
        data = []
        for event_file in run.iterdir():
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            data.extend(ea.Scalars("acc/test"))
        data.sort(key=lambda x: x.step)
        epochs.extend(x.step for x in data)
        accs.extend(x.value - data[0].value for x in data)  # accuracy - accuracy_before_finetune
        methods.extend([method] * len(data))
        factors.extend([factor] * len(data))

    df = pd.DataFrame(dict(Epoch=epochs, Accuracy=accs, Method=methods, Factor=factors))
    df["Run"] = df["Method"] + "_" + df["Factor"]
    fig = px.line(df, x="Epoch", y="Accuracy", color="Run", title="Test Accuracy Learning Curves")
    fig.show()


if __name__ == "__main__":
    main()
