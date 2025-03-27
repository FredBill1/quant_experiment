from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from .config import MODEL_PATH

PATH = MODEL_PATH.parent / "search/tboard"


def main() -> None:
    for run in PATH.iterdir():
        event_files = list(filter(Path.is_file, run.iterdir()))
        if len(event_files) <= 1:
            continue
        print(run)
        data = defaultdict(list)
        for event_file in event_files:
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            for key in ea.scalars.Keys():
                data[key].extend(ea.scalars.Items(key))

        with SummaryWriter(str(run)) as writer:
            for key, values in data.items():
                values.sort(key=lambda x: x.step)
                for value in values:
                    writer.add_scalar(key, value.value, value.step)

        backup = run / "backup"
        backup.mkdir(exist_ok=True, parents=True)
        for event_file in event_files:
            event_file.rename(backup / event_file.name)
    print("Done")


if __name__ == "__main__":
    main()
