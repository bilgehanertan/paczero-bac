#!/usr/bin/env python3
import argparse, json, os, sys, tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from src.tasks import get_task
    from paczo.run_paczo import Framework, set_seed

    targs = C.load_training_args(os.path.join(a.ckpt_dir, "training_args.bin"))
    targs.model_name = C.resolve_model_path(targs.model_name)
    tmp = tempfile.mkdtemp(prefix="mia_dump_")
    targs.output_dir, targs.logging_dir, targs.pac_diag_dir = tmp, tmp, tmp
    targs.result_file = os.path.join(tmp, "metrics.json")
    targs.run_name = "mia_dump_replay"
    targs.pac_dump_membership = os.path.abspath(a.out)
    targs.report_to = []

    set_seed(targs.seed)
    task = get_task(targs.task_name)
    train = task.sample_train_set(num_train=targs.num_train, num_dev=targs.num_dev, seed=targs.train_set_seed)
    fw = Framework(targs, task)
    try:
        fw.train(train[: -targs.num_dev], train[-targs.num_dev:])
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    dump = np.load(a.out)["membership"].astype(bool)
    mat, msha = C.load_matrix()
    res = {"task": targs.task_name, "train_set_seed": targs.train_set_seed, "dump_shape": list(dump.shape),
           "equal_to_frozen_matrix": bool((dump == mat).all()), "n_rows_differing": int((dump != mat).any(1).sum()),
           "frozen_matrix_sha256": msha}
    print("DUMP_CHECK", json.dumps(res), flush=True)
    sys.exit(0 if res["equal_to_frozen_matrix"] else 1)


if __name__ == "__main__":
    main()
