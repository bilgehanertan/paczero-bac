import argparse
import dataclasses
import os
import subprocess
from pathlib import Path

from paczo import config, run_paczo

DP_OPT = config.REPO / "baselines" / "dp-aggzo" / "opt"


def dp_launch(run_id, env):
    link = DP_OPT / "result" / f"{env.TASK}-{env.MODEL.split('/')[-1]}-{run_id}"
    overlay = {k: str(v) for k, v in dataclasses.asdict(env).items()} | {"TAG": run_id, "PYTHONPATH": str(DP_OPT)}
    return link, config.REPO / "result" / "runs" / run_id, overlay


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m paczo.run")
    p.add_argument("config", type=Path, help="experiment file under configs/")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run id")
    g.add_argument("--list", action="store_true", help="print the run ids of the file")
    p.add_argument("--dry-run", action="store_true", help="print the resolved configuration and exit")
    a = p.parse_args(argv)

    runs = config.load(a.config)
    if a.list:
        print("\n".join(runs))
        return
    if a.run not in runs:
        p.error(f"{a.config} has no run {a.run}")
    run = runs[a.run]
    args = config.build(run)
    print(config.format_resolved(run, args), flush=True)

    if run.runner == "paczo":
        if not a.dry_run:
            Path(args.result_file).parent.mkdir(parents=True, exist_ok=True)
            run_paczo.main(args)
        return
    link, out, overlay = dp_launch(run.id, args)
    print(f"launcher {DP_OPT / 'examples' / 'dpaggzo.sh'}\ncwd      {DP_OPT}\nTAG      {run.id}\nresult   {link} -> {out}", flush=True)
    if a.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(out)
    subprocess.run(["bash", "examples/dpaggzo.sh"], cwd=DP_OPT, env=os.environ | overlay, check=True)


if __name__ == "__main__":
    main()
