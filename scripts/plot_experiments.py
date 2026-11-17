#!/usr/bin/env python3
"""Render a verified experiment summary as standalone PNG/PDF figures."""
import argparse
from pathlib import Path
import tempfile

from experiment_common import atomic_json, digest, load
from experiments import collect


def render(summary, destination, suite, metric="decode_tokens_per_second", x="input_tokens"):
    summary = Path(summary).resolve(strict=True)
    value = load(summary)
    with tempfile.TemporaryDirectory(prefix="solid-plot-audit-") as temporary:
        verified = collect(value["plan"],Path(temporary)/"audit")
    if verified != value: raise ValueError("summary differs from its original audited runs")
    if metric not in ("decode_tokens_per_second","decode_loop_ms","prefill_ms","first_token_ms","request_ms","kv_system_tracked_bytes"):
        raise ValueError("unsupported plotted metric")
    if x not in ("input_tokens","block_tokens","budget_tokens"): raise ValueError("unsupported x axis")
    groups = [group for group in value["groups"] if group["group"]["suite"] == suite]
    if not groups or any(group["complete"] != group["expected"] for group in groups):
        raise ValueError("figures require a nonempty suite with all planned repetitions complete")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scale = 1/(1024**3) if metric == "kv_system_tracked_bytes" else 1
    labels = {"decode_tokens_per_second":"Decode throughput (token/s)","decode_loop_ms":"Decode loop (ms)",
              "prefill_ms":"Prefill (ms)","first_token_ms":"Prefill + first sample (ms)","request_ms":"Request (ms)",
              "kv_system_tracked_bytes":"Tracked KV system allocation (GiB)"}
    panels = sorted({(group["group"]["model"],group["group"]["condition"]) for group in groups})
    figure,axes = plt.subplots(1,len(panels),figsize=(5*len(panels),3.6),squeeze=False,constrained_layout=True)
    for ax,panel in zip(axes[0],panels):
        lines = {}
        for group in groups:
            identity = group["group"]
            if (identity["model"],identity["condition"]) != panel: continue
            label = identity["variant"] + ", " + ", ".join(
                {"input_tokens":"N", "block_tokens":"B", "budget_tokens":"C"}[key]+"="+
                ("auto" if key == "budget_tokens" and not identity[key] else str(identity[key]))
                for key in ("input_tokens","block_tokens","budget_tokens") if key != x)
            lines.setdefault(label,[]).append((identity[x],group["metrics"][metric]))
        for label,points in sorted(lines.items()):
            points.sort()
            if len({point[0] for point in points}) != len(points): raise ValueError("duplicate plot coordinates")
            if any(point[1] is None for point in points): raise ValueError("metric has no observations")
            ys = [point[1]["median"]*scale for point in points]
            ax.errorbar([point[0] for point in points],ys,
                        yerr=[[y-point[1]["min"]*scale for y,point in zip(ys,points)],
                              [point[1]["max"]*scale-y for y,point in zip(ys,points)]],
                        marker="o",capsize=3,linewidth=1.3,label=label)
        if x == "input_tokens": ax.set_xscale("log",base=2)
        ax.set(xlabel=x.replace("_"," "),ylabel=labels[metric],title=" / ".join(panel))
        ax.grid(axis="y",alpha=.25); ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=8,frameon=False)
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    for extension in ("png","pdf"): figure.savefig(destination/("figure."+extension),dpi=220)
    plt.close(figure)
    atomic_json(destination/"figure.json",{"summary":str(summary),"summary_sha256":digest(summary),
                "matplotlib_version":matplotlib.__version__,"display_scale":scale,
                "plan_sha256":value["plan_sha256"],"provenance":value["provenance"],"suite":suite,"metric":metric,"x":x,
                "error_bars":"minimum and maximum across independent measured repetitions; center=median",
                "groups":groups,"files":{path.name:digest(path) for path in destination.iterdir() if path.is_file()}})
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary",type=Path); parser.add_argument("destination",type=Path); parser.add_argument("--suite",required=True)
    parser.add_argument("--metric",default="decode_tokens_per_second"); parser.add_argument("--x",default="input_tokens")
    args = parser.parse_args()
    try: print(render(args.summary,args.destination,args.suite,args.metric,args.x))
    except (OSError,ValueError,KeyError,ImportError) as error: parser.exit(2,f"plot_experiments: {error}\n")
