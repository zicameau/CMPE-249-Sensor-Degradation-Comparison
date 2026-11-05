# BEVFusion setup

`bevfusion/` is a git submodule pointing to our fork, [christiankso/bevfusion](https://github.com/christiankso/bevfusion), branch `cmpe249-fixes`. The fork is based on [mit-han-lab/bevfusion](https://github.com/mit-han-lab/bevfusion) commit `326653d`; the upstream repo does not build or run as-is, and our fix commit lists every change. See [../results/baseline_fusion.md](../results/baseline_fusion.md) for what each fix does.

## Reproduce

BEVFusion needs Linux. Run it from a clone inside WSL2 Ubuntu, not from this Windows checkout (files under `/mnt/c` are slow from Linux):

```bash
git clone -b cmpe249-fixes https://github.com/christiankso/bevfusion.git ~/bevfusion
cd ~/bevfusion/docker && docker build . -t bevfusion
```

Then follow the container, data-prep, and test steps in [../results/baseline_fusion.md](../results/baseline_fusion.md).

To get the submodule when cloning this repo:

```bash
git clone --recursive -b BEVFusion https://github.com/zicameau/CMPE-249-Sensor-Degradation-Comparison.git
```

## Updating the submodule after new BEVFusion commits

1. In `~/bevfusion` (WSL): commit, then `git push fork cmpe249-fixes`.
2. In this repo:

```bash
git submodule update --remote bevfusion
git add bevfusion
git commit -m "Bump bevfusion submodule"
```
