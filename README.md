#  Reinforment learning Doom Agent

A DDQN reinforcement learning agent that can learn to play doom under a level.
The idea is kill largest amount of enemies in an episode.

## Requeriments

* conda
* A GPU can improve training time: Is required that nvidia module is loaded.

## Setup

**Step 1:** Create project environment.

```bash
conda env create --file environment.yml
```

**Step 2:** Before all you need activate environment every time you use it with this:

```bash
source activate doom-agent
```

or forget use it defining shortcuts(aliases):

**bash**:
```bash
./setup_shortcuts bash; source ~/.bashrc
```
or this if you use zsh:

**zsh**:
```bash
./setup_shortcuts zsh; source ~/.zshrc
```

Also exist use gpu option with optiprime adding gup param like this:

```bash
./setup_shortcuts zsh gpu; source ~/.zshrc
```

## Use

* Train agent:

```bash
agent-train [--weights weights_file_path]
```
Note: When train process finish you can see report/weights_file under reports path.

* See evolution of train process loading tensor board:
```bash
agent-metrics
```
After go to dash: http://localhost:6006

* Play agent:

```bash
agent-play --weights weights/best_2019_02_09_17_19_52-weights-loss_0.0123.h5
```

* To view all options:

```bash
agent-train --help
```

or

```bash
agent-play --help
```
