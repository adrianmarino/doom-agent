#  Reinforment learning Doom Agent

A DDQN reinforcement learning agent that can learn to play doom under a level.

## Requeriments

* conda
* A respectable video card (i.e. GeForce GTX 1060 or higher)

## Setup

**Step 1:** Create project environment.

```bash
conda env create --file environment.yml
```

**Step 2:** Before all you need activate environment every time you use it with this:

```bash
source activate doom-agent
```

or forget use it creating a bash/zsh alias:

**bash**:
```bash
echo -e "export PATH=.:\$PATH" >> ~/.bashrc
echo "alias agent-train='conda activate doom-agent;rm -rf logs metrics checkpoints; python agent-train.py'" >> ~/.bashrc
echo "alias agent-play='conda activate doom-agent;python agent-play.py'" >> ~/.bashrc
echo "alias agent-metrics='conda activate doom-agent; tensorboard --logdir metrics'" >> ~/.bashrc
source ~/.bashrc
```

**zsh**:
```bash
echo -e "export PATH=.:\$PATH" >> ~/.zshrc
echo "alias agent-train='conda activate doom-agent;rm -rf logs metrics checkpoints; python agent-train.py'" >> ~/.zshrc
echo "alias agent-play='conda activate doom-agent;python agent-play.py'" >> ~/.zshrc
echo "alias agent-metrics='conda activate doom-agent; tensorboard --logdir metrics'" >> ~/.zshrc
source ~/.zshrc
```

with this you can use object detector as a regular command in the following way:

```bash
agent-train params
```

instead of:

```bash
conda activate doom-agent
rm -rf logs metrics checkpoints
python agent-train.py params
```

## Use

* Train agent:

```bash
agent-train [--weights a_weights_file]
```
Note: When train process finish you can see report/weights_file under reports path.

* See evolution of train process:
```bash
agent-metrics
```
After go to dash: http://localhost:6006

* Play agent:

```bash
agent-play --weights reports/a_weights_file
```

* To view all options:

```bash
agent-train/play --help
```


## Metrics

Run tensorboard to see metrics dash
```bash
tensorboard --logdir metrics
```