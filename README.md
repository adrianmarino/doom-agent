#  Reinforment learning Doom Agent

A **DDQN** reinforcement learning agent that can learn to play doom under a level.
The idea is kill largest amount of enemies in an episode.

## Requeriments

* conda
* A GPU can improve training time: Is required that nvidia module is loaded.

## Getting started

**Step 1:** Create project environment.

```bash
conda env create --file environment.yml
```

**Step 2:** Before all you need activate environment every time you use it with this:

```bash
source activate doom-agent
```

or forget use it defining shortcuts(aliases) as follow:

* If you are a **bash** shell user:

    ```bash
    ./setup-shortcuts bash; source ~/.bashrc
    ```

* If you are a **zsh** shell user:

    ```bash
    ./setup-shortcuts zsh; source ~/.zshrc
    ```

Also you can use gpu via optiprime adding `gpu` param like this:

```bash
./setup-shortcuts zsh gpu; source ~/.zshrc
```

**Step 3:** Run agent in `defend the center` scenario.

```bash
agent-demo
```

## Use

#### Train
    
```bash
agent-train [--weights weights_file_path]
```

See evolution of train process running tensor board:

```bash
agent-metrics
```
After go to dash: [http://localhost:6006](http://localhost:6006)

**Note**: When the training process ends you will find best weights file under `reports` path.

#### Play

Play agent in `defend the center` scenario:
```bash
agent-play --config scenarios/defend_the_center/agent.yml \
           --weights scenarios/defend_the_center/weights/best_weights-loss_0.0208.h5
```

Play agent in `basic` scenario:
```bash
agent-play --config scenarios/basic/agent.yml \
           --weights scenarios/basic/weights.h5
```

#### Help

To see all available args:

```bash
agent-train/agent-play  --help
```


## Scenarios


#### Basic
The purpose of the scenario is just to check if using this
framework to train some AI i 3D environment is feasible.

Map is a rectangle with gray walls, ceiling and floor.
Player is spawned along the longer wall, in the center.
A red, circular monster is spawned randomly somewhere along
the opposite wall. Player can only (config) go left/right
and shoot. 1 hit is enough to kill the monster. Episode
finishes when monster is killed or on timeout.


#### Defend the center

The purpose of this scenario is to teach the agent that killing the
monsters is GOOD and when monsters kill you is BAD. In addition,
wasting amunition is not very good either. Agent is rewarded only
for killing monsters so he has to figure out the rest for himself.

Map is a large circle. Player is spawned in the exact center.
5 melee-only, monsters are spawned along the wall. Monsters are
killed after a single shot. After dying each monster is respawned
after some time. Episode ends when the player dies (it's inevitable
because of limitted ammo).
