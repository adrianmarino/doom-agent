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
agent-play --weights weights_file
```

* Show demo:

```bash
agent-demo
```


* To view all options:

```bash
agent-train --help
```

or

```bash
agent-play --help
```


## Scenarios(environments)


### Basic
The purpose of the scenario is just to check if using this
framework to train some AI i 3D environment is feasible.

Map is a rectangle with gray walls, ceiling and floor.
Player is spawned along the longer wall, in the center.
A red, circular monster is spawned randomly somewhere along
the opposite wall. Player can only (config) go left/right
and shoot. 1 hit is enough to kill the monster. Episode
finishes when monster is killed or on timeout.


### Deadly Corridor

The purpose of this scenario is to teach the agent to navigate towards
his fundamental goal (the vest) and make sure he survives at the
same time.

Map is a corridor with shooting monsters on both sides (6 monsters
in total). A green vest is placed at the oposite end of the corridor.
Reward is proportional (negative or positive) to change of the
distance between the player and the vest. If player ignores monsters
on the sides and runs str## DEADLY CORRIDOR
The purpose of this scenario is to teach the agent to navigate towards
his fundamental goal (the vest) and make sure he survives at the
same time.
aight for the vest he will be killed somewhere
along the way. To ensure this behavior doom_skill = 5 (config) is
needed.


### Deffend the center

The purpose of this scenario is to teach the agent that killing the
monsters is GOOD and when monsters kill you is BAD. In addition,
wasting amunition is not very good either. Agent is rewarded only
for killing monsters so he has to figure out the rest for himself.

Map is a large circle. Player is spawned in the exact center.
5 melee-only, monsters are spawned along the wall. Monsters are
killed after a single shot. After dying each monster is respawned
after some time. Episode ends when the player dies (it's inevitable
because of limitted ammo).


### My way home

The purpose of this scenario is to teach the agent how to navigate
in a labirynth-like surroundings and reach his ultimate goal
(and learn what it actually is).

Map is a series of rooms with interconnection and 1 corridor
with a dead end. Each room has a different color. There is a
green vest in one of the rooms (the same room every time).
Player is spawned in randomly choosen room facing a random
direction. Episode ends when vest is reached or on timeout/


### Health Gathering

The purpose of this scenario is to teach the agent how to survive
without knowing what makes him survive. Agent know only that life
is precious and death is bad so he must learn what prolongs his
existence and that his health is connected with it.

Map is a rectangle with green, acidic floor which hurts the player
periodically. Initially there are some medkits spread uniformly
over the map. A new medkit falls from the skies every now and then.
Medkits heal some portions of player's health - to survive agent
needs to pick them up. Episode finishes after player's death or
on timeout.
