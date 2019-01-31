# UROP
This is the work for UROP Proposal and Exercises for LIS, CSAIL, MIT

### Requirements
python 2.7.12\
opencv-python (3.4.5.20)\
numpy (1.16.0)

## 2.1 Planning
### 1. MDP and Value Iteration
* The `grid_world.py` implements a 2-dimensional grid-world MDP environment.
* The agent (grey circle) has four actions: move up, right, down, and left.
* View the position of the agent as the state.
* It random generates the positions of a gold (yellow grid with reward = 1), hells (red grids with reward = -1), and obstacles (black grids). The agent will start at (0,0) position, which is the top left of the map.
* To run this program, simply run `$python grid_world.py`

* **Member functions:**
  * value_iter: Performs value iteration for the given grid map.
  * state_transition: Outputs the next state given the current state and action.
  * move_agent: Moves the agent in the map.
  * render_map: Render the environment with cv2.
