#!/usr/bin/env python
# -*- encoding: UTF-8 -*-
import cv2
import numpy as np


class GridMap:

    def __init__(self, n_raw, n_col, grid_size, agent_raw, agent_col, n_hell, n_obs, discount):
        self.agent_raw = agent_raw  # initial position of agent
        self.agent_col = agent_col  # initial position of agent
        self.actions = [0, 1, 2, 3]  # [up, right, down, left]
        self.n_actions = len(self.actions)
        self.n_raw = n_raw
        self.n_col = n_col
        self.grid_size = grid_size
        self.n_hell = n_hell
        self.n_obs = n_obs
        self.gamma = discount  # discount factor

        self.n_gold = 1
        self.__reward_val = 1
        self.__trans_prob = 0.8  # transition probability

        # random generate gold, hell, obstacles, exclude (0, 0)
        __grid_idx = np.random.choice(np.arange(1, self.n_raw * self.n_col),
                                      size=(self.n_gold + self.n_hell + self.n_obs), replace=False)

        print 'grid_idx = ', __grid_idx

        self.gold_raw, self.gold_col = np.unravel_index(indices=np.max(__grid_idx),
                                                        dims=(self.n_raw, self.n_col))
        __grid_idx = np.delete(arr=__grid_idx, obj=np.argmax(__grid_idx))

        self.hell_raw, self.hell_col = np.unravel_index(indices=__grid_idx[:self.n_hell],
                                                        dims=(self.n_raw, self.n_col))

        self.obs_raw, self.obs_col = np.unravel_index(indices=__grid_idx[self.n_hell:self.n_hell + self.n_obs],
                                                      dims=(self.n_raw, self.n_col))

        # Set reward array
        self.reward = np.zeros((self.n_raw, self.n_col))
        self.reward[self.gold_raw, self.gold_col] = self.__reward_val
        self.reward[self.hell_raw, self.hell_col] = -self.__reward_val

        # Set value array
        self.value = np.copy(self.reward)

        if n_obs == 1:
            self.value[self.obs_raw, self.obs_col] = np.NINF

        else:
            for i in range(n_obs):
                self.value[self.obs_raw[i], self.obs_col[i]] = np.NINF

        print 'gold is at ', self.gold_raw, self.gold_col

        return

    def value_iter(self):
        """
        current state: [self.agent_raw, self.agent_col]
        update self.value iteratively
        """

        while True:
            __prev = np.copy(self.value)

            for stat_idx in range(N_RAW * N_COL):  # for each state
                raw, col = np.unravel_index(indices=stat_idx, dims=(self.n_raw, self.n_col))

                # skip if state is obstacle
                if np.isin(stat_idx, np.ravel_multi_index((self.obs_raw, self.obs_col), (self.n_raw, self.n_col))):
                    continue

                if np.isin(stat_idx, np.ravel_multi_index((self.gold_raw, self.gold_col), (self.n_raw, self.n_col))):
                    continue

                if np.isin(stat_idx, np.ravel_multi_index((self.hell_raw, self.hell_col), (self.n_raw, self.n_col))):
                    continue

                __best_expect_value = np.NINF

                for __expect_action in self.actions:
                    __expect_value = 0

                    for __real_action in self.actions:
                        if __real_action == __expect_action:
                            __prob = 0.8
                        elif abs(__real_action - __expect_action) == 2:
                            __prob = 0.0
                        else:
                            __prob = 0.1
                        __expect_value += __prev[self.state_transition(raw, col, __real_action)] * __prob

                    __best_expect_value = max(__expect_value, __best_expect_value)

                self.value[raw, col] = np.around(self.reward[raw, col] + self.gamma * __best_expect_value, decimals=3)

            # break the while loop
            if np.all(np.equal(self.value, __prev)):
                break

        return

    def state_transition(self, cur_raw, cur_col, action):
        """
        :param cur_raw: raw of agent (current state)
        :param cur_col: column of agent (current state)
        :param action: step for agent to do
        :return: next_raw, next_col (next state)
        """
        if action == 0 and cur_raw > 0 \
                and not np.isneginf(self.value[cur_raw - 1, cur_col]):
            return cur_raw - 1, cur_col  # move up

        elif action == 1 and cur_col < self.n_col - 1 \
                and not np.isneginf(self.value[cur_raw, cur_col + 1]):
            return cur_raw, cur_col + 1  # move right

        elif action == 2 and cur_raw < self.n_raw - 1\
                and not np.isneginf(self.value[cur_raw + 1, cur_col]):
            return cur_raw + 1, cur_col  # move down

        elif action == 3 and cur_col > 0 \
                and not np.isneginf(self.value[cur_raw, cur_col - 1]):
            return cur_raw, cur_col - 1  # move left

        else:
            return cur_raw, cur_col  # stay

    def move_agent(self, action):  # this is for render
        if action == 0 and self.agent_raw > 0 \
                and not np.isneginf(self.value[self.agent_raw - 1, self.agent_col]):
            self.agent_raw -= 1  # move up

        elif action == 1 and self.agent_col < self.n_col - 1 \
                and not np.isneginf(self.value[self.agent_raw, self.agent_col + 1]):
            self.agent_col += 1  # move right

        elif action == 2 and self.agent_raw < self.n_raw - 1\
                and not np.isneginf(self.value[self.agent_raw + 1, self.agent_col]):
            self.agent_raw +=\
                1  # move down

        elif action == 3 and self.agent_col > 0 \
                and not np.isneginf(self.value[self.agent_raw, self.agent_col - 1]):
            self.agent_col -= 1  # move left

        else:
            pass
        return

    def render_map(self):
        img_width = self.n_col * self.grid_size
        img_height = N_RAW * self.grid_size
        img = np.ones((img_height, img_width, 3), np.uint8) * 255

        # Draw black lines as grids
        for idx in range(self.n_col):  # vertical line, BGR order
            cv2.line(img, ((idx+1) * self.grid_size, 0), ((idx+1) * self.grid_size, img_height), (0, 0, 0), 1)
        for idx in range(self.n_raw):  # horizontal line, BGR order
            cv2.line(img, (0, (idx+1) * self.grid_size), (img_width, (idx+1) * self.grid_size), (0, 0, 0), 1)

        # Draw gold
        cv2.rectangle(img, (self.gold_col * self.grid_size, self.gold_raw * self.grid_size),
                      ((self.gold_col + 1) * self.grid_size, (self.gold_raw + 1) * self.grid_size),
                      (0, 250, 250), -1)

        # Draw hell
        for idx in range(self.n_hell):
            cv2.rectangle(img, (self.hell_col[idx] * self.grid_size, self.hell_raw[idx] * self.grid_size),
                          ((self.hell_col[idx] + 1) * self.grid_size, ((self.hell_raw[idx] + 1) * self.grid_size)),
                          (0, 0, 255), -1)

        # Draw obstacles
        for idx in range(self.n_obs):
            cv2.rectangle(img, (self.obs_col[idx] * self.grid_size, self.obs_raw[idx] * self.grid_size),
                          ((self.obs_col[idx] + 1) * self.grid_size, (self.obs_raw[idx] + 1) * self.grid_size),
                          (0, 0, 0), -1)

        # Draw agent
        cv2.circle(img, (self.agent_col * self.grid_size + self.grid_size // 2,
                         self.agent_raw * self.grid_size + self.grid_size // 2),
                   self.grid_size // 4, (100, 100, 100), -1)

        cv2.imshow('temp', img)
        cv2.waitKey(500)

        return


if __name__ == '__main__':
    N_RAW = 10   # numbers of grid per raw
    N_COL = 10   # numbers of grid per column
    GRID_SIZE = 50  # pixel per length

    AGENT_RAW = 0  # initial position for agent
    AGENT_COL = 0  # initial position for agent

    N_HELL = 5  # number of hell
    N_OBS = 20   # numbers of obstacles

    DISCOUNT = 0.9

    temp = GridMap(N_RAW, N_COL, GRID_SIZE, AGENT_RAW, AGENT_COL, N_HELL, N_OBS, DISCOUNT)
    temp.render_map()

    temp.value_iter()
    print 'Value function'
    print temp.value

    while True:
        agent_val = temp.value[temp.agent_raw, temp.agent_col]

        exp_vals = []
        for act in temp.actions:
            exp_vals.append(temp.value[temp.state_transition(temp.agent_raw, temp.agent_col, act)])
        exp_vals = np.array(exp_vals)

        if agent_val < np.max(exp_vals):
            temp.move_agent(np.argmax(exp_vals))
            temp.render_map()

        else:
            if temp.agent_raw == temp.gold_raw and temp.agent_col == temp.gold_col:
                print 'Reach gold!'
            else:
                print 'Infeasible map'
            break

    temp.render_map()
    cv2.destroyAllWindows()
