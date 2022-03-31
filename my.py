import numpy as np
import time


class CartPoleEnv:
    """_summary_"""

    def __init__(self):
        """_summary_"""
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4

        self.bins = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ]
        )

    def reset(self):
        """_summary_"""

        self.state = np.random.uniform(-0.05, 0.05, 4)
        return self.state

    def step(self, action):
        """_summary_

        Args:
            action (_type_): _description_
        """

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        else:
            reward = 0.0

        return self.state, reward, done


class Qagent:
    """_summary_"""

    def __init__(self, env):
        self.env = env
        self.qtable = dict()
        self.data = {"max": [0], "avg": [0]}

    def writing(self, state):
        """_summary_"""

        index = tuple()
        for i in range(len(self.env.bins)):
            index += np.digitize(state[i], self.env.bins[i])
        return index

    def get_action(self, index, epsilon):
        """_summary_

        Args:
            index (_type_): _description_
            epsilon (_type_): _description_
        """

        q_table = self.qtable[index]
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice([0, 1])
        else:
            action = q_table.index(np.max(q_table))
        return action

    def qlearning(self, episodes=5000, gamma=0.95, lr=0.1, timestep=5000, epsilon=1):
        """_summary_

        Args:
            episodes (int, optional): _description_. Defaults to 5000.
            gamma (float, optional): _description_. Defaults to 0.95.
            lr (float, optional): _description_. Defaults to 0.1.
            timestep (int, optional): _description_. Defaults to 5000.
            epsilon (int, optional): _description_. Defaults to 1.
        """

        for episode in range(1, episodes + 1):
            state = self.env.reset()  # initial observation
            steps = 0
            score = 0
            done = False
            epsilon = 1
            while not done:
                steps += 1
                ep_start = time.time()
                action = self.get_action(self.writing(state), epsilon)
                observation, reward, done = self.env.step(action)
                next_state = self.writing(observation)
                scrore += reward

                if not done:
                    max_future_q = np.max(self.qtable[next_state])
                    current_q = self.qtable[current_state + action]
                    new_q = (1 - lr) * current_q + lr * (reward + gamma * max_future_q)
                    self.qtable[current_state + action] = new_q

                current_state = next_state

                epsilon = epsilon - 0.01

            # End of the loop update
            else:
                rewards += score
                runs.append(score)
                if (
                    score > 195 and steps >= 100 and solved == False
                ):  # considered as a solved:
                    solved = True
                    print(
                        "Solved in episode : {} in time {}".format(
                            episode, (time.time() - ep_start)
                        )
                    )

            # Timestep value update
            if episode % timestep == 0:
                print(
                    "Episode : {} | Reward -> {} | Max reward : {} | Time : {}".format(
                        episode, rewards / timestep, max(runs), time.time() - ep_start
                    )
                )
                self.data["max"].append(max(runs))
                self.data["avg"].append(rewards / timestep)
                if rewards / timestep >= 195:
                    print("Solved in episode : {}".format(episode))
                rewards, runs = 0, [0]
