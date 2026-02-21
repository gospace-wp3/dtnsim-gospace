import random

class RLAgent:

    def __init__(self):

        # Rate actions
        self.rates = [1e6, 2e6, 5e6]

        # Drop actions (number of bundles)
        self.drop_levels = [0, 1, 2, 3, 4, 5, 10]

        # Combined actions
        self.actions = []
        for r in self.rates:
            for d in self.drop_levels:
                self.actions.append((r, d))

        self.q_table = {}

        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.9

    def discretize(self, state):
        radio_bin = min(int(state["radio_queue"] // 100), 20)
        dep_bin = min(int(state["departure_rate"] // 1e6), 5)
        energy_bin = min(int(state["energy_spent"] // 0.5), 10)
        return (radio_bin, dep_bin, energy_bin)

    def act(self, state):
        s = self.discretize(state)

        if s not in self.q_table:
            self.q_table[s] = {a: 0.0 for a in self.actions}

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        return max(self.q_table[s], key=self.q_table[s].get)

    def learn(self, state, action, reward, next_state):
        s  = self.discretize(state)
        s2 = self.discretize(next_state)

        if s not in self.q_table:
            self.q_table[s] = {a: 0.0 for a in self.actions}
        if s2 not in self.q_table:
            self.q_table[s2] = {a: 0.0 for a in self.actions}

        best_next = max(self.q_table[s2].values())

        self.q_table[s][action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[s][action]
        )
