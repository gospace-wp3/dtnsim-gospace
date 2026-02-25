import random

class RLAgent:

    def __init__(self):

        # Rate actions
        self.rates = [2e6, 2e6, 2e6]

        # Drop actions (number of bundles)
        self.drop_levels = [0,1,2,3,5]

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
        
        dep_bin = min(int(state["departure_rate"] // 1e6), 5)
        energy_bin = min(int(state["energy_spent"] // 0.5), 10)
        pressure = state["pressure"]
        if pressure < 0.05:
            pressure_bin = 0
        elif pressure < 0.15:
            pressure_bin = 1
        elif pressure < 0.30:
            pressure_bin = 2
        else:
            pressure_bin = 3
    
    # Drops (0–3)
        d = state["dropped"]
        if d == 0:
            drop_bin = 0
        elif d <= 5:
            drop_bin = 1
        elif d <= 20:
            drop_bin = 2
        else:
            drop_bin = 3
        return (energy_bin, dep_bin, pressure_bin,drop_bin)

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
