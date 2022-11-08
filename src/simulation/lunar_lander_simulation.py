from .simulator import Simulator

class LunarLanderSimulator(Simulator):

    def __init__(self, objectives):
        super().__init__(objectives)
        self.env = gym.make("LunarLandar-v2")

    def simulate(self, genome_id, genome, neural_network):

        observation = env.reset()

        step = 0
        data = []
        while 1:
            step += 1
            output = neural_network.activate(observation)
            action = np.argmax(output)

            observation, reward, done, info = env.step(action)

            if done:
                break