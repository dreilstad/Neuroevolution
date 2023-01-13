
class TartarusConfiguration:
    def __init__(self, layout, N, K, moves=80):
        self.N = N
        self.K = K
        self.layout = layout
        self.moves = moves

        # C1 = (N^2 - K) / (N-2)^2 ???
        self.C1 = 1.8

        # C2 = N^2 / K
        self.C2 = (self.N - 2)**2 / self.K

    def __getitem__(self, key):
        return self.layout[key]

    def update(self, layout):
        self.layout = layout

    def state_evaluation(self):
        """
        Compute state evaluation
        """

        sum_distance_to_edge = 0.0
        for i in range(self.N):
            for j in range(self.N):
                if self.layout[i][j]:
                    sum_distance_to_edge += self.min_dist_to_edge(i, j)
                    print(self.min_dist_to_edge(i, j))

        performance_score = self.C1 * (self.K - (2/self.N) * sum_distance_to_edge - self.C2)
        return performance_score

    def min_dist_to_edge(self, x, y):
        left_dist = ((y - 0)**2)**2
        right_dist = ((y - (self.N - 1))**2)**2
        up_dist = ((x - 0)**2)**2
        down_dist = ((x - (self.N - 1))**2)**2

        return min(left_dist, right_dist, up_dist, down_dist)