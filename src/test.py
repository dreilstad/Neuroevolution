import random

from itertools import repeat


def mutPolynomialBounded(individual, eta, low, up, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: A value or a :term:`python:sequence` of values that
                is the lower bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that
               is the upper bound of the search space.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (xu - xl)
            x = min(max(x, xl), xu)
            individual[i] = x
    return individual,

def execute(self, solution: FloatSolution) -> FloatSolution:
    for i in range(solution.number_of_variables):
        rand = random.random()

        if rand <= self.probability:
            y = solution.variables[i]
            yl, yu = solution.lower_bound[i], solution.upper_bound[i]

            if yl == yu:
                y = yl
            else:
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                rnd = random.random()
                mut_pow = 1.0 / (self.distribution_index + 1.0)
                if rnd <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, self.distribution_index + 1.0))
                    deltaq = pow(val, mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, self.distribution_index + 1.0))
                    deltaq = 1.0 - pow(val, mut_pow)

                y += deltaq * (yu - yl)
                if y < solution.lower_bound[i]:
                    y = solution.lower_bound[i]
                if y > solution.upper_bound[i]:
                    y = solution.upper_bound[i]

            solution.variables[i] = y

    return solution