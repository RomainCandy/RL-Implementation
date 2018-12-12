import random
from enum import Enum
from typing import List, Dict
import copy
from itertools import accumulate
# import cProfile

# http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

MUTATION_RATE = .5
ADD_CONNECTION_RATE = .1
ADD_NODE_RATE = .1


class Counter:
    def __init__(self):
        self.counter = 0

    def get_innovation(self):
        temp = self.counter
        self.counter += 1
        return temp


class Type(Enum):
    input = 0
    hidden = 1
    output = 2


class NodeGenes:
    def __init__(self, node_type: Type, idx: int):
        self.node_type = node_type
        self.id = idx

    def __eq__(self, other):
        assert isinstance(other, NodeGenes)
        res = self.id == other.id
        return res

    def __str__(self):
        return f"Node: {self.id} {self.node_type}"

    def __repr__(self):
        return str(self)


class ConnectionGenes:

    def __init__(self, in_node: NodeGenes, out_node: NodeGenes, weight: float,
                 enable: bool, innovation_number: int):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enable = enable
        self.innovation_number = innovation_number

    def disable(self):
        self.enable = False

    def __eq__(self, other):
        assert isinstance(other, ConnectionGenes)
        return (self.in_node == other.in_node and self.out_node == other.out_node) or (
                self.in_node == other.out_node and self.out_node == other.in_node
        )

    def __str__(self):
        return f"in_node: {self.in_node}, out_node: {self.out_node}," \
               f" weight: {self.weight :.2f}, enable: {self.enable}, in_num: {self.innovation_number}"

    def __repr__(self):
        return str(self)


class Genome:
    def __init__(self, nodes: List[NodeGenes], connections: Dict[int, ConnectionGenes]):
        self.nodes = nodes
        self.connections = connections

    def add_connection_mutation(self, counter: Counter):
        node1 = random.choice(self.nodes)
        if node1.node_type == Type.input:
            node2 = random.choice([x for x in self.nodes if x.node_type != Type.input])
        elif node1.node_type == Type.output:
            node2 = random.choice([x for x in self.nodes if x.node_type != Type.output])
        else:
            node2 = random.choice(self.nodes)

        if node1.node_type == Type.output or node1.node_type == Type.hidden and node2.node_type == Type.input:
            node1, node2 = node2, node1
        innovation = counter.get_innovation()
        new_connection = ConnectionGenes(node1, node2, 2 * random.random() - 1, True, innovation)
        if new_connection not in self.connections.values():
            self.connections[new_connection.innovation_number] = new_connection
        else:
            counter.counter -= 1

    def add_node_mutation(self, counter: Counter):
        if not self.connections:
            raise AttributeError("there is no connection")
        split_con = random.choice(list(self.connections.values()))
        in_node = split_con.in_node
        out_node = split_con.out_node
        weight = split_con.weight
        split_con.disable()
        new_node = NodeGenes(Type.hidden, len(self.nodes))
        self.nodes.append(new_node)
        innovation = counter.get_innovation()
        con1 = ConnectionGenes(in_node, new_node, 1, True, innovation)
        innovation = counter.get_innovation()
        con2 = ConnectionGenes(new_node, out_node, weight, True, innovation)
        self.connections[con1.innovation_number] = con1
        self.connections[con2.innovation_number] = con2

    def mutation(self):
        for con in self.connections.values():
            if random.random() < .8:
                con.weight *= 4 * random.random() - 2
            else:
                con.weight = 4 * random.random() - 2

    @staticmethod
    def crossover(fittest_genome, genome2):
        nodes_fittest = copy.copy(fittest_genome.nodes)
        child = Genome(nodes_fittest, {})
        for innovation_fittest, con_fittest in fittest_genome.connections.items():
            if innovation_fittest in genome2.connections:
                r = random.random()
                if r < .5:
                    child.connections[innovation_fittest] = copy.copy(con_fittest)
                else:
                    child.connections[innovation_fittest] = copy.copy(genome2.connections[innovation_fittest])
                # matching gene
            else:
                child.connections[innovation_fittest] = copy.copy(con_fittest)
                # not matching gene
        return child

    def __hash__(self):
        return hash(str(self.connections) + str(self.nodes))

    def __str__(self):
        return "Nodes: " + str(self.nodes) + "\nConnections: " + str(self.connections)

    def __repr__(self):
        return str(self)


def measure_compatibility_genome(genome1: Genome, genome2: Genome, c1: float, c2: float, c3: float):
    max_inno_gen1 = max(genome1.connections)
    max_inno_gen2 = max(genome2.connections)
    # N = max(len(genome2.connections), len(genome1.connections))
    N = 1
    matching = 0
    weighted_diff = 0
    disjoint = 0
    excess = 0
    max_idx = max(max_inno_gen1, max_inno_gen2)
    min_idx = min(max_inno_gen1, max_inno_gen2)
    for idx in range(max_idx + 1):
        if idx <= min_idx:
            if idx in genome1.connections:
                if idx in genome2.connections:
                    matching += 1
                    weighted_diff += abs(genome1.connections[idx].weight - genome2.connections[idx].weight)
                else:
                    disjoint += 1
            elif idx in genome2.connections:
                disjoint += 1
        else:
            if idx in genome1.connections or idx in genome2.connections:
                excess += 1
    # print("excess ", excess, "disjoint", disjoint, "matching", matching, "weighted: ", weighted_diff)
    return (c1 * excess + c2 * disjoint) / N + c3 * weighted_diff / matching


class FitnessGenome:
    def __init__(self, genome: Genome, fitness: float):
        self.genome = genome
        self.fitness = fitness

    def __gt__(self, other):
        assert isinstance(other, FitnessGenome)
        return self.fitness > other.fitness


class Species:
    def __init__(self, mascot: Genome):
        self.mascot = mascot
        self.members = [mascot]
        self.fitness_pop = []
        self.overall_adjusted_fitness = 0

    def add_adjusted_fitness(self, adjusted_fitness):
        self.overall_adjusted_fitness += adjusted_fitness

    def reset(self):
        self.mascot = random.choice(self.members)
        self.members.clear()
        self.fitness_pop.clear()
        self.overall_adjusted_fitness = 0


class Evaluator:
    def __init__(self, pop_size: int, starting_genome: Genome,
                 node_innovation: Counter, connection_innovation: Counter):
        self.pop_size = pop_size
        self.node_innovation = node_innovation
        self.connection_innovation = connection_innovation
        self.map_species = {}
        self.score_map = {}
        self.genomes = [starting_genome for _ in range(pop_size)]
        self.species = []
        self.next_generation = []
        self.highest_score = float("-infinity")
        self.best_genome = None

    def evaluate(self):
        """
        place genome into species
        evaluate genome and assign fitness
        keep best genome from each species
        breed the rest of genomes
        :return: void
        """
        for sp in self.species:
            sp.reset()
        self.score_map.clear()
        self.next_generation.clear()
        self.map_species.clear()

        # 1)
        for genome in self.genomes:
            for sp in self.species:
                if measure_compatibility_genome(genome, sp.mascot, 1, 1, .4) < 10:
                    self.map_species[genome] = sp
                    sp.members.append(genome)
                    break
            else:
                new_sp = Species(genome)
                self.map_species[genome] = new_sp
                self.species.append(new_sp)
        # clear dead species
        self.species = [s for s in self.species if s.members]

        # 2)
        for genome in self.genomes:
            sp = self.map_species[genome]
            score = self.evaluate_genome(genome)
            if score >= self.highest_score:
                self.highest_score = score
                self.best_genome = genome
            adjusted_score = score / len(sp.members)
            sp.add_adjusted_fitness(adjusted_score)
            sp.fitness_pop.append(FitnessGenome(genome, adjusted_score))
            self.score_map[genome] = adjusted_score

        # 3)
        for sp in self.species:
            best_sp = max(sp.fitness_pop)
            self.next_generation.append(best_sp.genome)

        # 4)
        k = len(self.genomes) - len(self.species)
        cum_weights_sp = list(accumulate((s.overall_adjusted_fitness for s in self.species), lambda x, y: x+y))
        species = random.choices(self.species, cum_weights=cum_weights_sp, k=k)
        for sp in species:
            # cum_weights = list(accumulate(sp.fitness_pop, lambda x, y: x.fitness + y.fitness))
            cum_weights = list()
            old_val = 0
            for fit in sp.fitness_pop:
                cum_weights.append(old_val + fit.fitness)
                old_val += fit.fitness
            gen1, gen2 = random.choices(sp.fitness_pop, cum_weights=cum_weights, k=2)
            gen1 = gen1.genome
            gen2 = gen2.genome
            if self.score_map[gen1] > self.score_map[gen2]:
                child = Genome.crossover(gen1, gen2)
            else:
                child = Genome.crossover(gen2, gen1)
            if random.random() < MUTATION_RATE:
                child.mutation()
            if random.random() < ADD_CONNECTION_RATE:
                child.add_connection_mutation(self.connection_innovation)
            if random.random() < ADD_NODE_RATE:
                child.add_node_mutation(self.node_innovation)
            self.next_generation.append(child)

        self.genomes, self.next_generation = self.next_generation, []

    def evaluate_genome(self, genome: Genome):
        raise NotImplementedError


class TestEvaluator(Evaluator):
    def __init__(self, pop_size):
        starting_point = Genome([], {})
        node_counter = Counter()
        connection_counter = Counter()
        for _ in range(2):
            starting_point.nodes.append(NodeGenes(Type.input, node_counter.get_innovation()))
        starting_point.nodes.append(NodeGenes(Type.output, node_counter.get_innovation()))
        c1 = connection_counter.get_innovation()
        c2 = connection_counter.get_innovation()
        starting_point.connections[c1] = ConnectionGenes(starting_point.nodes[0],
                                                         starting_point.nodes[-1], .5, True, c1)
        starting_point.connections[c2] = ConnectionGenes(starting_point.nodes[1],
                                                         starting_point.nodes[-1], .5, True, c2)
        super(TestEvaluator, self).__init__(pop_size, starting_point, node_counter, connection_counter)

    def evaluate_genome(self, genome, pr=False):
        weight_sum = sum((abs(x.weight) for x in genome.connections.values() if x.enable))
        if pr:
            print("WEIGHTED SUM: ", weight_sum)
        return 1000 / (abs(100 - weight_sum))
        # return len(genome.connections)


t = TestEvaluator(100)
import time
tt = time.time()
for i in range(1, 101):
    t.evaluate()
    # for x in t.genomes:
    #     print(x)
    # print("="*100)
    # for x in t.species:
    #     print(x.mascot)
    print("GENERATION: ", i, "HIGHEST SCORE: ", t.highest_score, "NB OF SPECIES:   ", len(t.species))
t.evaluate_genome(t.best_genome, pr=True)
print(time.time() - tt)


# def main():
#     nodes = [NodeGenes(Type.input, i) for i in range(3)] + [NodeGenes(Type.output, i) for i in range(3, 5)]
#     connections1 = [ConnectionGenes(random.choice(tuple(x for x in nodes if x.node_type == Type.input)),
#                                     random.choice(tuple(x for x in nodes if x.node_type == Type.output)),
#                                     random.random(), True, i)for i in range(1, 6)]
#
#     g1 = copy.deepcopy(Genome(nodes, {x.innovation_number: x for x in connections1}))
#     g2 = Genome(nodes, {x.innovation_number: x for x in connections1})
#     g1.connections[8] = ConnectionGenes(nodes[0], nodes[1], 1.11, False, 8)
#     g2.connections[6] = ConnectionGenes(nodes[-1], nodes[-2], -1.11, False, 5)
#     g2.connections[7] = ConnectionGenes(nodes[-1], nodes[-2], -1.11, False, 6)
#     g2.connections[9] = ConnectionGenes(nodes[-1], nodes[-2], -1.11, False, 5)
#     g2.connections[10] = ConnectionGenes(nodes[-1], nodes[-2], -1.11, False, 6)
#     for x in g1.connections.values():
#         x.weight = random.random()
#         x.enable = False
#         # print(x.weight)
#     for x in g1.connections.items():
#         pass
#         # print(x)
#     # print('-'*100)
#     for x in g2.connections.items():
#         pass
#         # print(x)
#     # print("="*50)
#     child = Genome.crossover(g1, g2)
#     # for x in child.connections.items():
#         # print(x)
#     print(g1.connections.keys())
#     print('-'*50)
#     print(g2.connections.keys())
#     print("="*50)
#     print(measure_compatibility_genome(g1, g2, 1, 1, 1))
#     print(measure_compatibility_genome(g2, g1, 1, 1, 1))
#
#
# if __name__ == "__main__":
#     main()
