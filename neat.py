# neat.py
import random
import numpy as np
import copy
import pickle
import os
from collections import defaultdict

class NodeGene:
    def __init__(self, node_id, node_type, activation='sigmoid'):
        self.id = node_id
        self.type = node_type
        self.activation = activation
        self.value = 0.0
        
    def activate(self, x):
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-4.9 * x))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return max(0, x)
        else:
            return x

class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled=True, innovation=0):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

class Genome:
    def __init__(self):
        self.nodes = {}
        self.connections = {}
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        
    def add_node(self, node):
        self.nodes[node.id] = node
        
    def add_connection(self, conn):
        key = (conn.in_node, conn.out_node)
        self.connections[key] = conn
        
    def get_connection(self, in_node, out_node):
        return self.connections.get((in_node, out_node), None)
        
    def feed_forward(self, inputs):
        for node in self.nodes.values():
            node.value = 0.0
            
        input_nodes = [node for node in self.nodes.values() if node.type == 'input']
        input_nodes.sort(key=lambda x: x.id)
        
        for i, node in enumerate(input_nodes):
            if i < len(inputs):
                node.value = inputs[i]
                
        sorted_nodes = sorted(self.nodes.values(), 
                             key=lambda x: (0 if x.type == 'input' else 1 if x.type == 'hidden' else 2, x.id))
        
        for node in sorted_nodes:
            if node.type == 'input':
                continue
                
            total_input = 0.0
            for conn in self.connections.values():
                if conn.out_node == node.id and conn.enabled:
                    in_node = self.nodes[conn.in_node]
                    total_input += in_node.value * conn.weight
                    
            node.value = node.activate(total_input)
            
        output_nodes = [node for node in self.nodes.values() if node.type == 'output']
        output_nodes.sort(key=lambda x: x.id)
        return [node.value for node in output_nodes]
        
    def mutate_add_connection(self, innovation_counter, max_attempts=20):
        nodes = list(self.nodes.values())
        
        for _ in range(max_attempts):
            node1 = random.choice(nodes)
            node2 = random.choice(nodes)
            
            if (node1.type != 'output' and node2.type != 'input' and 
                node1.id != node2.id and 
                (node1.id, node2.id) not in self.connections):
                
                if not self.would_create_cycle(node1.id, node2.id):
                    weight = random.uniform(-2, 2)
                    innovation = next(innovation_counter)
                    conn = ConnectionGene(node1.id, node2.id, weight, True, innovation)
                    self.add_connection(conn)
                    return True
                    
        return False
        
    def mutate_add_node(self, innovation_counter, node_counter):
        if not self.connections:
            return False
            
        enabled_conns = [conn for conn in self.connections.values() if conn.enabled]
        if not enabled_conns:
            return False
            
        conn_to_split = random.choice(enabled_conns)
        conn_to_split.enabled = False
        
        new_node_id = next(node_counter)
        new_node = NodeGene(new_node_id, 'hidden')
        self.add_node(new_node)
        
        innov1 = next(innovation_counter)
        conn1 = ConnectionGene(conn_to_split.in_node, new_node_id, 1.0, True, innov1)
        
        innov2 = next(innovation_counter)
        conn2 = ConnectionGene(new_node_id, conn_to_split.out_node, conn_to_split.weight, True, innov2)
        
        self.add_connection(conn1)
        self.add_connection(conn2)
        
        return True
        
    def would_create_cycle(self, from_node, to_node):
        visited = set()
        
        def has_path(start, target):
            if start == target:
                return True
            visited.add(start)
            for conn in self.connections.values():
                if conn.in_node == start and conn.enabled and conn.out_node not in visited:
                    if has_path(conn.out_node, target):
                        return True
            return False
            
        return has_path(to_node, from_node)
        
    def mutate(self, innovation_counter, node_counter):
        
        if random.random() < 0.05:
            self.mutate_add_connection(innovation_counter)
              
        if random.random() < 0.03:
            self.mutate_add_node(innovation_counter, node_counter)
            
        for conn in self.connections.values():
            if random.random() < 0.02:
                conn.enabled = not conn.enabled
        
    def crossover(self, other):
        child = Genome()
        
        if self.fitness > other.fitness:
            fitter_parent, other_parent = self, other
        else:
            fitter_parent, other_parent = other, self
            
        for node_id, node in fitter_parent.nodes.items():
            child.add_node(copy.copy(node))
            
        all_innovations = set()
        for conn in fitter_parent.connections.values():
            all_innovations.add(conn.innovation)
        for conn in other_parent.connections.values():
            all_innovations.add(conn.innovation)
            
        for innov in sorted(all_innovations):
            conn1 = None
            conn2 = None
            
            for conn in fitter_parent.connections.values():
                if conn.innovation == innov:
                    conn1 = conn
                    break
            for conn in other_parent.connections.values():
                if conn.innovation == innov:
                    conn2 = conn
                    break
                    
            if conn1 and conn2:
                if random.random() < 0.5:
                    inherited_conn = copy.copy(conn1)
                else:
                    inherited_conn = copy.copy(conn2)
                    
                if (not conn1.enabled or not conn2.enabled) and random.random() < 0.75:
                    inherited_conn.enabled = False
                    
            elif conn1:
                inherited_conn = copy.copy(conn1)
            else:
                inherited_conn = copy.copy(conn2)
                
            child.add_connection(inherited_conn)
            
        return child
        
    def distance(self, other, c1=1.0, c2=1.0, c3=0.4):
        innovations1 = set(conn.innovation for conn in self.connections.values())
        innovations2 = set(conn.innovation for conn in other.connections.values())
        
        matching = innovations1.intersection(innovations2)
        disjoint = innovations1.symmetric_difference(innovations2)
        
        weight_diff = 0.0
        for innov in matching:
            conn1 = next(conn for conn in self.connections.values() if conn.innovation == innov)
            conn2 = next(conn for conn in other.connections.values() if conn.innovation == innov)
            weight_diff += abs(conn1.weight - conn2.weight)
            
        n = max(len(innovations1), len(innovations2))
        if n == 0:
            n = 1
            
        return (c1 * len(disjoint) / n + 
                c2 * len(disjoint) / n + 
                c3 * weight_diff / len(matching)) if matching else 1.0

class Species:
    def __init__(self, representative):
        self.genomes = [representative]
        self.representative = representative
        self.best_fitness = representative.fitness
        self.staleness = 0
        
    def add_genome(self, genome):
        self.genomes.append(genome)
        if genome.fitness > self.best_fitness:
            self.best_fitness = genome.fitness
            self.staleness = 0
        else:
            self.staleness += 1
            
    def adjust_fitnesses(self):
        for genome in self.genomes:
            genome.adjusted_fitness = genome.fitness / len(self.genomes)
            
    def breed_child(self, innovation_counter, node_counter):
        if len(self.genomes) == 0:
            return None
            
        if random.random() < 0.25 or len(self.genomes) == 1:
            parent = random.choice(self.genomes)
            child = copy.deepcopy(parent)
        else:
            parent1 = random.choice(self.genomes)
            parent2 = random.choice(self.genomes)
            child = parent1.crossover(parent2)
            
        child.mutate(innovation_counter, node_counter)
        return child
        
    def cull(self, survival_rate=0.2):
        self.genomes.sort(key=lambda x: x.fitness, reverse=True)
        survivors_count = max(1, int(len(self.genomes) * survival_rate))
        self.genomes = self.genomes[:survivors_count]

class NEAT:
    def __init__(self, input_size, output_size, population_size=100):
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        self.population = []
        self.species = []
        self.generation = 0
        self.best_fitness = 0
        self.best_genome = None
        self.best_genome_overall = None
        
        self.node_counter = self._counter(1000)
        self.innovation_counter = self._counter(10000)
        
        self.speciation_threshold = 3.0
        self.stale_species = 15
        
        self._create_initial_population()
        
    def _counter(self, start=0):
        i = start
        while True:
            yield i
            i += 1
            
    def _create_initial_population(self):
        for _ in range(self.population_size):
            genome = Genome()
            
            for i in range(self.input_size):
                node_id = i
                genome.add_node(NodeGene(node_id, 'input'))
                
            for i in range(self.output_size):
                node_id = self.input_size + i
                genome.add_node(NodeGene(node_id, 'output', 'sigmoid'))
                
            for i in range(self.input_size):
                for j in range(self.output_size):
                    out_node = self.input_size + j
                    weight = random.uniform(-2, 2)
                    innovation = next(self.innovation_counter)
                    conn = ConnectionGene(i, out_node, weight, True, innovation)
                    genome.add_connection(conn)
                    
            self.population.append(genome)
            
    def speciate(self):
        for species in self.species:
            species.genomes = []
            
        for genome in self.population:
            found_species = False
            for species in self.species:
                distance = genome.distance(species.representative)
                if distance < self.speciation_threshold:
                    species.add_genome(genome)
                    found_species = True
                    break
                    
            if not found_species:
                self.species.append(Species(genome))
                
        self.species = [s for s in self.species if len(s.genomes) > 0]
        
    def breed_new_generation(self):
        total_adj_fitness = 0
        for species in self.species:
            species.adjust_fitnesses()
            species.cull()
            total_adj_fitness += sum(g.adjusted_fitness for g in species.genomes)
            
        children = []
        if self.best_genome_overall:
            children.append(copy.deepcopy(self.best_genome_overall))
            
        for species in self.species:
            if total_adj_fitness > 0:
                breed_count = int(len(self.population) * (sum(g.adjusted_fitness for g in species.genomes) / total_adj_fitness))
            else:
                breed_count = 1
                
            if species.genomes:
                best_in_species = max(species.genomes, key=lambda x: x.fitness)
                if len(children) < self.population_size:
                    children.append(copy.deepcopy(best_in_species))
                breed_count -= 1
                
            for _ in range(breed_count):
                child = species.breed_child(self.innovation_counter, self.node_counter)
                if child and len(children) < self.population_size:
                    children.append(child)
                    
        while len(children) < self.population_size:
            species = random.choice(self.species)
            child = species.breed_child(self.innovation_counter, self.node_counter)
            if child:
                children.append(child)
                
        self.population = children
        self.generation += 1
        
    def run_generation(self, fitness_function):
        current_gen_best_fitness = 0
        current_best_genome = None
        
        for genome in self.population:
            genome.fitness = fitness_function(genome)
            
            if genome.fitness > current_gen_best_fitness:
                current_gen_best_fitness = genome.fitness
                current_best_genome = genome
            
        if current_best_genome and current_gen_best_fitness > self.best_fitness:
            self.best_fitness = current_gen_best_fitness
            self.best_genome = copy.deepcopy(current_best_genome)
            self.best_genome_overall = copy.deepcopy(current_best_genome)
            
        self.speciate()
        self.breed_new_generation()
        
        self.species = [s for s in self.species if s.staleness < self.stale_species]
        
        return current_gen_best_fitness
        
    def save_best(self, filename):
        if self.best_genome_overall:
            with open(filename, 'wb') as f:
                pickle.dump(self.best_genome_overall, f)
                
    def load_best(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.best_genome_overall = pickle.load(f)
                self.best_genome = self.best_genome_overall
                self.best_fitness = self.best_genome_overall.fitness
            return True
        return False