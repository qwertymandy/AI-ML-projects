import random
import math

def euclidean_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

def route_distance(route, city_coords):
    return sum(
        euclidean_distance(city_coords[route[i]], city_coords[route[(i + 1) % len(route)]])
        for i in range(len(route))
    )

def create_route(city_count):
    route = list(range(city_count))
    random.shuffle(route)
    return route

def initial_population(pop_size, city_count):
    return [create_route(city_count) for _ in range(pop_size)]

def rank_routes(population, city_coords):
    return sorted(
        [(route_distance(route, city_coords), route) for route in population],
        key=lambda x: x[0]
    )

def selection(ranked_pop, elite_size):
    selected = [route for _, route in ranked_pop[:elite_size]]
    fitness_values = [1 / d for d, _ in ranked_pop]
    total_fit = sum(fitness_values)
    probs = [f / total_fit for f in fitness_values]

    while len(selected) < len(ranked_pop):
        # Tournament selection (improved exploration & balance)
        tournament_size = 5
        tournament = random.sample(ranked_pop, tournament_size)
        tournament.sort(key=lambda x: x[0])
        selected.append(tournament[0][1])
    return selected

def ordered_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child_p1 = parent1[start:end]
    child_p2 = [gene for gene in parent2 if gene not in child_p1]
    return child_p2[:start] + child_p1 + child_p2[start:]

def mutate(route, mutation_rate):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]

def next_generation(current_pop, city_coords, elite_size, mutation_rate):
    ranked = rank_routes(current_pop, city_coords)
    selected = selection(ranked, elite_size)
    children = []
    length = len(selected)
    
    # Elitism: keep elite routes directly
    children.extend(selected[:elite_size])

    # Produce children with crossover and mutation
    while len(children) < length:
        parent1 = random.choice(selected)
        parent2 = random.choice(selected)
        child = ordered_crossover(parent1, parent2)
        mutate(child, mutation_rate)
        children.append(child)

    return children

def genetic_algorithm(city_coords, pop_size=150, elite_size=30, mutation_rate=0.02, generations=700):
    city_count = len(city_coords)
    population = initial_population(pop_size, city_count)
    best_route = None
    best_distance = float('inf')
    
    for gen in range(generations):
        ranked = rank_routes(population, city_coords)
        current_best_distance = ranked[0][0]
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = ranked[0][1]
        
        population = next_generation(population, city_coords, elite_size, mutation_rate)

        if gen % 50 == 0 or gen == generations-1:
            print(f"Generation {gen}: shortest path = {best_distance:.2f}")

    print("Finished Evolution.")
    return best_route, best_distance

if __name__ == "__main__":
    city_coords = [
        (60, 200), (180, 200), (80, 180), (140, 180),
        (20, 160), (100, 160), (200, 160), (140, 140),
        (40, 120), (100, 120)
    ]

    best_route, best_distance = genetic_algorithm(
        city_coords,
        pop_size=150,
        elite_size=30,
        mutation_rate=0.02,
        generations=700
    )
    print("Best route found:")
    print(best_route)
    print(f"Total distance: {best_distance:.2f}")

    print("Route coordinates:")
    for idx in best_route:
        print(city_coords[idx])
