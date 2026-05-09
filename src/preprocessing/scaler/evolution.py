import numpy as np
import cv2
from typing import List, Tuple
import random

class EvolutionaryOptimizer:
    """
    An evolutionary algorithm to estimate the optimal scaling factor for pixel art.
    Prioritizes larger scaling factors when losses are equal.
    """
    
    def __init__(
        self, 
        population_size: int = 50, 
        generations: int = 50, 
        mutation_rate: float = 0.2,
        mutation_strength: float = 0.1,
        min_s: float = 1.0,
        max_s: float = 20.0
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.min_s = min_s
        self.max_s = max_s

    def compute_loss(self, s: float, images: List[np.ndarray], batch_size: int = None) -> float:
        """Calculates average reconstruction loss across a batch of images."""
        if s <= 0:
            return float('inf')
            
        eval_images = images
        if batch_size and batch_size < len(images):
            eval_images = random.sample(images, batch_size)
            
        total_loss = 0.0
        for img in eval_images:
            h, w = img.shape[:2]
            target_w, target_h = round(w / s), round(h / s)
            
            if target_w <= 0 or target_h <= 0:
                total_loss += 1e9
                continue
                
            down = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            up = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
            
            loss = np.mean((img.astype(np.float32) - up.astype(np.float32))**2)
            total_loss += loss
            
        return total_loss / len(eval_images)

    def evolve(self, images: List[np.ndarray], initial_guess: float = 5.0, batch_size: int = 10) -> float:
        """Runs the evolutionary optimization loop with stochastic batching."""
        population = [
            max(self.min_s, min(self.max_s, initial_guess + random.uniform(-0.5, 0.5)))
            for _ in range(self.population_size)
        ]
        
        best_s = initial_guess
        best_loss = self.compute_loss(best_s, images, batch_size=min(len(images), 50))
        
        for gen in range(self.generations):
            current_batch = random.sample(images, min(len(images), batch_size)) if batch_size else images
            
            scores = []
            for s in population:
                loss = self.compute_loss(s, current_batch)
                scores.append((loss, s))
            
            scores.sort(key=lambda x: (x[0], -x[1]))
            
            current_best_loss, current_best_s = scores[0]
            
            if current_best_loss < best_loss:
                stable_loss = self.compute_loss(current_best_s, images, batch_size=min(len(images), 50))
                if stable_loss < best_loss:
                    best_loss = stable_loss
                    best_s = current_best_s
                
            parents = [s for loss, s in scores[:self.population_size // 2]]
            
            next_population = parents.copy()
            while len(next_population) < self.population_size:
                parent = random.choice(parents)
                child = parent
                if random.random() < self.mutation_rate:
                    strength = self.mutation_strength * (1 - gen/self.generations)
                    child += random.gauss(0, strength)
                
                if random.random() < 0.1:
                    child = (child + random.choice(parents)) / 2
                
                child = max(self.min_s, min(self.max_s, child))
                next_population.append(child)
                
            population = next_population
            
            if gen % 10 == 0 or gen == self.generations - 1:
                print(f"Generation {gen}: Best S = {best_s:.6f}, Stable Loss = {best_loss:.4f}", flush=True)
                
        return best_s

if __name__ == "__main__":
    # Small self-test if run directly
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Create a 20x20 pixel art upscaled by 5
    pixel_art = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    upscaled = cv2.resize(pixel_art, (100, 100), interpolation=cv2.INTER_NEAREST)
    
    optimizer = EvolutionaryOptimizer(generations=10, population_size=10)
    found_s = optimizer.evolve([upscaled], initial_guess=4.0)
    print(f"Test found S: {found_s}")
