import pygame
from stadium import PongEnvironment

if __name__ == "__main__":
    env = PongEnvironment(max_score=5)

    running = True
    done = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        _, reward, done = env.apply(mode='human')

        if done:
            print("Game Over!")
            env.reset()  # Reset for new game