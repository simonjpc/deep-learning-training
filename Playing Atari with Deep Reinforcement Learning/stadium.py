import pygame
import numpy as np

class PongEnvironment:
    def __init__(self, width=400, height=300, max_score=10):
        pygame.init()
        self.width = width
        self.height = height
        self.max_score = max_score  # Maximum score to end the game
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.ball = pygame.Rect(self.width // 2, self.height // 2, 10, 10)
        self.agent_paddle = pygame.Rect(self.width - 20, self.height // 2 - 30, 10, 60)
        self.opponent_paddle = pygame.Rect(10, self.height // 2 - 30, 10, 60)

        self.ball_speed = [3, 3]
        self.agent_score = 0
        self.opponent_score = 0

        return self.get_state()

    def get_state(self):
        surface = pygame.surfarray.array3d(pygame.display.get_surface())
        return np.transpose(surface, (1, 0, 2))

    def apply(self, action=None, mode='agent'):
        reward = 0
        done = False

        if mode == 'agent':
            if action == 1:
                self.agent_paddle.move_ip(0, -5)
            elif action == 2:
                self.agent_paddle.move_ip(0, 5)

        elif mode == 'human':
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.agent_paddle.move_ip(0, -5)
            if keys[pygame.K_DOWN]:
                self.agent_paddle.move_ip(0, 5)

        self.agent_paddle.clamp_ip(pygame.Rect(0, 0, self.width, self.height))

        self.move_opponent_paddle()
        self.ball.move_ip(*self.ball_speed)

        # Ball-wall collision
        if self.ball.top <= 0 or self.ball.bottom >= self.height:
            self.ball_speed[1] *= -1

        # Paddle collisions
        if self.ball.colliderect(self.agent_paddle):
            self.ball_speed[0] *= -1
            reward = 1

        if self.ball.colliderect(self.opponent_paddle):
            self.ball_speed[0] *= -1

        # Scoring
        if self.ball.left <= 0:  # Agent scores
            self.agent_score += 1
            reward = 5
            self.ball_reset()

        elif self.ball.right >= self.width:  # Opponent scores
            self.opponent_score += 1
            reward = -2
            self.ball_reset()

        # Check for game over
        if self.agent_score >= self.max_score or self.opponent_score >= self.max_score:
            done = True

        self.render()

        return self.get_state(), reward, done

    def ball_reset(self):
        """ Reset ball to center after a point is scored """
        self.ball = pygame.Rect(self.width // 2, self.height // 2, 10, 10)
        self.ball_speed = [3 * (-1 if np.random.rand() > 0.5 else 1), 3 * (-1 if np.random.rand() > 0.5 else 1)]

    def move_opponent_paddle(self):
        if self.opponent_paddle.centery < self.ball.centery:
            self.opponent_paddle.move_ip(0, 3)
        elif self.opponent_paddle.centery > self.ball.centery:
            self.opponent_paddle.move_ip(0, -3)

        self.opponent_paddle.clamp_ip(pygame.Rect(0, 0, self.width, self.height))

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), self.agent_paddle)
        pygame.draw.rect(self.screen, (255, 255, 255), self.opponent_paddle)
        pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)
        pygame.draw.line(self.screen, (255, 255, 255), (self.width // 2, 0), (self.width // 2, self.height))

        # Draw scores
        font = pygame.font.SysFont(None, 30)
        agent_text = font.render(f'Agent: {self.agent_score}', True, (255, 255, 255))
        opponent_text = font.render(f'Opponent: {self.opponent_score}', True, (255, 255, 255))
        self.screen.blit(agent_text, (self.width - 120, 10))
        self.screen.blit(opponent_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)
