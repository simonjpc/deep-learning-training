import gc
import pygame
import numpy as np

class PongEnvironment:
    def __init__(self, width=84, height=84+20, max_score=20):
        pygame.init()
        self.width = width
        self.height = height
        self.max_score = max_score
        self.score_height = 20
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ball_speed_val = 1.2
        self.pads_speed_val = 1
        self.pads_speed_val_agent = 3 # for AI agent paddle speed = 5. For human paddle speed = 1
        self.reset()

    def reset(self):
        #Clear surface arrays if they exist
        if hasattr(self, '_surface_array'):
            del self._surface_array
        
        self.ball = pygame.Rect(self.width // 2, self.height // 2 + self.score_height, 2, 2)
        self.agent_paddle = pygame.Rect(self.width - 10, self.height // 2 - 15 + self.score_height, 4, 8)
        self.opponent_paddle = pygame.Rect(10, self.height // 2 - 15 + self.score_height, 4, 8)

        self.ball_speed = [self.ball_speed_val, self.ball_speed_val * 0.9]
        self.agent_score = 0
        self.opponent_score = 0

        gc.collect()
        return self.get_state()

    def get_state(self):
        surface = pygame.surfarray.array3d(pygame.display.get_surface())
        return np.transpose(surface, (1, 0, 2))

    def get_state_gray(self):
        surface = pygame.surfarray.array3d(pygame.display.get_surface())
        gray = np.dot(surface[..., :3], [0.2989, 0.5870, 0.1140])
        output = np.transpose(gray, (1, 0))
        game_area = output[self.score_height:, :]
        desired_mean = (np.max(game_area) - np.min(game_area)) // 2 + 1
        return (game_area - desired_mean) / (np.max(desired_mean) + 1e-6)

    def apply(self, action=None, mode='agent', render_flag=False):
        reward = 0
        done = False
        constant_increase = 0.05
        
        if mode == 'agent':
            if action == 1:
                self.agent_paddle.move_ip(0, -self.pads_speed_val_agent)
            elif action == 2:
                self.agent_paddle.move_ip(0, self.pads_speed_val_agent)

        elif mode == 'human':
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.agent_paddle.move_ip(0, -self.pads_speed_val_agent)
            if keys[pygame.K_DOWN]:
                self.agent_paddle.move_ip(0, self.pads_speed_val_agent)

        self.agent_paddle.clamp_ip(pygame.Rect(0, self.score_height, self.width, self.height - self.score_height))

        self.move_opponent_paddle()
        self.ball.move_ip(*self.ball_speed)

        # Ball-wall collision
        if self.ball.top <= self.score_height or self.ball.bottom >= self.height:
            self.ball_speed[1] *= -1

        # Handle paddle collisions (deterministic bounce with speed variation near edges)
        if self.ball.colliderect(self.agent_paddle):
            self.handle_paddle_collision(self.agent_paddle, is_agent=True)
            self.ball_speed[0] = self.ball_speed[0] - constant_increase
            self.ball_speed[1] = self.ball_speed[1] - constant_increase if self.ball_speed[1] < 0 else self.ball_speed[1] + constant_increase

        elif self.ball.colliderect(self.opponent_paddle):
            self.handle_paddle_collision(self.opponent_paddle, is_agent=False)
            self.ball_speed[0] = self.ball_speed[0] + constant_increase
            self.ball_speed[1] = self.ball_speed[1] - constant_increase if self.ball_speed[1] < 0 else self.ball_speed[1] + constant_increase

        # Scoring
        if self.ball.left <= 0:
            self.agent_score += 1
            reward = 1
            self.ball_reset()

        elif self.ball.right >= self.width:
            self.opponent_score += 1
            reward = -1
            self.ball_reset()

        if self.agent_score >= self.max_score or self.opponent_score >= self.max_score:
            done = True

        if render_flag:
            self.render()

        return self.get_state_gray(), reward, done

    def handle_paddle_collision(self, paddle, is_agent):
        """
        Handles paddle collision with a slight deterministic speed increase near paddle edges.
        """
        # Horizontal bounce
        if is_agent:
            self.ball_speed[0] = -abs(self.ball_speed[0])  # Reflect left
            self.ball.right = paddle.left - 1
        else:
            self.ball_speed[0] = abs(self.ball_speed[0])  # Reflect right
            self.ball.left = paddle.right + 1

        # Determine relative contact position (0 = top, 1 = bottom)
        relative_contact = (self.ball.centery - paddle.top) / paddle.height

        # If near edges, increase vertical speed slightly (deterministic "edge boost")
        edge_threshold = 0.2  # Top 20% and bottom 20% count as edges
        speed_boost = 1.4 if relative_contact < edge_threshold or relative_contact > (1 - edge_threshold) else 1.0

        # Apply edge boost to vertical speed (keeping direction)
        self.ball_speed[1] *= speed_boost
        self.ball_speed[0] *= speed_boost

        # Limit vertical speed to prevent extreme angles
        max_horizontal_speed = self.ball_speed_val * 2.1
        max_vertical_speed = self.ball_speed_val * 1.7
        self.ball_speed[0] = float(np.clip(self.ball_speed[0], -max_horizontal_speed, max_horizontal_speed))
        self.ball_speed[1] = float(np.clip(self.ball_speed[1], -max_vertical_speed, max_vertical_speed))

    def ball_reset(self):
        self.ball = pygame.Rect(self.width // 2, self.height // 2 + self.score_height, 2, 2)
        horizontal_rand = np.random.rand()
        vertical_rand = np.random.rand()
        
        self.ball_speed = [
            self.ball_speed_val * (-1 if horizontal_rand >= 0.6 else 1),
            self.ball_speed_val * 0.9 * (-1 if vertical_rand >= 0.5 else 1)
        ]

    def move_opponent_paddle(self):
        if self.opponent_paddle.centery < self.ball.centery:
            self.opponent_paddle.move_ip(0, self.pads_speed_val)
        elif self.opponent_paddle.centery > self.ball.centery:
            self.opponent_paddle.move_ip(0, -self.pads_speed_val)

        self.opponent_paddle.clamp_ip(pygame.Rect(0, self.score_height, self.width, self.height - self.score_height))

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), self.agent_paddle)
        pygame.draw.rect(self.screen, (255, 255, 255), self.opponent_paddle)
        pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)

        font = pygame.font.SysFont(None, self.score_height)
        agent_text = font.render(f'{self.agent_score}', True, (255, 255, 255))
        opponent_text = font.render(f'{self.opponent_score}', True, (255, 255, 255))
        self.screen.blit(agent_text, (self.width - 22, 1))
        self.screen.blit(opponent_text, (15, 1))

        pygame.draw.line(self.screen, (255, 255, 255), (0, self.score_height), (self.width, self.score_height), 1)
        
        pygame.display.flip()
        self.clock.tick(40)

    def stop(self):
        if hasattr(self, '_surface_array'):
            del self._surface_array
        pygame.display.quit()
        pygame.quit()
        gc.collect()
