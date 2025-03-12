# import gc
# import pygame
# import numpy as np

# class PongEnvironment:
#     def __init__(self, width=336, height=336, max_score=10):
#         pygame.init()
#         self.width = width
#         self.height = height
#         self.max_score = max_score  # Maximum score to end the game
#         self.screen = pygame.display.set_mode((width, height))
#         self.clock = pygame.time.Clock()
#         self.ball_speed_val = 7
#         self.pads_speed_val = 4
#         self.reset()

#     def reset(self):
#         # Clear surface arrays if they exist
#         if hasattr(self, '_surface_array'):
#             del self._surface_array
        
#         self.ball = pygame.Rect(self.width // 2, self.height // 2, 10, 10)
#         self.agent_paddle = pygame.Rect(self.width - 20, self.height // 2 - 30, 10, 60)
#         self.opponent_paddle = pygame.Rect(10, self.height // 2 - 30, 10, 60)

#         self.ball_speed = [self.ball_speed_val, self.ball_speed_val * 0.5]
#         self.agent_score = 0
#         self.opponent_score = 0

#         gc.collect()
#         return self.get_state()

#     def get_state(self):
#         surface = pygame.surfarray.array3d(pygame.display.get_surface())
#         return np.transpose(surface, (1, 0, 2))
    
#     def get_state_gray(self):
#         surface = pygame.surfarray.array3d(pygame.display.get_surface())
#         gray = np.dot(surface[..., :3], [0.2989, 0.5870, 0.1140])  # Standard grayscale conversion
#         output = np.transpose(gray, (1, 0))
#         # min, max = np.min(output), np.max(output)
#         mean = np.mean(output)
#         # std = np.std(output) + 1e-6
#         return (output - np.min(output)) / (np.max(output) + 1e-6)  # Shape (H, W)

#     def apply(self, action=None, mode='agent', render_flag=False):
#         reward = 0
#         done = False

#         if mode == 'agent':
#             if action == 1:
#                 self.agent_paddle.move_ip(0, -self.pads_speed_val)
#             elif action == 2:
#                 self.agent_paddle.move_ip(0, self.pads_speed_val)

#         elif mode == 'human':
#             keys = pygame.key.get_pressed()
#             if keys[pygame.K_UP]:
#                 self.agent_paddle.move_ip(0, -self.pads_speed_val)
#             if keys[pygame.K_DOWN]:
#                 self.agent_paddle.move_ip(0, self.pads_speed_val)

#         self.agent_paddle.clamp_ip(pygame.Rect(0, 0, self.width, self.height))

#         self.move_opponent_paddle()
#         self.ball.move_ip(*self.ball_speed)

#         # Ball-wall collision
#         if self.ball.top <= 0 or self.ball.bottom >= self.height:
#             self.ball_speed[1] *= -1

#         # Paddle collisions
#         if self.ball.colliderect(self.agent_paddle):
#             self.ball_speed[0] *= -1
#             reward = 0.1

#         if self.ball.colliderect(self.opponent_paddle):
#             self.ball_speed[0] *= -1

#         # Scoring
#         if self.ball.left <= 0:  # Agent scores
#             self.agent_score += 1
#             reward = 1
#             self.ball_reset()

#         elif self.ball.right >= self.width:  # Opponent scores
#             self.opponent_score += 1
#             reward = -1
#             self.ball_reset()

#         # Check for game over
#         if self.agent_score >= self.max_score or self.opponent_score >= self.max_score:
#             done = True

#         if render_flag:
#             self.render()

#         # return self.get_state(), reward, done
#         return self.get_state_gray(), reward, done

#     def ball_reset(self):
#         """ Reset ball to center after a point is scored """
#         self.ball = pygame.Rect(self.width // 2, self.height // 2, 10, 10)
#         self.ball_speed = [self.ball_speed_val * (-1 if np.random.rand() > 0.5 else 1), self.ball_speed_val * 0.5 * (-1 if np.random.rand() > 0.5 else 1)]

#     def move_opponent_paddle(self):
#         if self.opponent_paddle.centery < self.ball.centery:
#             self.opponent_paddle.move_ip(0, self.pads_speed_val)
#         elif self.opponent_paddle.centery > self.ball.centery:
#             self.opponent_paddle.move_ip(0, -self.pads_speed_val)

#         self.opponent_paddle.clamp_ip(pygame.Rect(0, 0, self.width, self.height))

#     def render(self):
#         self.screen.fill((0, 0, 0))
#         pygame.draw.rect(self.screen, (255, 255, 255), self.agent_paddle)
#         pygame.draw.rect(self.screen, (255, 255, 255), self.opponent_paddle)
#         pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)

#         # Draw scores
#         font = pygame.font.SysFont(None, 30)
#         agent_text = font.render(f'Agent: {self.agent_score}', True, (255, 255, 255))
#         opponent_text = font.render(f'Opponent: {self.opponent_score}', True, (255, 255, 255))
#         self.screen.blit(agent_text, (self.width - 120, 10))
#         self.screen.blit(opponent_text, (10, 10))

#         pygame.display.flip()
#         self.clock.tick(30)

#     def stop(self):
#         # Clear surface arrays
#         if hasattr(self, '_surface_array'):
#             del self._surface_array
#         pygame.display.quit()
#         pygame.quit()
#         gc.collect()

import gc
import pygame
import numpy as np

class PongEnvironment:
    def __init__(self, width=84, height=84, max_score=20):
        pygame.init()
        self.width = width
        self.height = height
        self.max_score = max_score
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ball_speed_val = 7
        self.pads_speed_val = 5
        self.pads_speed_val_agent = 10
        self.reset()

    def reset(self):
        #Clear surface arrays if they exist
        if hasattr(self, '_surface_array'):
            del self._surface_array
        
        self.ball = pygame.Rect(self.width // 2, self.height // 2, 10, 10)
        self.agent_paddle = pygame.Rect(self.width - 20, self.height // 2 - 30, 10, 60)
        self.opponent_paddle = pygame.Rect(10, self.height // 2 - 30, 10, 60)

        self.ball_speed = [self.ball_speed_val, self.ball_speed_val * 0.5]
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
        desired_mean = (np.max(output) - np.min(output)) // 2 + 1
        return (output - desired_mean) / (np.max(desired_mean) + 1e-6)

    def apply(self, action=None, mode='agent', render_flag=False):
        reward = 0
        done = False

        if mode == 'agent':
            if action == 1:
                self.agent_paddle.move_ip(0, -self.pads_speed_val_agent)
            elif action == 2:
                self.agent_paddle.move_ip(0, self.pads_speed_val_agent)

        elif mode == 'human':
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.agent_paddle.move_ip(0, -self.pads_speed_val)
            if keys[pygame.K_DOWN]:
                self.agent_paddle.move_ip(0, self.pads_speed_val)

        self.agent_paddle.clamp_ip(pygame.Rect(0, 0, self.width, self.height))

        self.move_opponent_paddle()
        self.ball.move_ip(*self.ball_speed)

        # Ball-wall collision
        if self.ball.top <= 0 or self.ball.bottom >= self.height:
            self.ball_speed[1] *= -1

        # Handle paddle collisions (deterministic bounce with speed variation near edges)
        if self.ball.colliderect(self.agent_paddle):
            self.handle_paddle_collision(self.agent_paddle, is_agent=True)
            reward = 1

        elif self.ball.colliderect(self.opponent_paddle):
            self.handle_paddle_collision(self.opponent_paddle, is_agent=False)

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
        edge_threshold = 0.3  # Top 30% and bottom 30% count as edges
        speed_boost = 1.2 if relative_contact < edge_threshold or relative_contact > (1 - edge_threshold) else 1.0

        # Apply edge boost to vertical speed (keeping direction)
        self.ball_speed[1] *= speed_boost

        # Limit vertical speed to prevent extreme angles
        max_vertical_speed = self.ball_speed_val * 1.5
        self.ball_speed[1] = np.clip(self.ball_speed[1], -max_vertical_speed, max_vertical_speed)

    def ball_reset(self):
        self.ball = pygame.Rect(self.width // 2, self.height // 2, 10, 10)
        self.ball_speed = [
            self.ball_speed_val * (-1 if np.random.rand() > 0.6 else 1), # right start more frequent than left start
            self.ball_speed_val * 0.5 * (-1 if np.random.rand() > 0.5 else 1)
        ]

    def move_opponent_paddle(self):
        if self.opponent_paddle.centery < self.ball.centery:
            self.opponent_paddle.move_ip(0, self.pads_speed_val)
        elif self.opponent_paddle.centery > self.ball.centery:
            self.opponent_paddle.move_ip(0, -self.pads_speed_val)

        self.opponent_paddle.clamp_ip(pygame.Rect(0, 0, self.width, self.height))

    def render(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), self.agent_paddle)
        pygame.draw.rect(self.screen, (255, 255, 255), self.opponent_paddle)
        pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)

        font = pygame.font.SysFont(None, 30)
        agent_text = font.render(f'Agent: {self.agent_score}', True, (255, 255, 255))
        opponent_text = font.render(f'Opponent: {self.opponent_score}', True, (255, 255, 255))
        self.screen.blit(agent_text, (self.width - 120, 10))
        self.screen.blit(opponent_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

    def stop(self):
        if hasattr(self, '_surface_array'):
            del self._surface_array
        pygame.display.quit()
        pygame.quit()
        gc.collect()
