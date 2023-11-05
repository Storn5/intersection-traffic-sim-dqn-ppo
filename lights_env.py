import math
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from stable_baselines3.common.env_checker import check_env

# Constants
CHECK_ENV = False
REWARD_AWT = True
TIME_LIMIT = 100_000
WIDTH, HEIGHT = 900, 600
ROAD_WIDTH = 200
LIGHT_SIZE = 40
MAX_WHEEL_ANGLE = 45.0
MAX_SPEED = 0.3
WHEEL_TURN_SPEED = 0.1
ACCELERATION = 0.0001
FRICTION = 0.00003
CAR_WIDTH = 30
CAR_LENGTH = 50
HALF_WIDTH, HALF_HEIGHT, HALF_ROAD_WIDTH, QUARTER_ROAD_WIDTH, HALF_CAR_LENGTH, HALF_CAR_WIDTH = WIDTH // 2, HEIGHT // 2, ROAD_WIDTH // 2, ROAD_WIDTH // 4, CAR_LENGTH // 2, CAR_WIDTH // 2
BASE_REWARD = 10
AWT_COEF = 1
AQL_COEF = 0
FIXED_TIME = True

BLACK = (0, 0, 0)
GRAY = (122, 122, 122)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
CYAN = (0, 255, 255)
DARK_RED = (100, 0, 0)
DARK_GREEN = (0, 100, 0)
DARK_YELLOW = (100, 100, 0)

class Car:
  def __init__(self, x, y, origin, target, render_mode=None, screen=None):
    self.exited = False
    self.wait_time = 0
    self.car_x = x
    self.car_y = y
    self.car_origin = origin
    self.car_target = target
    self.car_angle = 0 if origin == 2 else 90 if origin == 3 else 180 if origin == 0 else 270
    self.car_angle_rad = math.radians(self.car_angle)
    self.car_speed = 0.0
    self.wheel_angle = self.car_angle
    self.wheel_angle_rad = math.radians(self.wheel_angle)
    self.front_radius = 0.0
    self.rear_radius = 0.0
    self.front_wheel_x = self.car_x + (HALF_CAR_LENGTH * math.cos(self.car_angle_rad))
    self.front_wheel_y = self.car_y - (HALF_CAR_LENGTH * math.sin(self.car_angle_rad))
    self.rear_wheel_x = self.car_x - (HALF_CAR_LENGTH * math.cos(self.car_angle_rad))
    self.rear_wheel_y = self.car_y + (HALF_CAR_LENGTH * math.sin(self.car_angle_rad))

    if render_mode in ('human', 'rgb_array'):
      self.car_surf = pygame.Surface((CAR_LENGTH, CAR_WIDTH)).convert_alpha(screen)
      color = YELLOW if origin == 0 else GREEN if origin == 1 else BLUE if origin == 2 else RED
      self.car_surf.fill(color)
      self.wheel_surf = pygame.Surface((10, 5)).convert_alpha(screen)
      self.wheel_surf.fill(WHITE)

  def automatic_input(self, lights_horizontal, lights_vertical):
    if self.check_if_in_queue() == -1 or (((self.car_origin == 0 or self.car_origin == 2) and lights_horizontal == 2) \
      or ((self.car_origin == 1 or self.car_origin == 3) and lights_vertical == 2)):
      self.car_speed += ACCELERATION
      self.car_speed = min(MAX_SPEED, self.car_speed)
    else:
      self.car_speed -= ACCELERATION
      self.car_speed = max(0, self.car_speed)
      self.wait_time += 1

    if self.car_speed > 0:
      self.car_speed -= FRICTION

  def update_position(self):
    self.wheel_angle_rad = math.radians(self.wheel_angle)
    if self.wheel_angle_rad - self.car_angle_rad != 0:
      self.front_radius = CAR_LENGTH / math.sin(self.wheel_angle_rad - self.car_angle_rad)
      self.rear_radius = self.front_radius * math.cos(self.wheel_angle_rad - self.car_angle_rad)
    else:
      self.front_radius, self.rear_radius = 0.0, 0.0
    # Calculate the car's new position
    self.car_x += self.car_speed * math.cos(self.wheel_angle_rad)
    self.car_y -= self.car_speed * math.sin(self.wheel_angle_rad)
    # Check if car exited
    if self.car_x > WIDTH - HALF_CAR_LENGTH and self.car_target == 0:
      self.exited = True
    if self.car_x < HALF_CAR_LENGTH and self.car_target == 2:
      self.exited = True
    if self.car_y > HEIGHT - HALF_CAR_LENGTH and self.car_target == 3:
      self.exited = True
    if self.car_y < HALF_CAR_LENGTH and self.car_target == 1:
      self.exited = True
    # Calculate the car's new angle
    if self.front_radius != 0:
      delta_angle = (self.car_speed / (2 * math.pi * self.front_radius)) * 360
      self.car_angle += delta_angle
      self.car_angle_rad = math.radians(self.car_angle)
      self.wheel_angle += delta_angle
      self.wheel_angle_rad = math.radians(self.wheel_angle)
    # Calculate position of wheels
    self.front_wheel_x = self.car_x + (HALF_CAR_LENGTH * math.cos(self.car_angle_rad))
    self.front_wheel_y = self.car_y - (HALF_CAR_LENGTH * math.sin(self.car_angle_rad))
    self.rear_wheel_x = self.car_x - (HALF_CAR_LENGTH * math.cos(self.car_angle_rad))
    self.rear_wheel_y = self.car_y + (HALF_CAR_LENGTH * math.sin(self.car_angle_rad))

  def check_collisions(self, other_cars, lights_horizontal, lights_vertical):
    if self.exited:
      return
    # Check car collisions
    for car in other_cars:
      if (not car.exited) and self != car and math.sqrt(((self.car_x - car.car_x)**2) + ((self.car_y - car.car_y)**2)) <= CAR_LENGTH:
        self.car_speed = 0
        car.car_speed = 0
        return True
    return False

  def check_if_in_queue(self):
    if self.car_origin == 0 and self.car_x > HALF_WIDTH + HALF_ROAD_WIDTH:
      return 0
    elif self.car_origin == 1 and self.car_y < HALF_HEIGHT - HALF_ROAD_WIDTH:
      return 1
    elif self.car_origin == 2 and self.car_x < HALF_WIDTH - HALF_ROAD_WIDTH:
      return 2
    elif self.car_origin == 3 and self.car_y > HALF_HEIGHT + HALF_ROAD_WIDTH:
      return 3
    return -1

  def draw(self, screen):
    if self.exited:
      return
    # Draw the car
    rotated_car = pygame.transform.rotate(self.car_surf, self.car_angle)
    rotated_car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
    screen.blit(rotated_car, rotated_car_rect)
    # Draw the wheels
    rotated_front_wheel = pygame.transform.rotate(self.wheel_surf, self.wheel_angle)
    rotated_front_wheel_1_rect = rotated_front_wheel.get_rect(center=(self.front_wheel_x + (HALF_CAR_WIDTH * math.sin(self.car_angle_rad)), self.front_wheel_y + (HALF_CAR_WIDTH * math.cos(self.car_angle_rad))))
    rotated_front_wheel_2_rect = rotated_front_wheel.get_rect(center=(self.front_wheel_x - (HALF_CAR_WIDTH * math.sin(self.car_angle_rad)), self.front_wheel_y - (HALF_CAR_WIDTH * math.cos(self.car_angle_rad))))
    rotated_rear_wheel = pygame.transform.rotate(self.wheel_surf, self.car_angle)
    rotated_rear_wheel_1_rect = rotated_rear_wheel.get_rect(center=(self.rear_wheel_x + (HALF_CAR_WIDTH * math.sin(self.car_angle_rad)), self.rear_wheel_y + (HALF_CAR_WIDTH * math.cos(self.car_angle_rad))))
    rotated_rear_wheel_2_rect = rotated_rear_wheel.get_rect(center=(self.rear_wheel_x - (HALF_CAR_WIDTH * math.sin(self.car_angle_rad)), self.rear_wheel_y - (HALF_CAR_WIDTH * math.cos(self.car_angle_rad))))
    screen.blit(rotated_front_wheel, rotated_front_wheel_1_rect)
    screen.blit(rotated_front_wheel, rotated_front_wheel_2_rect)
    screen.blit(rotated_rear_wheel, rotated_rear_wheel_1_rect)
    screen.blit(rotated_rear_wheel, rotated_rear_wheel_2_rect)

class LightsEnv(gym.Env):
  metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

  def __init__(self, render_mode=None, fixed_time=True, reward_awt=False):
    super().__init__()
    self.fixed_time = fixed_time
    self.reward_awt = reward_awt
    self.action_space = spaces.Discrete(4) if FIXED_TIME else spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
    self._time = 0
    self._episode_ended = False
    self.render_mode = render_mode
    self.clock = None
    self.screen = None
    if render_mode in ('human', 'rgb_array'):
      pygame.init()
      pygame.display.init()
      self.screen = pygame.display.set_mode((WIDTH, HEIGHT)) # or pygame.Surface???

  def reset(self, seed=None, options=None):
    super().reset(seed=seed, options=options)
    self._time = 0
    self.queue_right = 0
    self.queue_top = 0
    self.queue_left = 0
    self.queue_bottom = 0
    self.chance_horizontal = np.random.rand() * 0.25 + 0.375
    self.chance_top = np.random.rand() * 0.25 + 0.375
    self.chance_right = np.random.rand() * 0.25 + 0.375
    self.cars = []
    for _ in range(5):
      self.spawn_car()
    self.lights_horizontal, self.lights_vertical = 2, 0
    self.next_lights_vertical = True
    self._episode_ended = False

    observations = self.get_normalized_observation()
    if self.render_mode in ('human', 'rgb_array'):
      self.render()
    return observations, self.get_info()

  def step(self, action):
    num_steps = 1000 if self.fixed_time else int(((action + 1) / 20) * TIME_LIMIT)
    for _ in range(num_steps):
      if self._episode_ended:
        return self.reset()

      # Make sure episodes don't go on forever.
      self._time += 1
      if self._time > TIME_LIMIT:
        self._score = self.check_reward()
        self._episode_ended = True
        break

      if self.fixed_time:
        self.switch_lights(action)

      if self._time == self.next_car_spawn:
        self.update_queues()
        self.spawn_car()

      for car in self.cars:
        if not car.exited:
          car.automatic_input(self.lights_horizontal, self.lights_vertical)

      for car in self.cars:
        if not car.exited:
          car.update_position()

      for car in self.cars:
        if car.check_collisions(self.cars, self.lights_horizontal, self.lights_vertical):
          self._score = -1000
          self._episode_ended = True
          break

      if self._episode_ended:
        break

      if self.render_mode in ('human', 'rgb_array') and self._time % 100 == 0:
        self.render()

    observations = self.get_normalized_observation()
    info = self.get_info()
    reward = 0
    if self._episode_ended:
      reward = self._score
    else:
      reward = self.check_reward()

    if not self.fixed_time:
      self.switch_lights()

    return (
      observations,
      reward,
      self._episode_ended,
      self._time > TIME_LIMIT,
      info,
    )

  def switch_lights(self, action=None):
    if action is None:
      if self.lights_horizontal == 0 and self.lights_vertical == 2:
        self.lights_vertical = 1
      elif self.lights_horizontal == 2 and self.lights_vertical == 0:
        self.lights_horizontal = 1
      elif self.lights_horizontal == 0 and self.lights_vertical == 1 and not self.next_lights_vertical:
        self.lights_horizontal = 1
        self.lights_vertical = 0
      elif self.lights_horizontal == 0 and self.lights_vertical == 1 and self.next_lights_vertical:
        self.lights_vertical = 2
        self.next_lights_vertical = False
      elif self.lights_horizontal == 1 and self.lights_vertical == 0 and self.next_lights_vertical:
        self.lights_horizontal = 0
        self.lights_vertical = 1
      elif self.lights_horizontal == 1 and self.lights_vertical == 0 and not self.next_lights_vertical:
        self.lights_horizontal = 2
        self.next_lights_vertical = True
    else:
      if action == 0:
        self.lights_horizontal = 2
        self.lights_vertical = 0
      elif action == 1:
        self.lights_horizontal = 1
        self.lights_vertical = 0
      elif action == 2:
        self.lights_horizontal = 0
        self.lights_vertical = 1
      elif action == 3:
        self.lights_horizontal = 0
        self.lights_vertical = 2

  def spawn_car(self):
    # Spawn new car coming from random direction
    origin = 2
    if np.random.rand() < self.chance_horizontal:
      if np.random.rand() < self.chance_right:
        origin = 0
    else:
      if np.random.rand() < self.chance_top:
        origin = 1
      else:
        origin = 3
    target = 0 if origin == 2 else 1 if origin == 3 else 2 if origin == 0 else 3
    car = Car(0, 0, origin, target, self.render_mode, self.screen)
    if origin == 0:
      car.car_y = HALF_HEIGHT - QUARTER_ROAD_WIDTH
      car.car_x = HALF_WIDTH + ROAD_WIDTH
      for _ in range(self.queue_right):
        car.car_x += CAR_LENGTH * 2
      self.queue_right += 1
    elif origin == 2:
      car.car_y = HALF_HEIGHT + QUARTER_ROAD_WIDTH
      car.car_x = HALF_WIDTH - ROAD_WIDTH
      for _ in range(self.queue_left):
        car.car_x -= CAR_LENGTH * 2
      self.queue_left += 1
    elif origin == 1:
      car.car_x = HALF_WIDTH - QUARTER_ROAD_WIDTH
      car.car_y = HALF_HEIGHT - ROAD_WIDTH
      for _ in range(self.queue_top):
        car.car_y -= CAR_LENGTH * 2
      self.queue_top += 1
    elif origin == 3:
      car.car_x = HALF_WIDTH + QUARTER_ROAD_WIDTH
      car.car_y = HALF_HEIGHT + ROAD_WIDTH
      for _ in range(self.queue_bottom):
        car.car_y += CAR_LENGTH * 2
      self.queue_bottom += 1
    self.cars.append(car)
    self.next_car_spawn = self._time + np.random.randint(TIME_LIMIT // 5)

  def update_queues(self):
    self.queue_top, self.queue_bottom, self.queue_left, self.queue_right = 0, 0, 0, 0
    for car in self.cars:
      if not car.exited:
        car_queue = car.check_if_in_queue()
        if car_queue == 0:
          self.queue_right += 1
        elif car_queue == 1:
          self.queue_top += 1
        elif car_queue == 2:
          self.queue_left += 1
        elif car_queue == 3:
          self.queue_bottom += 1

  def check_reward(self):
    reward = BASE_REWARD
    if self.reward_awt:
      for car in self.cars:
        reward -= 100 * car.wait_time / TIME_LIMIT
      reward /= len(self.cars)
    else:
      reward -= 10 * (self.queue_top + self.queue_left + self.queue_bottom + self.queue_right)
    return reward

  def get_info(self):
    awt = 0
    for car in self.cars:
      awt += car.wait_time
    awt /= len(self.cars)
    aql = self.queue_top + self.queue_left + self.queue_bottom + self.queue_right
    return { 'awt': awt / 1000, 'aql': aql }

  def get_normalized_observation(self):
    self.update_queues()
    num_cars_horizontal, num_cars_vertical, num_cars_horizontal_prev, num_cars_vertical_prev = 0, 0, 0, 0
    for car in self.cars:
      if car.car_origin == 0 or car.car_origin == 2:
        if car.exited:
          num_cars_horizontal_prev += 1
        else:
          num_cars_horizontal += 1
      else:
        if car.exited:
          num_cars_vertical_prev += 1
        else:
          num_cars_vertical += 1
    cur_car_ratio = 0 if num_cars_horizontal == 0 and num_cars_vertical == 0 else num_cars_horizontal / (num_cars_vertical + num_cars_horizontal)
    prev_car_ratio = 0 if num_cars_horizontal_prev == 0 and num_cars_vertical_prev == 0 else num_cars_horizontal_prev / (num_cars_vertical_prev + num_cars_horizontal_prev)
    queue_ratio = 0 if self.queue_left == 0 and self.queue_bottom == 0 and self.queue_right == 0 and self.queue_top == 0 else (self.queue_left + self.queue_right) / (self.queue_top + self.queue_left + self.queue_bottom + self.queue_right)
    observations = np.array([self.lights_horizontal / 2, self.lights_vertical / 2, self.next_lights_vertical, cur_car_ratio, prev_car_ratio, queue_ratio], dtype=np.float32)
    return observations

  def close(self):
    if self.screen is not None:
      pygame.display.quit()
      pygame.quit()

  def render(self):
    if self.render_mode is None:
      return
    if self.clock is None:
      self.clock = pygame.time.Clock()

    # Clear the screen
    self.screen.fill(BLACK)

    # Draw sidewalks
    pygame.draw.rect(self.screen, GRAY, pygame.Rect(0, 0, (WIDTH - ROAD_WIDTH) // 2, (HEIGHT - ROAD_WIDTH) // 2))
    pygame.draw.rect(self.screen, GRAY, pygame.Rect((WIDTH + ROAD_WIDTH) // 2, 0, (WIDTH - ROAD_WIDTH) // 2, (HEIGHT - ROAD_WIDTH) // 2))
    pygame.draw.rect(self.screen, GRAY, pygame.Rect(0, (HEIGHT + ROAD_WIDTH) // 2, (WIDTH - ROAD_WIDTH) // 2, (HEIGHT - ROAD_WIDTH) // 2))
    pygame.draw.rect(self.screen, GRAY, pygame.Rect((WIDTH + ROAD_WIDTH) // 2, (HEIGHT + ROAD_WIDTH) // 2, (WIDTH - ROAD_WIDTH) // 2, (HEIGHT - ROAD_WIDTH) // 2))
    # Draw road markings
    for x in range(0, (WIDTH - ROAD_WIDTH) // 2, 100):
      pygame.draw.line(self.screen, WHITE, (x, HALF_HEIGHT), (x + 50, HALF_HEIGHT), 5)
    for x in range((WIDTH + ROAD_WIDTH) // 2 + 50, WIDTH, 100):
      pygame.draw.line(self.screen, WHITE, (x, HALF_HEIGHT), (x + 50, HALF_HEIGHT), 5)
    for y in range(50, (HEIGHT - ROAD_WIDTH) // 2, 100):
      pygame.draw.line(self.screen, WHITE, (HALF_WIDTH, y), (HALF_WIDTH, y + 50), 5)
    for y in range((HEIGHT + ROAD_WIDTH) // 2 + 50, HEIGHT, 100):
      pygame.draw.line(self.screen, WHITE, (HALF_WIDTH, y), (HALF_WIDTH, y + 50), 5)
    # Draw lights
    pygame.draw.rect(self.screen, BLACK, pygame.Rect((WIDTH + ROAD_WIDTH) // 2, (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 3), LIGHT_SIZE, LIGHT_SIZE * 3))
    pygame.draw.circle(self.screen, RED if self.lights_vertical == 0 else DARK_RED, ((WIDTH + ROAD_WIDTH) // 2 + (LIGHT_SIZE // 2), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 2.5)), 15)
    pygame.draw.circle(self.screen, YELLOW if self.lights_vertical == 1 else DARK_YELLOW, ((WIDTH + ROAD_WIDTH) // 2 + (LIGHT_SIZE // 2), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 1.5)), 15)
    pygame.draw.circle(self.screen, GREEN if self.lights_vertical == 2 else DARK_GREEN, ((WIDTH + ROAD_WIDTH) // 2 + (LIGHT_SIZE // 2), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 0.5)), 15)
    pygame.draw.rect(self.screen, BLACK, pygame.Rect((WIDTH - ROAD_WIDTH) // 2 - LIGHT_SIZE, (HEIGHT + ROAD_WIDTH) // 2, LIGHT_SIZE, LIGHT_SIZE * 3))
    pygame.draw.circle(self.screen, RED if self.lights_vertical == 0 else DARK_RED, ((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE // 2), (HEIGHT + ROAD_WIDTH) // 2 + (LIGHT_SIZE * 2.5)), 15)
    pygame.draw.circle(self.screen, YELLOW if self.lights_vertical == 1 else DARK_YELLOW, ((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE // 2), (HEIGHT + ROAD_WIDTH) // 2 + (LIGHT_SIZE * 1.5)), 15)
    pygame.draw.circle(self.screen, GREEN if self.lights_vertical == 2 else DARK_GREEN, ((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE // 2), (HEIGHT + ROAD_WIDTH) // 2 + (LIGHT_SIZE * 0.5)), 15)
    pygame.draw.rect(self.screen, BLACK, pygame.Rect((WIDTH + ROAD_WIDTH) // 2, (HEIGHT + ROAD_WIDTH) // 2, LIGHT_SIZE * 3, LIGHT_SIZE))
    pygame.draw.circle(self.screen, RED if self.lights_horizontal == 0 else DARK_RED, ((WIDTH + ROAD_WIDTH) // 2 + (LIGHT_SIZE * 2.5), (HEIGHT + ROAD_WIDTH) // 2 + (LIGHT_SIZE // 2)), 15)
    pygame.draw.circle(self.screen, YELLOW if self.lights_horizontal == 1 else DARK_YELLOW, ((WIDTH + ROAD_WIDTH) // 2 + (LIGHT_SIZE * 1.5), (HEIGHT + ROAD_WIDTH) // 2 + (LIGHT_SIZE // 2)), 15)
    pygame.draw.circle(self.screen, GREEN if self.lights_horizontal == 2 else DARK_GREEN, ((WIDTH + ROAD_WIDTH) // 2 + (LIGHT_SIZE * 0.5), (HEIGHT + ROAD_WIDTH) // 2 + (LIGHT_SIZE // 2)), 15)
    pygame.draw.rect(self.screen, BLACK, pygame.Rect((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 3), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE), LIGHT_SIZE * 3, LIGHT_SIZE))
    pygame.draw.circle(self.screen, RED if self.lights_horizontal == 0 else DARK_RED, ((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 2.5), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE // 2)), 15)
    pygame.draw.circle(self.screen, YELLOW if self.lights_horizontal == 1 else DARK_YELLOW, ((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 1.5), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE // 2)), 15)
    pygame.draw.circle(self.screen, GREEN if self.lights_horizontal == 2 else DARK_GREEN, ((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 0.5), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE // 2)), 15)

    for car in self.cars:
      car.draw(self.screen)

    if self.render_mode == 'human':
      assert self.screen is not None
      pygame.event.pump()
      self.clock.tick(self.metadata['render_fps'])
      pygame.display.flip()
    elif self.render_mode == 'rgb_array':
      full_res = np.transpose(
        np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
      )
      res_scale = 0.5
      return full_res[::int(1 / res_scale), ::int(1 / res_scale), :]

if __name__ == '__main__':
  register(
    id='Lights-v1',
    entry_point=LightsEnv,
    max_episode_steps=TIME_LIMIT // 10 + 1,
  )

  env = gym.make('Lights-v1', render_mode='human')
  obs, _ = env.reset()

  if CHECK_ENV:
    check_env(env, warn=True)
    obs, _ = env.reset()

  env.render()

  for step in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    print("obs=", obs, "reward=", reward, "done=", done)
    env.render()
    if done:
      print("Goal reached!", "reward=", reward)
      break
  env.close()
