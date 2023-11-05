import math
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from stable_baselines3.common.env_checker import check_env

# Constants
CHECK_ENV = False
SETUP = 2
TIME_LIMIT = 24_000
FRAMES_PER_STEP = 3
SPEED_PENALTY = 0.02
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

  def process_ai_input(self, action):
    if self.exited:
      return
    if action == 0 and self.wheel_angle - self.car_angle < MAX_WHEEL_ANGLE:
      self.wheel_angle += WHEEL_TURN_SPEED
    elif action == 1 and self.wheel_angle - self.car_angle > -MAX_WHEEL_ANGLE:
      self.wheel_angle -= WHEEL_TURN_SPEED
    elif action == 2:
      self.car_speed += ACCELERATION
      self.car_speed = min(MAX_SPEED, self.car_speed)
    elif action == 3:
      self.car_speed -= ACCELERATION
      self.car_speed = max(0, self.car_speed)

    if self.car_speed > 0:
      self.car_speed -= FRICTION


  def automatic_input(self, lights_horizontal, lights_vertical):
    if self.exited:
      return
    if (self.car_origin == 0 and lights_horizontal != 2 and self.car_x > HALF_WIDTH + HALF_ROAD_WIDTH) or\
      (self.car_origin == 1 and lights_vertical != 2 and self.car_y < HALF_HEIGHT - HALF_ROAD_WIDTH) or\
      (self.car_origin == 2 and lights_horizontal != 2 and self.car_x < HALF_WIDTH - HALF_ROAD_WIDTH) or\
      (self.car_origin == 3 and lights_vertical != 2 and self.car_y > HALF_HEIGHT + HALF_ROAD_WIDTH):
      self.car_speed -= ACCELERATION
      self.car_speed = max(0, self.car_speed)
    else:
      self.car_speed += ACCELERATION
      self.car_speed = min(MAX_SPEED, self.car_speed)

    if self.car_speed > 0:
      self.car_speed -= FRICTION

  def update_position(self):
    if self.exited:
      return
    self.wheel_angle_rad = math.radians(self.wheel_angle)
    if self.wheel_angle_rad - self.car_angle_rad != 0:
      self.front_radius = CAR_LENGTH / math.sin(self.wheel_angle_rad - self.car_angle_rad)
      self.rear_radius = self.front_radius * math.cos(self.wheel_angle_rad - self.car_angle_rad)
    else:
      self.front_radius, self.rear_radius = 0.0, 0.0
    # Calculate the car's new position
    self.car_x += self.car_speed * math.cos(self.wheel_angle_rad)
    self.car_y -= self.car_speed * math.sin(self.wheel_angle_rad)
    # Don't allow car to exit screen
    if self.car_x > WIDTH - HALF_CAR_LENGTH:
      self.exited = True
    if self.car_x < HALF_CAR_LENGTH:
      self.exited = True
    if self.car_y > HEIGHT - HALF_CAR_LENGTH:
      self.exited = True
    if self.car_y < HALF_CAR_LENGTH:
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

  def check_collisions(self, other_cars, action, lights_horizontal, lights_vertical):
    if self.exited:
      return
    # Check car collisions
    for car in other_cars:
      if (not car.exited) and self != car and math.sqrt(((self.car_x - car.car_x)**2) + ((self.car_y - car.car_y)**2)) <= CAR_LENGTH:
        return True
    # Check intersection with road edges
    if self.car_x < HALF_WIDTH - HALF_ROAD_WIDTH + HALF_CAR_LENGTH:
      if self.car_y < HALF_HEIGHT - HALF_ROAD_WIDTH + HALF_CAR_LENGTH:
        return self.car_x + self.car_y < 500
      elif self.car_y > HALF_HEIGHT + HALF_ROAD_WIDTH - HALF_CAR_LENGTH:
        return self.car_x - self.car_y < -100
    elif self.car_x > HALF_WIDTH + HALF_ROAD_WIDTH + HALF_CAR_LENGTH:
      if self.car_y < HALF_HEIGHT - HALF_ROAD_WIDTH + HALF_CAR_LENGTH:
        return self.car_x - self.car_y > 400
      elif self.car_y > HALF_HEIGHT + HALF_ROAD_WIDTH - HALF_CAR_LENGTH:
        return self.car_x + self.car_y > 1000
    # Check red light violation
    if action != 3:
      if lights_horizontal == 0:
        if self.car_origin == 0:
          return self.car_x > HALF_WIDTH and self.car_x < HALF_WIDTH + HALF_ROAD_WIDTH
        elif self.car_origin == 2:
          return self.car_x > HALF_WIDTH - HALF_ROAD_WIDTH and self.car_x < HALF_WIDTH
      elif lights_vertical == 0:
        if self.car_origin == 1:
          return self.car_y > HALF_HEIGHT - HALF_ROAD_WIDTH and self.car_y < HALF_HEIGHT
        elif self.car_origin == 3:
          return self.car_y > HALF_HEIGHT and self.car_y < HALF_HEIGHT + HALF_ROAD_WIDTH
    # Check intersection with oncoming lane
    out_of_lane = False
    if self.car_x < HALF_WIDTH - HALF_ROAD_WIDTH:
      out_of_lane = (self.car_y < HALF_HEIGHT) == (self.car_origin == 2)
    elif self.car_x > HALF_WIDTH + HALF_ROAD_WIDTH:
      out_of_lane = (self.car_y > HALF_HEIGHT) == (self.car_origin == 0)
    elif self.car_y < HALF_HEIGHT - HALF_ROAD_WIDTH:
      out_of_lane = (self.car_x > HALF_WIDTH) == (self.car_origin == 1)
    elif self.car_y > HALF_HEIGHT + HALF_ROAD_WIDTH:
      out_of_lane = (self.car_x < HALF_WIDTH) == (self.car_origin == 3)
    if out_of_lane:
      return True
    return False

  def check_success(self):
    if self.exited:
      return False
    # Check intersection with oncoming lane
    success = False
    goal_from_center = HALF_ROAD_WIDTH + QUARTER_ROAD_WIDTH if SETUP == 0 else HALF_ROAD_WIDTH
    if self.car_x < HALF_WIDTH - goal_from_center:
      success = self.car_y < HALF_HEIGHT and self.car_target == 2
    elif self.car_x > HALF_WIDTH + goal_from_center:
      success = self.car_y > HALF_HEIGHT and self.car_target == 0
    elif self.car_y < HALF_HEIGHT - goal_from_center:
      success = self.car_x > HALF_WIDTH and self.car_target == 1
    elif self.car_y > HALF_HEIGHT + goal_from_center:
      success = self.car_x < HALF_WIDTH and self.car_target == 3
    if success:
      self.car_speed = 0
      return True
    return False

  def check_reward(self, lights_horizontal, lights_vertical):
    # Make sure car doesn't accelerate at red light
    if (self.car_origin == 0 and lights_horizontal == 0) or\
      (self.car_origin == 1 and lights_vertical == 0) or\
      (self.car_origin == 2 and lights_horizontal == 0) or\
      (self.car_origin == 3 and lights_vertical == 0):
      return 1 if self.car_speed < 0.02 else -1
    elif self.car_speed < 0.02:
      return -1
    # Make sure car moves towards intersection
    if self.car_origin == 0 and self.car_x > HALF_WIDTH + QUARTER_ROAD_WIDTH:
      reward = (-self.car_x + WIDTH) / (HALF_WIDTH - HALF_ROAD_WIDTH)
    elif self.car_origin == 1 and self.car_y < HALF_HEIGHT - QUARTER_ROAD_WIDTH:
      reward = self.car_y / (HALF_HEIGHT - HALF_ROAD_WIDTH)
    elif self.car_origin == 2 and self.car_x < HALF_WIDTH - QUARTER_ROAD_WIDTH:
      reward = self.car_x / (HALF_WIDTH - HALF_ROAD_WIDTH)
    elif self.car_origin == 3 and self.car_y > HALF_HEIGHT + QUARTER_ROAD_WIDTH:
      reward = (-self.car_y + HEIGHT) / (HALF_HEIGHT - HALF_ROAD_WIDTH)
    # Make sure car turns towards target
    elif self.car_target == 0:
      reward = 1 + math.sqrt(HALF_WIDTH * HALF_WIDTH - HALF_HEIGHT * HALF_HEIGHT) / (abs(HALF_HEIGHT - self.car_y) + abs(WIDTH - self.car_x))
      reward += (90 - min(abs(self.wheel_angle % 360), 360 - abs(self.wheel_angle % 360))) / 20
      reward += (90 - min(abs(self.car_angle % 360), 360 - abs(self.car_angle % 360))) / 20
    elif self.car_target == 1:
      reward = 1 + math.sqrt(HALF_WIDTH * HALF_WIDTH - HALF_HEIGHT * HALF_HEIGHT) / (abs(HALF_WIDTH - self.car_x) + self.car_y)
      reward += (90 - min(abs(self.wheel_angle % 360 - 90), 360 - abs(self.wheel_angle % 360 - 90))) / 20
      reward += (90 - min(abs(self.car_angle % 360 - 90), 360 - abs(self.car_angle % 360 - 90))) / 20
    elif self.car_target == 2:
      reward = 1 + math.sqrt(HALF_WIDTH * HALF_WIDTH - HALF_HEIGHT * HALF_HEIGHT) / (abs(HALF_HEIGHT - self.car_y) + self.car_x)
      reward += (90 - min(abs(self.wheel_angle % 360 - 180), 360 - abs(self.wheel_angle % 360 - 180))) / 20
      reward += (90 - min(abs(self.car_angle % 360 - 180), 360 - abs(self.car_angle % 360 - 180))) / 20
    else:
      reward = 1 + math.sqrt(HALF_WIDTH * HALF_WIDTH - HALF_HEIGHT * HALF_HEIGHT) / (abs(HALF_WIDTH - self.car_x) + abs(HEIGHT - self.car_y))
      reward += (90 - min(abs(self.wheel_angle % 360 - 270), 360 - abs(self.wheel_angle % 360 - 270))) / 20
      reward += (90 - min(abs(self.car_angle % 360 - 270), 360 - abs(self.car_angle % 360 - 270))) / 20

    # Make sure car doesn't slow down unnecessarily
    if (self.car_origin == 0 and lights_horizontal == 2) or\
      (self.car_origin == 1 and lights_vertical == 2) or\
      (self.car_origin == 2 and lights_horizontal == 2) or\
      (self.car_origin == 3 and lights_vertical == 2):
      reward -= (SPEED_PENALTY / self.car_speed) ** 2
    return reward

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

class SimEnv(gym.Env):
  metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

  def __init__(self, render_mode=None):
    super().__init__()
    self.action_space = spaces.Discrete(4)
    shape = 26 if SETUP == 2 else 11
    self.observation_space = spaces.Box(low=0, high=1, shape=(shape,), dtype=np.float32)
    self._time = 0
    self._episode_ended = False
    self.render_mode = render_mode
    self.clock = None
    self.screen = None
    if render_mode in ('human', 'rgb_array'):
      pygame.init()
      pygame.display.init()
      self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

  def reset(self, seed=None, options=None):
    super().reset(seed=seed, options=options)
    self._time = 0
    self.lights_horizontal = 2 if SETUP == 0 else 0
    self.lights_vertical = 0 if SETUP == 0 else 2
    self.next_lights_vertical = True if SETUP == 0 else False
    self.next_switch_time = 7500
    car_origin = 2 if SETUP == 0 else np.random.choice((0, 1, 2, 3))
    remaining_spots = [0, 1, 2, 3]
    remaining_spots.remove(car_origin)
    car_target = 1 if SETUP == 0 else np.random.choice(remaining_spots)
    self.car1 = Car(HALF_WIDTH - ROAD_WIDTH, HALF_HEIGHT + QUARTER_ROAD_WIDTH, car_origin, car_target, self.render_mode, self.screen)
    if self.car1.car_origin == 0:
      self.car1.car_x = HALF_WIDTH + ROAD_WIDTH
      self.car1.car_y = HALF_HEIGHT - QUARTER_ROAD_WIDTH
    elif self.car1.car_origin == 1:
      self.car1.car_x = HALF_WIDTH - QUARTER_ROAD_WIDTH
      self.car1.car_y = HALF_HEIGHT - ROAD_WIDTH
    elif self.car1.car_origin == 3:
      self.car1.car_x = HALF_WIDTH + QUARTER_ROAD_WIDTH
      self.car1.car_y = HALF_HEIGHT + ROAD_WIDTH
    self.car1.car_speed = 0.12 if (self.lights_horizontal == 2 and (car_origin == 0 or car_origin == 2)) or (self.lights_vertical == 2 and (car_origin == 1 or car_origin == 3)) else 0
    self.cars = [self.car1,]
    if SETUP == 2:
      if 0 in remaining_spots:
        self.cars.append(Car(HALF_WIDTH + ROAD_WIDTH, HALF_HEIGHT - QUARTER_ROAD_WIDTH, 0, 2, self.render_mode, self.screen))
      if 1 in remaining_spots:
        self.cars.append(Car(HALF_WIDTH - QUARTER_ROAD_WIDTH, HALF_HEIGHT - ROAD_WIDTH, 1, 3, self.render_mode, self.screen))
      if 2 in remaining_spots:
        self.cars.append(Car(HALF_WIDTH - ROAD_WIDTH, HALF_HEIGHT + QUARTER_ROAD_WIDTH, 2, 0, self.render_mode, self.screen))
      if 3 in remaining_spots:
        self.cars.append(Car(HALF_WIDTH + QUARTER_ROAD_WIDTH, HALF_HEIGHT + ROAD_WIDTH, 3, 1, self.render_mode, self.screen))
    self._episode_ended = False

    observations = self.get_normalized_observation()
    if self.render_mode in ('human', 'rgb_array'):
      self.render()
    return observations, {}

  def step(self, action):
    for _ in range(FRAMES_PER_STEP):
      if self._episode_ended:
        return self.reset()

      # Make sure episodes don't go on forever.
      self._time += 1
      if self._time > TIME_LIMIT:
        self._score = -10000
        self._episode_ended = True
        break

      self.car1.process_ai_input(action)
      for car in self.cars:
        if car != self.car1:
          car.automatic_input(self.lights_horizontal, self.lights_vertical)

      for car in self.cars:
        car.update_position()

      if self.car1.check_collisions(self.cars, action, self.lights_horizontal, self.lights_vertical):
        self._score = -10000
        self._episode_ended = True
        break

      if self.car1.check_success():
        self._score = 200000
        self._score -= self._time * 20
        self._episode_ended = True
        break

      elif self.car1.exited:
        self._score = -10000
        self._episode_ended = True
        break

      if self._episode_ended:
        break

      if SETUP != 0 and self._time == self.next_switch_time:
        self.switch_lights()

    observations = self.get_normalized_observation()
    reward = 0
    if self._episode_ended:
      reward = self._score
    else:
      reward = self.car1.check_reward(self.lights_horizontal, self.lights_vertical)

    if self.render_mode in ('human', 'rgb_array') and self._time % 10 == 0:
      self.render()
    return (
      observations,
      reward,
      self._episode_ended,
      self._time > TIME_LIMIT,
      {},
    )

  def switch_lights(self):
    if self.lights_horizontal == 0 and self.lights_vertical == 2:
      self.lights_vertical = 1
      self.next_switch_time = self._time + 500
    elif self.lights_horizontal == 2 and self.lights_vertical == 0:
      self.lights_horizontal = 1
      self.next_switch_time = self._time + 500
    elif self.lights_horizontal == 0 and self.lights_vertical == 1 and not self.next_lights_vertical:
      self.lights_horizontal = 1
      self.lights_vertical = 0
      self.next_switch_time = self._time + 500
    elif self.lights_horizontal == 0 and self.lights_vertical == 1 and self.next_lights_vertical:
      self.lights_vertical = 2
      self.next_lights_vertical = False
      self.next_switch_time = self._time + 7500
    elif self.lights_horizontal == 1 and self.lights_vertical == 0 and self.next_lights_vertical:
      self.lights_horizontal = 0
      self.lights_vertical = 1
      self.next_switch_time = self._time + 500
    elif self.lights_horizontal == 1 and self.lights_vertical == 0 and not self.next_lights_vertical:
      self.lights_horizontal = 2
      self.next_lights_vertical = True
      self.next_switch_time = self._time + 7500

  def get_normalized_observation(self):
    # One-hot encode car_target (4 possible values)
    observations = np.array([self.car1.car_target == 0, self.car1.car_target == 1, self.car1.car_target == 2, self.car1.car_target == 3,
                             self.lights_horizontal / 2, self.lights_vertical / 2], dtype=np.float32)
    for car in self.cars:
      # Normalize positions, angles, speeds of cars
      observations = np.concatenate((observations, np.array([car.car_x / WIDTH, car.car_y / HEIGHT, car.car_angle % 360 / 360, car.wheel_angle % 360 / 360,
                                                             car.car_speed / MAX_SPEED], dtype=np.float32)))
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
    id='SimCar-v1',
    entry_point=SimEnv,
    max_episode_steps=TIME_LIMIT // FRAMES_PER_STEP + 1,
  )

  env = gym.make('SimCar-v1', render_mode='human')
  obs, _ = env.reset()

  if CHECK_ENV:
    check_env(env, warn=True)
    obs, _ = env.reset()

  env.render()

  for step in range(30000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
    if step % 10 == 0:
      env.render()
    if done:
      print("Goal reached!", "reward=", reward)
      break
  env.close()
