import pygame
import math
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WHEEL_INPUT = False
WIDTH, HEIGHT = 900, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Traffic Sim')
ROAD_WIDTH = 200
LIGHT_SIZE = 40
MAX_WHEEL_ANGLE = 45.0
MAX_SPEED = 0.3
WHEEL_TURN_SPEED = 0.1
ACCELERATION = 0.0001
FRICTION = 0.00003
SPEED_PENALTY = 0.02
CAR_WIDTH = 30
CAR_LENGTH = 50
HALF_WIDTH, HALF_HEIGHT, HALF_ROAD_WIDTH, QUARTER_ROAD_WIDTH, HALF_CAR_LENGTH, HALF_CAR_WIDTH = WIDTH // 2, HEIGHT // 2, ROAD_WIDTH // 2, ROAD_WIDTH // 4, CAR_LENGTH // 2, CAR_WIDTH // 2

# Colors (only for demo)
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

# Wheel input (only for demo)
if WHEEL_INPUT:
    pygame.joystick.init()
    print(pygame.joystick.get_count())
    joystick = pygame.joystick.Joystick(0)

class Car:
    def __init__(self, x, y, origin, target):
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
        # Surfaces (only for demo)
        self.car_surf = pygame.Surface((CAR_LENGTH, CAR_WIDTH)).convert_alpha()
        color = YELLOW if origin == 0 else GREEN if origin == 1 else BLUE if origin == 2 else RED
        self.car_surf.fill(color)
        self.wheel_surf = pygame.Surface((10, 5)).convert_alpha()
        self.wheel_surf.fill(WHITE)

    # Player input (only for demo)
    def process_player_input(self):
        if self.exited:
            return
        keys = pygame.key.get_pressed()
        if WHEEL_INPUT:
            steering = joystick.get_axis(0)
            self.wheel_angle = -steering * 45 + self.car_angle
        else:
            if keys[pygame.K_w]:
                self.car_speed += ACCELERATION
                self.car_speed = min(MAX_SPEED, self.car_speed)
            if keys[pygame.K_s]:
                self.car_speed -= ACCELERATION
                self.car_speed = max(0, self.car_speed)

            if keys[pygame.K_a] and self.wheel_angle - self.car_angle < MAX_WHEEL_ANGLE:
                self.wheel_angle += WHEEL_TURN_SPEED
            if keys[pygame.K_d] and self.wheel_angle - self.car_angle > -MAX_WHEEL_ANGLE:
                self.wheel_angle -= WHEEL_TURN_SPEED

        if self.car_speed > 0:
            self.car_speed -= FRICTION

    def process_ai_input(self, action):
        if self.exited:
            return
        if action == 0:
            self.car_speed += ACCELERATION
            self.car_speed = min(MAX_SPEED, self.car_speed)
        elif action == 1:
            self.car_speed -= ACCELERATION
            self.car_speed = max(0, self.car_speed)

        if self.car_speed > 0:
            self.car_speed -= FRICTION

        if action == 2 and self.wheel_angle - self.car_angle < MAX_WHEEL_ANGLE:
            self.wheel_angle += WHEEL_TURN_SPEED
        elif action == 3 and self.wheel_angle - self.car_angle > -MAX_WHEEL_ANGLE:
            self.wheel_angle -= WHEEL_TURN_SPEED

    def automatic_input(self, lights_horizontal, lights_vertical):
        if self.exited:
            return
        if ((self.car_origin == 0 or self.car_origin == 2) and lights_horizontal == 2) \
            or ((self.car_origin == 1 or self.car_origin == 3) and lights_vertical == 2):
            self.car_speed += ACCELERATION
            self.car_speed = min(MAX_SPEED, self.car_speed)
        else:
            self.car_speed -= ACCELERATION
            self.car_speed = max(0, self.car_speed)

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

    def check_collisions(self, other_cars, lights_horizontal, lights_vertical):
        if self.exited:
            return
        # Check car collisions
        for car in other_cars:
            if (not car.exited) and self != car and math.sqrt(((self.car_x - car.car_x)**2) + ((self.car_y - car.car_y)**2)) <= CAR_LENGTH:
                #print('Car crash')
                self.car_speed = 0
                car.car_speed = 0
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
        if self.car_x > HALF_WIDTH - HALF_ROAD_WIDTH and self.car_x < HALF_WIDTH + HALF_ROAD_WIDTH \
            and self.car_y > HALF_HEIGHT - HALF_ROAD_WIDTH and self.car_y < HALF_HEIGHT + HALF_ROAD_WIDTH \
            and ((lights_horizontal == 0 and (self.car_origin == 0 or self.car_origin == 2)) or ((lights_vertical == 0 and (self.car_origin == 1 or self.car_origin == 3)))):
            #print('Red light violation')
            return True
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
            #print('Lane violation')
            return True
        return False

    def check_success(self):
        if self.exited:
            return False
        # Check intersection with oncoming lane
        success = False
        if self.car_x < HALF_WIDTH - HALF_ROAD_WIDTH - QUARTER_ROAD_WIDTH:
            success = self.car_y < HALF_HEIGHT and self.car_target == 2
        elif self.car_x > HALF_WIDTH + HALF_ROAD_WIDTH + QUARTER_ROAD_WIDTH:
            success = self.car_y > HALF_HEIGHT and self.car_target == 0
        elif self.car_y < HALF_HEIGHT - HALF_ROAD_WIDTH - QUARTER_ROAD_WIDTH:
            success = self.car_x > HALF_WIDTH and self.car_target == 1
        elif self.car_y > HALF_HEIGHT + HALF_ROAD_WIDTH + QUARTER_ROAD_WIDTH:
            success = self.car_x < HALF_WIDTH and self.car_target == 3
        if success:
            self.car_speed = 0
            print('Success!')
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

    # Drawing (only for demo)
    def draw(self):
        if self.exited:
            return
        # Draw the car
        rotated_car = pygame.transform.rotate(self.car_surf, self.car_angle)
        rotated_car_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        SCREEN.blit(rotated_car, rotated_car_rect)
        # Draw the wheels
        rotated_front_wheel = pygame.transform.rotate(self.wheel_surf, self.wheel_angle)
        rotated_front_wheel_1_rect = rotated_front_wheel.get_rect(center=(self.front_wheel_x + (HALF_CAR_WIDTH * math.sin(self.car_angle_rad)), self.front_wheel_y + (HALF_CAR_WIDTH * math.cos(self.car_angle_rad))))
        rotated_front_wheel_2_rect = rotated_front_wheel.get_rect(center=(self.front_wheel_x - (HALF_CAR_WIDTH * math.sin(self.car_angle_rad)), self.front_wheel_y - (HALF_CAR_WIDTH * math.cos(self.car_angle_rad))))
        rotated_rear_wheel = pygame.transform.rotate(self.wheel_surf, self.car_angle)
        rotated_rear_wheel_1_rect = rotated_rear_wheel.get_rect(center=(self.rear_wheel_x + (HALF_CAR_WIDTH * math.sin(self.car_angle_rad)), self.rear_wheel_y + (HALF_CAR_WIDTH * math.cos(self.car_angle_rad))))
        rotated_rear_wheel_2_rect = rotated_rear_wheel.get_rect(center=(self.rear_wheel_x - (HALF_CAR_WIDTH * math.sin(self.car_angle_rad)), self.rear_wheel_y - (HALF_CAR_WIDTH * math.cos(self.car_angle_rad))))
        SCREEN.blit(rotated_front_wheel, rotated_front_wheel_1_rect)
        SCREEN.blit(rotated_front_wheel, rotated_front_wheel_2_rect)
        SCREEN.blit(rotated_rear_wheel, rotated_rear_wheel_1_rect)
        SCREEN.blit(rotated_rear_wheel, rotated_rear_wheel_2_rect)

car1 = Car(HALF_WIDTH - ROAD_WIDTH, HALF_HEIGHT + QUARTER_ROAD_WIDTH, 2, np.random.choice((0, 1, 3)))
print(car1.car_target)
car2 = Car(HALF_WIDTH - QUARTER_ROAD_WIDTH, 50, 1, 3)
car3 = Car(HALF_WIDTH + QUARTER_ROAD_WIDTH, HEIGHT - 50, 3, 1)
car4 = Car(HALF_WIDTH + QUARTER_ROAD_WIDTH, HEIGHT - 150, 3, 1)
car5 = Car(WIDTH - 50, HALF_HEIGHT - QUARTER_ROAD_WIDTH, 0, 2)
cars = [car1, car2, car3, car4, car5]
lights_horizontal, lights_vertical = 0, 2
next_lights_vertical = False
CHANGE_LIGHTS_1 = pygame.USEREVENT + 1
CHANGE_LIGHTS_2 = pygame.USEREVENT + 2
CHANGE_LIGHTS_3 = pygame.USEREVENT + 3
CHANGE_LIGHTS_4 = pygame.USEREVENT + 4
pygame.time.set_timer(CHANGE_LIGHTS_1, 10000)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == CHANGE_LIGHTS_1:
            if next_lights_vertical:
                lights_horizontal = 1
            else:
                lights_vertical = 1
            pygame.time.set_timer(CHANGE_LIGHTS_2, 1000)
            pygame.time.set_timer(CHANGE_LIGHTS_3, 0)
            pygame.time.set_timer(CHANGE_LIGHTS_4, 0)
        elif event.type == CHANGE_LIGHTS_2:
            if next_lights_vertical:
                lights_horizontal = 0
            else:
                lights_vertical = 0
            pygame.time.set_timer(CHANGE_LIGHTS_2, 0)
            pygame.time.set_timer(CHANGE_LIGHTS_3, 1000)
            pygame.time.set_timer(CHANGE_LIGHTS_4, 0)
        elif event.type == CHANGE_LIGHTS_3:
            if next_lights_vertical:
                lights_vertical = 1
            else:
                lights_horizontal = 1
            pygame.time.set_timer(CHANGE_LIGHTS_2, 0)
            pygame.time.set_timer(CHANGE_LIGHTS_3, 0)
            pygame.time.set_timer(CHANGE_LIGHTS_4, 1000)
        elif event.type == CHANGE_LIGHTS_4:
            if next_lights_vertical:
                lights_vertical = 2
            else:
                lights_horizontal = 2
            pygame.time.set_timer(CHANGE_LIGHTS_2, 0)
            pygame.time.set_timer(CHANGE_LIGHTS_3, 0)
            pygame.time.set_timer(CHANGE_LIGHTS_4, 0)
            next_lights_vertical = not next_lights_vertical

    car1.process_player_input()
    for car in [car2, car3, car4, car5]:
        car.automatic_input(lights_horizontal, lights_vertical)
    
    for car in cars:
        car.update_position()

    if car1.check_collisions(cars, lights_horizontal, lights_vertical):
        car1.car_speed = 0
    if car1.check_success():
        running = False
    reward = car1.check_reward(lights_horizontal, lights_vertical)
    print(reward)

    # Clear the screen
    SCREEN.fill(BLACK)

    # Draw sidewalks
    pygame.draw.rect(SCREEN, GRAY, pygame.Rect(0, 0, (WIDTH - ROAD_WIDTH) // 2, (HEIGHT - ROAD_WIDTH) // 2))
    pygame.draw.rect(SCREEN, GRAY, pygame.Rect((WIDTH + ROAD_WIDTH) // 2, 0, (WIDTH - ROAD_WIDTH) // 2, (HEIGHT - ROAD_WIDTH) // 2))
    pygame.draw.rect(SCREEN, GRAY, pygame.Rect(0, (HEIGHT + ROAD_WIDTH) // 2, (WIDTH - ROAD_WIDTH) // 2, (HEIGHT - ROAD_WIDTH) // 2))
    pygame.draw.rect(SCREEN, GRAY, pygame.Rect((WIDTH + ROAD_WIDTH) // 2, (HEIGHT + ROAD_WIDTH) // 2, (WIDTH - ROAD_WIDTH) // 2, (HEIGHT - ROAD_WIDTH) // 2))
    # Draw road markings
    for x in range(0, (WIDTH - ROAD_WIDTH) // 2, 100):
        pygame.draw.line(SCREEN, WHITE, (x, HALF_HEIGHT), (x + 50, HALF_HEIGHT), 5)
    for x in range((WIDTH + ROAD_WIDTH) // 2 + 50, WIDTH, 100):
        pygame.draw.line(SCREEN, WHITE, (x, HALF_HEIGHT), (x + 50, HALF_HEIGHT), 5)
    for y in range(50, (HEIGHT - ROAD_WIDTH) // 2, 100):
        pygame.draw.line(SCREEN, WHITE, (HALF_WIDTH, y), (HALF_WIDTH, y + 50), 5)
    for y in range((HEIGHT + ROAD_WIDTH) // 2 + 50, HEIGHT, 100):
        pygame.draw.line(SCREEN, WHITE, (HALF_WIDTH, y), (HALF_WIDTH, y + 50), 5)
    # Draw lights
    pygame.draw.rect(SCREEN, BLACK, pygame.Rect((WIDTH + ROAD_WIDTH) // 2, (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 3), LIGHT_SIZE, LIGHT_SIZE * 3))
    pygame.draw.circle(SCREEN, RED if lights_vertical == 0 else DARK_RED, ((WIDTH + ROAD_WIDTH) // 2 + (LIGHT_SIZE // 2), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 2.5)), 15)
    pygame.draw.circle(SCREEN, YELLOW if lights_vertical == 1 else DARK_YELLOW, ((WIDTH + ROAD_WIDTH) // 2 + (LIGHT_SIZE // 2), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 1.5)), 15)
    pygame.draw.circle(SCREEN, GREEN if lights_vertical == 2 else DARK_GREEN, ((WIDTH + ROAD_WIDTH) // 2 + (LIGHT_SIZE // 2), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 0.5)), 15)
    pygame.draw.rect(SCREEN, BLACK, pygame.Rect((WIDTH - ROAD_WIDTH) // 2 - LIGHT_SIZE, (HEIGHT + ROAD_WIDTH) // 2, LIGHT_SIZE, LIGHT_SIZE * 3))
    pygame.draw.circle(SCREEN, RED if lights_vertical == 0 else DARK_RED, ((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE // 2), (HEIGHT + ROAD_WIDTH) // 2 + (LIGHT_SIZE * 2.5)), 15)
    pygame.draw.circle(SCREEN, YELLOW if lights_vertical == 1 else DARK_YELLOW, ((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE // 2), (HEIGHT + ROAD_WIDTH) // 2 + (LIGHT_SIZE * 1.5)), 15)
    pygame.draw.circle(SCREEN, GREEN if lights_vertical == 2 else DARK_GREEN, ((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE // 2), (HEIGHT + ROAD_WIDTH) // 2 + (LIGHT_SIZE * 0.5)), 15)
    pygame.draw.rect(SCREEN, BLACK, pygame.Rect((WIDTH + ROAD_WIDTH) // 2, (HEIGHT + ROAD_WIDTH) // 2, LIGHT_SIZE * 3, LIGHT_SIZE))
    pygame.draw.circle(SCREEN, RED if lights_horizontal == 0 else DARK_RED, ((WIDTH + ROAD_WIDTH) // 2 + (LIGHT_SIZE * 2.5), (HEIGHT + ROAD_WIDTH) // 2 + (LIGHT_SIZE // 2)), 15)
    pygame.draw.circle(SCREEN, YELLOW if lights_horizontal == 1 else DARK_YELLOW, ((WIDTH + ROAD_WIDTH) // 2 + (LIGHT_SIZE * 1.5), (HEIGHT + ROAD_WIDTH) // 2 + (LIGHT_SIZE // 2)), 15)
    pygame.draw.circle(SCREEN, GREEN if lights_horizontal == 2 else DARK_GREEN, ((WIDTH + ROAD_WIDTH) // 2 + (LIGHT_SIZE * 0.5), (HEIGHT + ROAD_WIDTH) // 2 + (LIGHT_SIZE // 2)), 15)
    pygame.draw.rect(SCREEN, BLACK, pygame.Rect((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 3), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE), LIGHT_SIZE * 3, LIGHT_SIZE))
    pygame.draw.circle(SCREEN, RED if lights_horizontal == 0 else DARK_RED, ((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 2.5), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE // 2)), 15)
    pygame.draw.circle(SCREEN, YELLOW if lights_horizontal == 1 else DARK_YELLOW, ((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 1.5), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE // 2)), 15)
    pygame.draw.circle(SCREEN, GREEN if lights_horizontal == 2 else DARK_GREEN, ((WIDTH - ROAD_WIDTH) // 2 - (LIGHT_SIZE * 0.5), (HEIGHT - ROAD_WIDTH) // 2 - (LIGHT_SIZE // 2)), 15)

    for car in cars:
        car.draw()

    pygame.display.flip()

# Quit Pygame
pygame.quit()
