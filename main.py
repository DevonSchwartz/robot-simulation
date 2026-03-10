import asyncio
import math
import random
import sys
from dataclasses import dataclass

import pygame


# Window and world settings
WIDTH, HEIGHT = 1100, 700
FPS = 60
BALL_RADIUS = 10
BALL_COUNT = 10

# Robot settings
ROBOT_RADIUS = 18
ROBOT_SPEED = 2.1
ROBOT_TURN_SPEED = 0.045  # radians per frame
VISION_HIT_TOLERANCE = BALL_RADIUS + 6

# Bucket settings
BUCKET_WIDTH = 75
BUCKET_HEIGHT = 180
BUCKET_MARGIN = 20

# Colors
WHITE = (240, 240, 240)
FIELD_GREEN = (211, 232, 203)
BLACK = (20, 20, 20)
RED = (210, 40, 40)
BLUE = (45, 95, 225)
GRAY = (130, 130, 130)


@dataclass
class Ball:
    x: float
    y: float
    color: tuple[int, int, int]
    delivered: bool = False
    picked_up: bool = False


class Robot:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.heading = 0.0

        self.target_ball: Ball | None = None
        self.last_seen_ball: Ball | None = None
        self.carried_ball: Ball | None = None

        # Search state for "turn in a circle" behavior
        self.searching = False
        self.search_rotation = 0.0
        self.search_exhausted = False

    def vision_end(self) -> tuple[float, float]:
        # Extend line of sight to the edge of the field in the facing direction.
        max_dist = max(WIDTH, HEIGHT) * 2
        end_x = self.x + math.cos(self.heading) * max_dist
        end_y = self.y + math.sin(self.heading) * max_dist
        return end_x, end_y

    def detect_ball_on_vision_line(self, balls: list[Ball]) -> Ball | None:
        closest_ball = None
        closest_dist = float("inf")

        dir_x = math.cos(self.heading)
        dir_y = math.sin(self.heading)

        for ball in balls:
            if ball.delivered or ball.picked_up:
                continue

            vx = ball.x - self.x
            vy = ball.y - self.y

            # Projection length onto heading vector.
            proj = vx * dir_x + vy * dir_y
            if proj <= 0:
                continue

            # Perpendicular distance from ball center to vision ray.
            perp_x = vx - proj * dir_x
            perp_y = vy - proj * dir_y
            perp_dist = math.hypot(perp_x, perp_y)

            if perp_dist <= VISION_HIT_TOLERANCE and proj < closest_dist:
                closest_dist = proj
                closest_ball = ball

        return closest_ball

    def full_scan_for_any_ball(self, balls: list[Ball]) -> bool:
        # Quick omnidirectional fallback check to decide if there is anything left.
        return any(not b.delivered for b in balls)

    def update(self, balls: list[Ball], red_bucket: pygame.Rect, blue_bucket: pygame.Rect):
        # If carrying, deliver first before seeking another ball.
        if self.carried_ball is not None:
            bucket = red_bucket if self.carried_ball.color == RED else blue_bucket
            dx = bucket.centerx - self.x
            dy = bucket.centery - self.y
            dist = math.hypot(dx, dy)

            if dist > 1e-6:
                self.heading = math.atan2(dy, dx)

            if dist > ROBOT_SPEED:
                self.x += (dx / dist) * ROBOT_SPEED
                self.y += (dy / dist) * ROBOT_SPEED
                self.carried_ball.x = self.x
                self.carried_ball.y = self.y
            else:
                self.x = float(bucket.centerx)
                self.y = float(bucket.centery)
                self.carried_ball.x = float(bucket.centerx)
                self.carried_ball.y = float(bucket.centery)
                self.carried_ball.picked_up = False
                self.carried_ball.delivered = True
                self.carried_ball = None

            return

        # Validate existing target.
        if self.target_ball and (self.target_ball.delivered or self.target_ball.picked_up):
            self.target_ball = None

        if self.target_ball is None:
            seen = self.detect_ball_on_vision_line(balls)
            if seen is not None:
                self.target_ball = seen
                self.last_seen_ball = seen
                self.searching = False
                self.search_rotation = 0.0
                self.search_exhausted = False
            else:
                if self.search_exhausted:
                    return

                # Nothing directly in front: turn in a circle until a target is found.
                if not self.searching:
                    self.searching = True
                    self.search_rotation = 0.0

                self.heading += ROBOT_TURN_SPEED
                self.search_rotation += abs(ROBOT_TURN_SPEED)

                seen_while_turning = self.detect_ball_on_vision_line(balls)
                if seen_while_turning is not None:
                    self.target_ball = seen_while_turning
                    self.last_seen_ball = seen_while_turning
                    self.searching = False
                    self.search_rotation = 0.0
                    self.search_exhausted = False
                elif self.search_rotation >= 2 * math.pi:
                    # Full rotation complete and no target found: stay in place.
                    self.searching = False
                    self.search_rotation = 0.0
                    self.search_exhausted = True

                self.heading = self.heading % (2 * math.pi)
                return

        # Move toward target if present.
        if self.target_ball is not None:
            dx = self.target_ball.x - self.x
            dy = self.target_ball.y - self.y
            dist = math.hypot(dx, dy)

            if dist > 1e-6:
                desired_heading = math.atan2(dy, dx)
                self.heading = desired_heading

            if dist > ROBOT_SPEED:
                self.x += (dx / dist) * ROBOT_SPEED
                self.y += (dy / dist) * ROBOT_SPEED
            else:
                self.x = self.target_ball.x
                self.y = self.target_ball.y

                # Ball reached: pick it up, then move to bucket on subsequent frames.
                self.target_ball.picked_up = True
                self.carried_ball = self.target_ball
                self.target_ball = None

    def draw(self, surface: pygame.Surface):
        # Vision line
        end_x, end_y = self.vision_end()
        pygame.draw.line(surface, BLACK, (self.x, self.y), (end_x, end_y), 2)

        # Robot body
        pygame.draw.circle(surface, GRAY, (int(self.x), int(self.y)), ROBOT_RADIUS)
        pygame.draw.circle(surface, BLACK, (int(self.x), int(self.y)), ROBOT_RADIUS, 2)

        # Heading arrow
        arrow_len = ROBOT_RADIUS + 10
        tip_x = self.x + math.cos(self.heading) * arrow_len
        tip_y = self.y + math.sin(self.heading) * arrow_len
        pygame.draw.line(surface, BLACK, (self.x, self.y), (tip_x, tip_y), 3)


def random_spawn_points(count: int, forbidden_rects: list[pygame.Rect]) -> list[tuple[float, float]]:
    points = []
    attempts = 0

    while len(points) < count and attempts < 10000:
        attempts += 1
        x = random.randint(BUCKET_WIDTH + 50, WIDTH - BUCKET_WIDTH - 50)
        y = random.randint(40, HEIGHT - 40)

        # Avoid placing in forbidden zones.
        if any(r.collidepoint(x, y) for r in forbidden_rects):
            continue

        # Keep new balls from spawning on top of existing balls.
        if any(math.hypot(x - px, y - py) < BALL_RADIUS * 2.2 for px, py in points):
            continue

        points.append((x, y))

    return points


def create_balls(red_bucket: pygame.Rect, blue_bucket: pygame.Rect) -> list[Ball]:
    balls: list[Ball] = []
    points = random_spawn_points(BALL_COUNT, [red_bucket, blue_bucket])

    for i, (x, y) in enumerate(points):
        color = RED if i % 2 == 0 else BLUE
        balls.append(Ball(float(x), float(y), color))

    return balls


def draw_buckets(surface: pygame.Surface, red_bucket: pygame.Rect, blue_bucket: pygame.Rect):
    pygame.draw.rect(surface, RED, red_bucket, border_radius=8)
    pygame.draw.rect(surface, BLUE, blue_bucket, border_radius=8)

    pygame.draw.rect(surface, BLACK, red_bucket, 3, border_radius=8)
    pygame.draw.rect(surface, BLACK, blue_bucket, 3, border_radius=8)


def draw_balls(surface: pygame.Surface, balls: list[Ball]):
    for ball in balls:
        pygame.draw.circle(surface, ball.color, (int(ball.x), int(ball.y)), BALL_RADIUS)
        pygame.draw.circle(surface, BLACK, (int(ball.x), int(ball.y)), BALL_RADIUS, 1)


def draw_hud(surface: pygame.Surface, font: pygame.font.Font, balls: list[Ball], robot: Robot):
    delivered = sum(1 for b in balls if b.delivered)
    total = len(balls)

    if robot.target_ball is not None:
        status = "Tracking target"
    elif robot.carried_ball is not None:
        status = "Depositing"
    elif robot.searching:
        status = "Searching"
    else:
        status = "Idle"

    text = f"Delivered: {delivered}/{total}   |   Robot state: {status}"
    label = font.render(text, True, BLACK)
    surface.blit(label, (16, 12))


def setup_game():
    pygame.init()
    pygame.display.set_caption("Robot Ball Sorting Simulation")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22)

    red_bucket = pygame.Rect(
        BUCKET_MARGIN,
        HEIGHT // 2 - BUCKET_HEIGHT // 2,
        BUCKET_WIDTH,
        BUCKET_HEIGHT,
    )
    blue_bucket = pygame.Rect(
        WIDTH - BUCKET_MARGIN - BUCKET_WIDTH,
        HEIGHT // 2 - BUCKET_HEIGHT // 2,
        BUCKET_WIDTH,
        BUCKET_HEIGHT,
    )

    balls = create_balls(red_bucket, blue_bucket)
    robot = Robot(WIDTH / 2, HEIGHT / 2)

    return screen, clock, font, red_bucket, blue_bucket, balls, robot


def main_desktop():
    screen, clock, font, red_bucket, blue_bucket, balls, robot = setup_game()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        robot.update(balls, red_bucket, blue_bucket)

        screen.fill(FIELD_GREEN)
        draw_buckets(screen, red_bucket, blue_bucket)
        draw_balls(screen, balls)
        robot.draw(screen)
        draw_hud(screen, font, balls, robot)

        pygame.display.flip()
        clock.tick(FPS)


async def main_web():
    screen, _, font, red_bucket, blue_bucket, balls, robot = setup_game()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        robot.update(balls, red_bucket, blue_bucket)

        screen.fill(FIELD_GREEN)
        draw_buckets(screen, red_bucket, blue_bucket)
        draw_balls(screen, balls)
        robot.draw(screen)
        draw_hud(screen, font, balls, robot)

        pygame.display.flip()
        await asyncio.sleep(0)


def main():
    if sys.platform == "emscripten":
        asyncio.run(main_web())
    else:
        main_desktop()


if __name__ == "__main__":
    main()
