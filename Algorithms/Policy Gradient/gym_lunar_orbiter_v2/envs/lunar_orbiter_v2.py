import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Firing side engine is -0.03 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# To see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v2
#


FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 25.0
SIDE_ENGINE_POWER = 1.0

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [
    (-10, +10), (-10, 0), (-10, -20),      # define the shape of body
    (+10, -20), (+10, 0), (+10, +10), (0, 30)       # define the shape of body
]
#
# LANDER_POLY = [
#     (-14, +17), (-17, 0), (-17, -10),
#     (+17, -10), (+17, 0), (+14, +17)
# ]

"""##################### define the fire shape ####################"""
LEG_AWAY = 7                              # the horizontal distance between leg and body
LEG_DOWN = 30                              # the vertical distance between leg and body
LEG_W, LEG_H = 2, 7                        # the shape of legs
LEG_SPRING_TORQUE = 70                     # spring torque

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

# LEG_AWAY = 10
# LEG_DOWN = 18
# LEG_W, LEG_H = 2, 8
# LEG_SPRING_TORQUE = 40
#
# SIDE_ENGINE_HEIGHT = 14.0
# SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 1000                           # GUI window width
VIEWPORT_H = 500                            # GUI window height


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class LunarOrbiterV2(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    continuous = False

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.particles = []

        self.prev_reward = None
        """################## define flag to control distance reward ##################### """
        self.flag_to_control_1 = True
        self.flag_to_control_2 = True
        self.flag_to_control_3 = True
        """################## define flag to control distance reward ##################### """

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None

        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE                                            # window_width / scale = 600 / 30 = 20
        H = VIEWPORT_H / SCALE                                            # window_height / scale = 400 /30 = 13

        # terrain
        CHUNKS = 15                                                       # the number of vertex on surface
        height = self.np_random.uniform(0, H / 3, size=(CHUNKS + 1,))     # vertex's y-axis position (randomly generate)
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]           # vertex's x-axis position

        # interval length
        self.interval_length = chunk_x[4] - chunk_x[3]

        self.helipad_x1 = chunk_x[CHUNKS // 2 - 0]                        # define the x-position of left flag
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 0]                        # define the y-position of right flag
        self.helipad_y = H / 4                                            # define the y-position of flag-bottom

        smooth_y = [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)]

        """########################### change the position of flag ##############################"""
        # index = np.random.randint(1, CHUNKS-2)
        index = np.random.randint(1, CHUNKS-2)                      # constraint the position of flag, don't let it be on the most left and most right position
        self.helipad_one_x = chunk_x[index] + (chunk_x[index+1] - chunk_x[index])/2
        self.helipad_one_y = smooth_y[index]
        smooth_y[index+1] = self.helipad_one_y
        """########################### change the position of flag ##############################"""

        self.moon = self.world.CreateStaticBody(fixtures=fixtureDef(shape=edgeShape(vertices=[(0, 0), (W, 0)])))     # ???
        self.moon_polys = []
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])                                # use chunk_x and smooth_y to generate points
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(
                vertices=[p1, p2],                                        # use 2-neighbor points to create ground ???
                density=0,                                                # density of the ground
                friction=0.1)                                             # friction of the ground
            self.moon_polys.append([p1, p2, (p2[0], 0), (p1[0], 0)])
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])       # sky-poly shape, 4 points

        initial_y = VIEWPORT_H / SCALE                                    # starting altitude of the lander

        """########################### Create Earth, Stars, Smoke ##############################"""
        self._create_stars_large()
        self._create_stars_small()
        """########################### Create Earth, Stars ##############################"""

        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W / SCALE / 2, initial_y),                 # starting point: (middle, top)
            angle=0.0,                                                    # angle: no idea
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,                                      # the category the lander belongs to
                maskBits=0x001,                                           # the category the lander with collide with
                restitution=0.99)                                         # 0.99 bouncy
        )
        self.lander.color1 = (160/255, 160/255, 160/255)                              # filling color for lander
        self.lander.color2 = (160/255, 160/255, 160/255)                              # border color for lander
        self.lander.ApplyForceToCenter((                                  # not understood
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        ), True)
        # """
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y), # x <-> ; y <-> lander's "middle"
                angle=(i * 0.20),                                         # control the angle between legs and body
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,                                  # the category the legs belongs to
                    maskBits=0x001)                                       # the category the legs with collide with
            )
            leg.ground_contact = False
            leg.color1 = (1, 0, 0)                                  # filling color for legs
            leg.color2 = (1, 0, 0)                                  # border color for legs
            rjd = revoluteJointDef(                                       # para about joints between lander and legs
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.6                     # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.36
            else:
                rjd.lowerAngle = -0.36
                rjd.upperAngle = -0.9 + 0.6                              # angle between legs and body
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)
        # """
        """
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)
        """
        self.drawlist = [self.lander] + self.legs

        return self.step(np.array([0, 0]) if self.continuous else 0)[0]   # not understood

    def _create_particle(self, mass, x, y, ttl):                          # particle is the stuff come out of the engine
        p = self.world.CreateDynamicBody(
            position=(x, y-0.5),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=3 / SCALE, pos=(0, 0)),          # define the size of the particle
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,                                           # collide only with ground
                restitution=0.3)
        )
        p.ttl = ttl                                                       # not understood yet ??? what is ttl ???
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):                                                   # when to clean particles, i.e. how long particles last
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    """########################### Create Stars ##############################"""
    def _create_stars_large(self):
        self.stars_poly_large = []
        numOfStars = 200; W = VIEWPORT_W/SCALE; H = VIEWPORT_H / SCALE
        position_y = np.random.uniform(0, H, size=(numOfStars,))
        position_x = np.random.uniform(-W, W, size=(numOfStars,))
        for i in range(0, 200):
            self.stars_poly_large.append((position_x[i], position_y[i]))
        return self.stars_poly_large

    def _create_stars_small(self):
        self.stars_poly_small = []
        numOfStars = 500; W = VIEWPORT_W/SCALE; H = VIEWPORT_H / SCALE
        position_y = np.random.uniform(0, H, size=(numOfStars,))
        position_x = np.random.uniform(-W, W, size=(numOfStars,))
        for i in range(0, 500):
            self.stars_poly_small.append((position_x[i], position_y[i]))
        return self.stars_poly_small
    """########################### Create Star s##############################"""

    def step(self, action):                                                                # physical engine
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)                            # constrain value to be between (-1, +1)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]        # not understood

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5                       # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:                                                                          # if continuous, engine power (0.5, 1.0)
                m_power = 1.0                                                              # if not, engine power = 1
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]        # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)     # not understood: ox, oy
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1], m_power)        # maybe ox, oy are the effect brought by engine
            # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power), impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):    # try to understand this
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                           self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power), impulse_pos,
                                 True)
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)                                   # world.Step(time_step, velocity_iter, position_iter)

        # new landing point
        # self.new_landing_point_x = self.helipad_one_x - self.interval_length/2
        self.new_landing_point_x = self.helipad_one_x
        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            # (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),             # what "VIEWPORT_W / SCALE / 2" exactly is ? ~ = W/2
            (pos.x - self.new_landing_point_x) / (VIEWPORT_W / SCALE / 2),
            # (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            (pos.y - (self.helipad_one_y + LEG_DOWN / SCALE + 0.2)) / (VIEWPORT_H / SCALE / 2),

            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,                                  # velocity in x-axis
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,                                  # velocity in y-axis
            self.lander.angle,                                                       # angle
            20.0 * self.lander.angularVelocity / FPS,                                #
            1.0 if self.legs[0].ground_contact else 0.0,                             # check if left leg contact ground of not
            1.0 if self.legs[1].ground_contact else 0.0                              # check if right leg contact ground of not
        ]
        assert len(state) == 8

        reward = 0                                                                   # define reward...
        shaping = \
            - 150 * np.sqrt(state[0] * state[0] + state[1] * state[1]) \
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) \
            - 80 * abs(state[4]) + 10 * state[6] + 10 * state[7]
        # change the reward for s[4], -10*abs(state[4]) ???

        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * 0.03                                                    # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power * 0.003                                                   # -0.3 for firing main engine, -0.03 for firing side engine

        done = False
        if self.game_over or abs(state[0]) >= 1.0:                                  # game over or lander go out of window
            done = True
            reward = -200

        """########################## flag to control distance reward ############################"""
        if abs(state[0]) >= 0.2:                                        # if not close enough, give penalty for every time-step
            reward -= 0.2

        if self.flag_to_control_1:                           # 0.1 <= x < 0.20
            if 0.1 <= abs(state[0]) < 0.20 and 0.07 <= abs(state[1]) < 0.15:
                reward = +70
                self.flag_to_control_1 = False

        if self.flag_to_control_2:                           # 0.03 <= x < 0.1
            if 0.01 <= abs(state[0]) < 0.1 and 0.01 <= abs(state[1]) < 0.07:
                reward = +130
                self.flag_to_control_2 = False

        if self.flag_to_control_3:
            if abs(state[0]) < 0.01 and abs(state[1]) < 0.01:
                reward = +200
                self.flag_to_control_3 = False
        """########################## flag to control distance reward ############################"""
        if abs(state[0]) == 0 and abs(state[1]) == 0:
            reward = +300                                           # for my rocket, it's always bouncing, so give it reward when it close enough
            done = True
        if not self.lander.awake:
            done = True
            reward = +100                                       # give rocket reward when it stop bouncing

        return np.array(state, dtype=np.float32), reward, done, {}                  # what is "{}" return here

    def render(self, mode='human'):                                                 # graphic engine
        from gym.envs.classic_control import rendering

        if self.viewer is None:                                                     # create window viewer
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            # obj.color1 = (max(0.1, 0.1 + obj.ttl), max(0.1, 0.1 * obj.ttl), max(0.1, 0.1 * obj.ttl))
            obj.color2 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0.0, 0.0, 0.0))                           # define the color of the background sky

        """"################################## create stars ########################################"""
        # draw large stars
        for pos in self.stars_poly_large:
            self.viewer.draw_polygon(v=((pos[0], pos[1]),
                                        (pos[0] + 0.05, pos[1]),
                                        (pos[0] + 0.05, pos[1] + 0.05),
                                        (pos[0], pos[1] + 0.05)),
                                     color=(1.0, 1.0, 1.0))

        # draw small stars
        for pos in self.stars_poly_small:
            self.viewer.draw_polygon(v=((pos[0], pos[1]),
                                        (pos[0] + 0.02, pos[1]),
                                        (pos[0] + 0.02, pos[1] + 0.02),
                                        (pos[0], pos[1] + 0.02)),
                                     color=(1.0, 0.9, 0.9))

        # draw "earth"
        self.viewer.draw_circle_earth(radius=2.5, res=100, color=(0.0, 0.40, 0.80))
        self.viewer.draw_polygon(v=((-0.02+10, 0+10.3),
                                    (-1.0+10, 1.4+10.3),
                                    (1.1+10, 1.5+10.3),
                                    (0+10, 0+10.3),
                                    (1.2+10, -0.5+10.3),
                                    (-0.3+10, -2+10.3)), color=(50/255, 200/255, 50/255))

        """"################################## create stars ########################################"""

        """################################# Change the color of moon terrain ####################################"""
        """ if do this, then moon terrain is not static body"""
        for p in self.moon_polys:
            self.viewer.draw_polygon(p, color=(169/255, 169/255, 169/255))
        """################################# Change the color of moon terrain ####################################"""

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:                                                 # what does this for loop try to do ???
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flag_y1 = self.helipad_one_y; flag_y2 = self.helipad_one_y + 70/SCALE
        flag_x = self.helipad_one_x
        self.viewer.draw_polyline(v=((flag_x, flag_y1), (flag_x, flag_y2)), color=(1, 1, 1))
        self.viewer.draw_polygon(v=((flag_x, flag_y2),
                                    (flag_x, flag_y2-25/SCALE),
                                    (flag_x+40/SCALE, flag_y2-25/SCALE),
                                    (flag_x+40/SCALE, flag_y2)),
                                 color=(178/255, 34/255, 34/255))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class LunarOrbiterV2Continuous(LunarOrbiterV2):
    continuous = True


def heuristic(env, s):
    # Heuristic for:
    # 1. Testing.
    # 2. Demonstration rollout.
    angle_targ = s[0] * 2.5 + s[2] * 1.0                    # original: s[0] * 0.5; now: s[0] * 2.5; works perfect
    if angle_targ > 0.4: angle_targ = 0.4                   # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55 * np.abs(s[0])                        # target y should be proportional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

    if s[6] or s[7]:                                        # legs have contact
        angle_todo = 0
        hover_todo = -(s[3]) * 0.5                          # override to reduce fall speed, that's all we need after contact

    if env.continuous:                                      # what's doing here ???
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


def demo_heuristic_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        total_reward += r

        if render:
            still_open = env.render()
            if still_open == False: break

        if steps % 20 == 0 or done:                                # print state info every 20 steps
            print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
    return total_reward


if __name__ == '__main__':
    demo_heuristic_lander(LunarOrbiterV2(), render=True)

