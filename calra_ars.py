import datetime
import glob
import math
import os
import random
import sys
import time

import cv2
import gym
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class HP:
    def __init__(self,
                 nb_steps=100,
                 episode_length=2000,
                 learning_rate=0.02,
                 num_deltas=16,
                 num_best_deltas=16,
                 noise=0.03,
                 seed=1,
                 env_name='',
                 record_every=50):
        self.nb_steps = nb_steps
        self.episode_length = episode_length
        self.learning_rate = learning_rate
        self.num_deltas = num_deltas
        self.num_best_deltas = num_best_deltas
        assert self.num_best_deltas <= self.num_deltas
        self.noise = noise
        self.seed = seed
        self.env_name = env_name
        self.record_every = record_every


class Policy:
    def __init__(self, input_size, output_size, hp, theta=None):
        self.input_size = input_size
        self.output_size = output_size
        if theta is not None:
            self.theta = theta
        else:
            self.theta = np.zeros((output_size, input_size))
        self.hp = hp

    def evaluate(self, inputs, delta=None, direction=None):
        if direction is None:
            return self.theta.dot(inputs)
        elif direction == "+":
            return (self.theta + self.hp.noise * delta).dot(inputs)
        elif direction == "-":
            return (self.theta - self.hp.noise * delta).dot(inputs)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.num_deltas)]

    # update weights according to biggest reward

    def update(self, rollouts, sigma_rewards):
        # sigma_rewards is the standard deviation of the rewards
        old_theta = self.theta.copy()
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
            step += (r_pos - r_neg) * delta
        theta_update = self.hp.learning_rate / (self.hp.num_best_deltas * sigma_rewards) * step
        self.theta += theta_update
        if np.array_equal(old_theta, self.theta):
            print("Theta did not change.")


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class CarEnv:
    def __init__(self,
                 img_width=224,
                 img_height=224,
                 show_cam=False,
                 control_type='continuous',
                 car_model='model3',
                 seconds_per_episode=15
                 ):
        self.img_width = img_width
        self.img_height = img_height
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.car = self.blueprint_library.filter(car_model)[0]
        self.show_cam = show_cam
        self.control_type = control_type
        self.front_camera = None
        self.actor_list = []
        self.seconds_per_episode = seconds_per_episode

        if self.control_type == 'continuous':
            self.action_space = np.array(['throttle', 'steer', 'brake'])

    def reset(self):
        self.collision_hist = []
        self.steering_cache = []

        if len(self.actor_list) > 0:
            for actor in self.actor_list:
                actor.destroy()
        self.actor_list = []

        try:
            self.transform = random.choice(self.world.get_map().get_spawn_points())
            self.vehicle = self.world.spawn_actor(self.car, self.transform)
            self.actor_list.append(self.vehicle)
        except:
            self.reset()

        # Attach RGB Camera
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.img_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.img_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        # This establishes location on the vehicle that the sensor attaches to, we are
        # aiming for the hood in this case.
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        # Spawn and attach a collision sensor to the vehicle
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # This loop will wait until the camera is functioning before continuing
        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    def collision_data(self, event):
        self.collision_hist.append(event)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.img_height, self.img_width, 4))
        i3 = i2[:, :, :3]
        if self.show_cam:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action, step_num):
        # Continuous action space for use with ARS
        if self.control_type == 'continuous':
            self.vehicle.apply_control(carla.VehicleControl(throttle=np.clip(action[0], 0.0, 1.0),
                                                            steer=np.clip(action[1], -1.0, 1.0),
                                                            brake=np.clip(action[2], 0.0, 1.0)))

        # Append current steering control to cache to punish high values. Cleared with reset
        # at start of episode.
        self.steering_cache.append(action[1])
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))

        # Reward System:
        if len(self.collision_hist) != 0:
            # Check to see if on first step (for rough spawns by Carla)
            # Disregards collisions registered on first step of episode
            if step_num == 1:
                self.collision_hist = []
                done = False
                reward = 0
            else:
                done = True
                print('Collision!')
                reward = -200
        elif kmh < 60 & kmh > 0.2:
            done = False
            reward = 1
            # Reward lighter steering when moving
            if np.abs(action[1]) < 0.3:
                reward += 9
            elif np.abs(action[1]) > 0.5 and np.abs(action[1]) < 0.9:
                reward -= 1
            elif np.abs(action[1]) >= 0.9:
                reward -= 6
        elif kmh <= 0.2:
            done = False
            reward = -10
        else:
            done = False
            reward = 20
            # Reward lighter steering when moving
            if np.abs(action[1]) < 0.3:
                reward += 20
            # Reduce score for heavy steering
            if np.abs(action[1]) > 0.5 and np.abs(action[1]) < 0.9:
                reward -= 10
            elif np.abs(action[1]) >= 0.9:
                reward -= 20

        # Penalize consistent and heavily directional steering
        reward -= (np.abs(np.mean(self.steering_cache)) + np.abs(action[1])) * 10 / 2

        # Terminate episode after set number of seconds
        if self.episode_start + self.seconds_per_episode < time.time():
            done = True

        return self.front_camera, reward, done, None


# Generate a test car
test_car = CarEnv()

# Generate images for testing VGG19 predictions (kept small for example)
train_images = np.array([test_car.reset().reshape(224, 224, 3) for i in range(1)])

#train_images.shape

# An experiment with flattening shapes
train_imgs2 = train_images.reshape(train_images.shape[0], -1)
#train_imgs2.shape



for actor in test_car.actor_list:
    actor.destroy()
del test_car


# base_model = VGG19(weights='imagenet',
#                    include_top=False,
#                    input_shape=(224, 224, 3))
#
#
# # Testing to determine output shape of VGG19
# prediction = base_model.predict(train_images[0].reshape(1, 224, 224, 3) / 255.)
# prediction.shape


# We can see that the CNN has produced 512 7x7 convolutions. Let's look at some
# fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# for i in range(4):
#     col = i % 2
#     row = i // 2
#     axes[row, col].imshow(prediction[0][:, :, i])
#
# prediction.flatten().shape
# print('Maximum output value:', np.max(prediction))
# print('Minimum output value:', np.min(prediction))
#
# plt.style.use('ggplot')
# plt.figure(figsize=(11, 7))
# plt.title('Distribution of CNN output values')
# plt.xlabel('Output Value')
# plt.ylabel('Density')
# plt.hist(prediction.flatten());

# some initial weights for the ARS process to start with.
class ARSAgent:
    def __init__(self,
                 hp=None,
                 env=None,
                 base_model=True,
                 policy=None,
                 weights_dir='ars_weights',
                 initial_train=False
                 ):

        self.hp = hp or HP()
        np.random.seed(self.hp.seed)
        self.env = env or CarEnv(control_type='continuous')
        self.output_size = self.env.action_space.shape[0]
        self.record_video = False
        self.history = {'step': [],
                        'score': [],
                        'theta': []}
        self.generate_theta = False
        self.historical_steps = 0

        if base_model is None:
            self.input_size = self.env.front_camera.shape
        else:
            base_model = VGG19(weights='imagenet',
                               include_top=False,
                               input_shape=(self.env.img_height, self.env.img_width, 3))
            input_size = 1
            for dim in base_model.output_shape:
                if dim is not None:
                    input_size *= dim
            self.input_size = input_size

        if policy is None and initial_train == True:
            self.generate_theta = True
        self.base_model = base_model
        self.policy = policy or Policy(self.input_size, self.output_size, self.hp)
        self.weights_dir = mkdir('', weights_dir)

    # Explore the policy on one specific direction and over one episode
    def explore(self, direction=None, delta=None):
        state = self.env.reset()
        done = False
        sum_rewards = 0.0
        steps = 0
        while not done:
            # Get data from front camera and divide by 255 to normalize
            state = self.env.front_camera.reshape(1, 224, 224, 3) / 255.
            steps += 1
            if self.base_model:
                # Use base model to make prediction, flatten and divide by 10 to normalize
                state = self.base_model.predict(state).flatten() / 10.
            else:
                state = state.flatten()
            action = self.policy.evaluate(state, delta, direction)
            state, reward, done, _ = self.env.step(action, steps)
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
        print('Worker saw {} steps'.format(steps))
        # Average the rewards per step to account for variable FPS seen by workers
        print('Sum of episode rewards:', sum_rewards)
        adjusted_reward = sum_rewards / steps
        print('Adjusted Reward for episode:', adjusted_reward)
        return adjusted_reward

    def train(self):
        # Experimental. Used to train an initial set of weights for ARS to modify.
        if self.generate_theta:
            print('Training initial weights...')
            pred_model = keras.models.Sequential()
            base_model = VGG19(weights='imagenet',
                               include_top=False,
                               input_shape=(224, 224, 3))
            for layer in base_model.layers:
                layer.trainable = False
            pred_model.add(base_model)
            pred_model.add(keras.layers.Flatten())
            pred_model.add(Dense(3, input_dim=base_model.output_shape, activation='linear'))
            pred_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            X = train_imgs2.reshape(50, 224, 224, 3) / 255.
            y = np.array([1., 0., 0.])
            y = np.tile(y, (50, 1))
            print("prefit")
            pred_model.fit(X, y, epochs=5, workers=2)
            # print("pre saving")
            # pred_model.save(f'{"ars"}__{int(time.time())}.model')
            self.policy.theta = pred_model.get_weights()[-2].T

        for step in range(self.hp.nb_steps):
            self.historical_steps += 1
            print('Performing step {}. ({}/{})'.format(self.historical_steps,
                                                       step + 1,
                                                       self.hp.nb_steps
                                                       ))
            start = time.time()
            # Only record video during evaluation, every n steps
            if step % self.hp.record_every == 0:
                self.env.show_cam = True
            # initialize the random noise deltas and the positive/negative rewards
            deltas = self.policy.sample_deltas()
            positive_rewards = [0] * self.hp.num_deltas
            negative_rewards = [0] * self.hp.num_deltas

            # play an episode each with positive deltas and negative deltas, collect rewards
            for k in range(self.hp.num_deltas):
                positive_rewards[k] = self.explore(direction="+", delta=deltas[k])
                negative_rewards[k] = self.explore(direction="-", delta=deltas[k])

            # Compute the standard deviation of all rewards
            sigma_rewards = np.array(positive_rewards + negative_rewards).std()

            # Sort the rollouts by the max(r_pos, r_neg) and select the deltas with best rewards
            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.hp.num_best_deltas]
            rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]

            # Update the policy
            self.policy.update(rollouts, sigma_rewards)

            if step % self.hp.record_every == 0:
                # Play an episode with the new weights and print the score
                reward_evaluation = self.run_episode()
                print('Step:', step + 1, 'Reward:', reward_evaluation)
                self.history['step'].append(self.historical_steps)
                self.history['score'].append(reward_evaluation)
                self.history['theta'].append(self.policy.theta.copy())
                # self.save()

            end = time.time()
            print('Time to complete update step:', end - start)
            self.env.show_cam = False

        self.save()

    def save(self):
        save_file = mkdir(self.weights_dir, str(datetime.date.today()))
        np.savetxt(save_file + '/recent_weights.csv'.format(self.historical_steps),
                   self.policy.theta,
                   delimiter=','
                   )

    def run_episode(self):
        return self.explore()

    def clean_up(self):
        for actor in self.env.actor_list:
            actor.destroy()


# Establish hyperparameters to pass to agent
#nb_steps was 10
hp_test = HP(nb_steps=1,
             noise=0.05,
             learning_rate=0.02,
             num_deltas=16,
             num_best_deltas=8,
             record_every=1
             )

# Instantiate the agent using these hyperparameters
ars_agent = ARSAgent(hp=hp_test)

# Load in previously trained weights if you have some
# weights = np.genfromtxt('ars_weights/2020-12-21/recent_weights.csv', delimiter=',')
# ars_agent.policy.theta = weights

# Train the agent
ars_agent.train()
ars_agent.pred_model.save(f'{"ars"}__{int(time.time())}.model')
print("saving")

# A quick sanity check to make sure that our theta was changing each time
for i, theta in enumerate(ars_agent.history['theta']):
    if i + 1 == len(ars_agent.history['theta']):
        break
    if np.array_equal(theta, ars_agent.history['theta'][i + 1]):
        print('same')
    else:
        print('different')

# Save weights for later use
ars_agent.save()

# Clean up actors and clear agent from memory
ars_agent.clean_up()
del ars_agent


# class Normalizer:
#     # Normalizes the inputs
#     def __init__(self, nb_inputs):
#         self.n = np.zeros(nb_inputs)
#         self.mean = np.zeros(nb_inputs)
#         self.mean_diff = np.zeros(nb_inputs)  #numerator
#         self.var = np.zeros(nb_inputs)
#
#     def observe(self, x):
#         self.n += 1.0
#         last_mean = self.mean.copy()
#         self.mean += (x - self.mean) / self.n
#         self.mean_diff += (x - last_mean) * (x - self.mean)
#         self.var = (self.mean_diff / self.n).clip(min=1e-2)
#
#     def normalize(self, inputs):
#         obs_mean = self.mean
#         obs_std = np.sqrt(self.var)
#         return (inputs - obs_mean) / obs_std
#
# class ARSTrainer:
#     def __init__(self,
#                  hp=None,
#                  input_size=None,
#                  output_size=None,
#                  normalizer=None,
#                  policy=None,
#                  monitor_dir=None):
#
#         self.hp = hp or HP()
#         np.random.seed(self.hp.seed)
#         self.env = gym.make(self.hp.env_name)
#         if monitor_dir is not None:
#             should_record = lambda i: self.record_video
#             self.env = gym.wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)
#         self.hp.episode_length = self.hp.episode_length
#         self.input_size = input_size or self.env.observation_space.shape[0]
#         self.output_size = output_size or self.env.action_space.shape[0]
#         self.normalizer = normalizer or Normalizer(self.input_size)
#         self.policy = policy or Policy(self.input_size, self.output_size, self.hp)
#         self.record_video = False
#
#     # Explore the policy on one specific direction and over one episode
#     def explore(self, direction=None, delta=None):
#         state = self.env.reset()
#         done = False
#         num_plays = 0.0
#         sum_rewards = 0.0
#         while not done and num_plays < self.hp.episode_length:
#             self.normalizer.observe(state)
#             state = self.normalizer.normalize(state)
#             action = self.policy.evaluate(state, delta, direction)
#             state, reward, done, _ = self.env.step(action)
#             reward = max(min(reward, 1), -1)
#             sum_rewards += reward
#             num_plays += 1
#         return sum_rewards
#
#     def train(self):
#         for step in range(self.hp.nb_steps):
#             # initialize the random noise deltas and the positive/negative rewards
#             deltas = self.policy.sample_deltas()
#             positive_rewards = [0] * self.hp.num_deltas
#             negative_rewards = [0] * self.hp.num_deltas
#
#             # play an episode each with positive deltas and negative deltas, collect rewards
#             for k in range(self.hp.num_deltas):
#                 positive_rewards[k] = self.explore(direction="+", delta=deltas[k])
#                 negative_rewards[k] = self.explore(direction="-", delta=deltas[k])
#
#             # Compute the standard deviation of all rewards
#             sigma_rewards = np.array(positive_rewards + negative_rewards).std()
#
#             # Sort the rollouts by the max(r_pos, r_neg) and select the deltas with best rewards
#             scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
#             order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:self.hp.num_best_deltas]
#             rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
#
#             # Update the policy
#             self.policy.update(rollouts, sigma_rewards)
#
#             # Only record video during evaluation, every n steps
#             if step % self.hp.record_every == 0:
#                 self.record_video = True
#             # Play an episode with the new weights and print the score
#             reward_evaluation = self.explore()
#             print('Step: ', step, 'Reward: ', reward_evaluation)
#             self.record_video = False
#
#
# hp_test = HP(nb_steps=50,
#              noise=0.05,
#              learning_rate=0.02,
#              num_deltas=32,
#              num_best_deltas=16,
#              record_every=1
#              )
#
# len(ars_agent.history['theta'])
#
# weights = ars_agent.history['theta'][-1]

