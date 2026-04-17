"""
CARLA Scenario Runner with Robust Exception Handling
Cycles through all scenarios with traffic, pedestrians, and dynamic weather.
"""

import carla
import random
import time
import sys
import json
import traceback
from typing import List, Tuple, Optional
import gymnasium as gym
from src.env.environment import CarlaEnv
import src.config.configuration as config


class CarlaScenarioRunner:
    """Runs CARLA scenarios with automatic recovery from errors"""
    
    def __init__(self, host="localhost", port=2000, timeout=10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.vehicles = []
        self.walkers = []
        self.walker_controllers = []
        
    def connect(self):
        """Connect to CARLA server with retry logic"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                print(f"Connecting to CARLA server at {self.host}:{self.port} (attempt {attempt + 1}/{max_retries})...")
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(self.timeout)
                self.world = self.client.get_world()
                print("✓ Connected to CARLA server successfully!")
                return True
            except Exception as e:
                print(f"✗ Connection failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    print("Failed to connect after maximum retries.")
                    return False
        return False
    
    def spawn_ego_vehicle(self, autopilot=True, vehicle_model="vehicle.tesla.model3"):
        """Spawn ego vehicle with error handling"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            ego_bp = blueprint_library.filter(vehicle_model)[0]
            spawn_points = self.world.get_map().get_spawn_points()
            
            if not spawn_points:
                print("✗ No spawn points available!")
                return False
            
            spawn_point = random.choice(spawn_points)
            self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
            self.ego_vehicle.set_autopilot(autopilot)
            print(f"✓ Spawned ego vehicle at {spawn_point.location}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to spawn ego vehicle: {e}")
            traceback.print_exc()
            return False
    
    def spawn_traffic(self, num_vehicles=20):
        """Spawn NPC vehicles with error handling"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bps = blueprint_library.filter("vehicle.*")
            spawn_points = self.world.get_map().get_spawn_points()
            
            spawned = 0
            for sp in random.sample(spawn_points, min(num_vehicles, len(spawn_points))):
                bp = random.choice(vehicle_bps)
                try:
                    vehicle = self.world.try_spawn_actor(bp, sp)
                    if vehicle is not None:
                        vehicle.set_autopilot(True)
                        self.vehicles.append(vehicle)
                        spawned += 1
                except:
                    continue
            
            print(f"✓ Spawned {spawned}/{num_vehicles} traffic vehicles")
            return True
            
        except Exception as e:
            print(f"✗ Failed to spawn traffic: {e}")
            traceback.print_exc()
            return False
    
    def spawn_pedestrians(self, num_pedestrians=30):
        """Spawn pedestrians with error handling"""
        try:
            blueprint_library = self.world.get_blueprint_library()
            walker_bps = blueprint_library.filter("walker.pedestrian.*")
            walker_controller_bp = blueprint_library.find("controller.ai.walker")
            
            spawned = 0
            for _ in range(num_pedestrians):
                try:
                    loc = self.world.get_random_location_from_navigation()
                    if loc is None:
                        continue
                    
                    transform = carla.Transform(loc)
                    walker_bp = random.choice(walker_bps)
                    walker = self.world.try_spawn_actor(walker_bp, transform)
                    
                    if walker is None:
                        continue
                    
                    controller = self.world.spawn_actor(
                        walker_controller_bp,
                        carla.Transform(),
                        attach_to=walker
                    )
                    
                    controller.start()
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                    controller.set_max_speed(random.uniform(0.5, 1.5))
                    
                    self.walkers.append(walker)
                    self.walker_controllers.append(controller)
                    spawned += 1
                    
                except:
                    continue
            
            print(f"✓ Spawned {spawned}/{num_pedestrians} pedestrians")
            return True
            
        except Exception as e:
            print(f"✗ Failed to spawn pedestrians: {e}")
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """Clean up all spawned actors"""
        try:
            print("Cleaning up actors...")
            
            # Destroy ego vehicle
            if self.ego_vehicle is not None:
                self.ego_vehicle.destroy()
                self.ego_vehicle = None
            
            # Destroy traffic vehicles
            for vehicle in self.vehicles:
                try:
                    vehicle.destroy()
                except:
                    pass
            self.vehicles.clear()
            
            # Destroy pedestrians
            for controller in self.walker_controllers:
                try:
                    controller.stop()
                    controller.destroy()
                except:
                    pass
            self.walker_controllers.clear()
            
            for walker in self.walkers:
                try:
                    walker.destroy()
                except:
                    pass
            self.walkers.clear()
            
            print("✓ Cleanup complete")
            
        except Exception as e:
            print(f"✗ Cleanup error: {e}")
            traceback.print_exc()
    
    def run_scenario(self, duration=60, num_vehicles=20, num_pedestrians=30):
        """Run a complete scenario with error handling"""
        try:
            print("\n" + "="*70)
            print("STARTING NEW SCENARIO")
            print("="*70)
            
            # Spawn all actors
            if not self.spawn_ego_vehicle():
                return False
            
            self.spawn_traffic(num_vehicles)
            self.spawn_pedestrians(num_pedestrians)
            
            # Update spectator to follow ego vehicle
            spectator = self.world.get_spectator()
            
            # Run scenario
            start_time = time.time()
            tick_count = 0
            
            while time.time() - start_time < duration:
                try:
                    self.world.wait_for_tick()
                    tick_count += 1
                    
                    # Update spectator position every 10 ticks
                    if tick_count % 10 == 0 and self.ego_vehicle is not None:
                        transform = self.ego_vehicle.get_transform()
                        spectator.set_transform(
                            carla.Transform(
                                transform.location + carla.Location(z=30),
                                carla.Rotation(pitch=-90)
                            )
                        )
                    
                except KeyboardInterrupt:
                    print("\n⚠ Scenario interrupted by user")
                    return False
                except Exception as e:
                    print(f"✗ Error during scenario tick: {e}")
                    continue
            
            print(f"✓ Scenario completed ({duration}s, {tick_count} ticks)")
            return True
            
        except Exception as e:
            print(f"✗ Scenario failed: {e}")
            traceback.print_exc()
            return False
        finally:
            self.cleanup()


def get_scenario_list():
    """Load scenarios from JSON file"""
    try:
        with open(config.ENV_SCENARIOS_FILE, 'r') as f:
            scenarios_dict = json.load(f)
        scenario_list = list(scenarios_dict.keys())
        print(f"✓ Loaded {len(scenario_list)} scenarios from {config.ENV_SCENARIOS_FILE}")
        return scenario_list, scenarios_dict
    except Exception as e:
        print(f"✗ Failed to load scenarios: {e}")
        traceback.print_exc()
        return [], {}


def unwrap_env(env):
    """Unwrap gym environment to get the base CarlaEnv"""
    while hasattr(env, 'env'):
        env = env.env
    return env


def run_gym_environment_scenarios(num_episodes=5, time_limit=60, scenario_filter=None):
    """Run scenarios using the Gym environment with cycling and error handling"""
    
    env = None
    scenario_list = []
    scenarios_dict = {}
    
    try:
        print("Loading scenarios...")
        scenario_list, scenarios_dict = get_scenario_list()
        
        if not scenario_list:
            print("✗ No scenarios found. Exiting.")
            return
        
        # Print all scenarios
        print("\nAvailable scenarios:")
        for idx, scenario_name in enumerate(scenario_list):
            scenario_info = scenarios_dict[scenario_name]
            print(f"  {idx:2d}. {scenario_name:40s} | Map: {scenario_info['map_name']:15s} | Weather: {scenario_info['weather_condition']:20s} | Traffic: {scenario_info['traffic_density']}")
        
        # Filter scenarios if needed
        if scenario_filter:
            filtered_scenarios = [s for s in scenario_list if scenario_filter.lower() in s.lower()]
            if filtered_scenarios:
                scenario_list = filtered_scenarios
                print(f"\n✓ Filtered to {len(scenario_list)} scenarios matching '{scenario_filter}'")
            else:
                print(f"\n⚠ No scenarios match filter '{scenario_filter}', using all scenarios")
        
        print(f"\nInitializing CARLA Gym environment...")
        
        # Create environment directly (without gym.make to avoid wrappers initially)
        env = CarlaEnv(
            continuous=False,
            scenarios=[],  # Empty means use all scenarios
            time_limit=time_limit,
            initialize_server=True,
            random_weather=True,
            synchronous_mode=True,
            show_sensor_data=True,
            random_traffic=True,
            has_traffic=True,
            apply_physics=True,
            autopilot=False,
            verbose=True
        )
        
        print("✓ Environment initialized successfully")
        
    except Exception as e:
        print(f"✗ Failed to initialize environment: {e}")
        traceback.print_exc()
        return
    
    # Run episodes cycling through scenarios
    episode_count = 0
    scenario_idx = 0
    successful_episodes = 0
    failed_episodes = 0
    
    while episode_count < num_episodes:
        scenario_name = scenario_list[scenario_idx % len(scenario_list)]
        
        try:
            print("\n" + "="*70)
            print(f"EPISODE {episode_count + 1}/{num_episodes}")
            print(f"Scenario: {scenario_name}")
            print(f"Map: {scenarios_dict[scenario_name]['map_name']}")
            print(f"Weather: {scenarios_dict[scenario_name]['weather_condition']}")
            print(f"Traffic: {scenarios_dict[scenario_name]['traffic_density']}")
            print("="*70)
            
            # Reset with specific scenario
            obs, info = env.reset(options={'scenario_name': scenario_name})
            
            step_count = 0
            total_reward = 0.0
            episode_start_time = time.time()
            
            while True:
                try:
                    # Random action (replace with your agent's policy)
                    action = env.action_space.sample()
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    step_count += 1
                    
                    if step_count % 50 == 0:
                        elapsed = time.time() - episode_start_time
                        print(f"  Step {step_count:4d} | Reward: {reward:6.2f} | Total: {total_reward:7.2f} | Time: {elapsed:.1f}s")
                    
                    if terminated or truncated:
                        status = "TERMINATED" if terminated else "TRUNCATED"
                        elapsed = time.time() - episode_start_time
                        print(f"\n{'='*70}")
                        print(f"✓ Episode {status}")
                        print(f"  Steps: {step_count}")
                        print(f"  Total Reward: {total_reward:.2f}")
                        print(f"  Duration: {elapsed:.1f}s")
                        print(f"{'='*70}")
                        successful_episodes += 1
                        break
                        
                except KeyboardInterrupt:
                    print("\n⚠ Episode interrupted by user")
                    env.close()
                    return
                    
                except Exception as e:
                    print(f"\n✗ Step error: {e}")
                    traceback.print_exc()
                    failed_episodes += 1
                    break
            
            episode_count += 1
            scenario_idx += 1
            
        except KeyboardInterrupt:
            print("\n⚠ Training interrupted by user")
            break
            
        except Exception as e:
            print(f"\n✗ Episode failed: {e}")
            traceback.print_exc()
            failed_episodes += 1
            scenario_idx += 1  # Skip to next scenario
            episode_count += 1
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total Episodes: {episode_count}")
    print(f"Successful: {successful_episodes}")
    print(f"Failed: {failed_episodes}")
    print(f"Success Rate: {(successful_episodes/episode_count*100):.1f}%")
    print("="*70)
    
    # Cleanup
    try:
        env.close()
        print("\n✓ Environment closed successfully")
    except Exception as e:
        print(f"✗ Error closing environment: {e}")


def run_interactive_scenario_selector():
    """Interactive mode to select and run specific scenarios"""
    scenario_list, scenarios_dict = get_scenario_list()
    
    if not scenario_list:
        print("✗ No scenarios found.")
        return
    
    print("\n" + "="*70)
    print("INTERACTIVE SCENARIO SELECTOR")
    print("="*70)
    print("\nAvailable scenarios:")
    
    for idx, scenario_name in enumerate(scenario_list):
        scenario_info = scenarios_dict[scenario_name]
        print(f"  {idx:2d}. {scenario_name}")
    
    print("\nEnter scenario indices to run (comma-separated), or 'all' for all scenarios:")
    selection = input("> ").strip()
    
    if selection.lower() == 'all':
        selected_scenarios = scenario_list
    else:
        try:
            indices = [int(x.strip()) for x in selection.split(',')]
            selected_scenarios = [scenario_list[i] for i in indices if 0 <= i < len(scenario_list)]
        except:
            print("✗ Invalid input. Exiting.")
            return
    
    print(f"\n✓ Selected {len(selected_scenarios)} scenarios")
    
    time_limit = int(input("Time limit per episode (seconds, default=60): ") or "60")
    repetitions = int(input("Repetitions per scenario (default=1): ") or "1")
    
    total_episodes = len(selected_scenarios) * repetitions
    print(f"\nRunning {total_episodes} episodes...")
    
    # Create custom scenario list
    episode_scenarios = []
    for _ in range(repetitions):
        episode_scenarios.extend(selected_scenarios)
    
    # Run with custom list
    env = None
    try:
        env = CarlaEnv(
            continuous=False,
            time_limit=time_limit,
            initialize_server=True,
            random_weather=True,
            synchronous_mode=True,
            show_sensor_data=True,
            random_traffic=True,
            verbose=True
        )
        
        for i, scenario_name in enumerate(episode_scenarios):
            print(f"\n{'='*70}")
            print(f"Episode {i+1}/{total_episodes}: {scenario_name}")
            print(f"{'='*70}")
            
            try:
                obs, info = env.reset(options={'scenario_name': scenario_name})
                
                while True:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if terminated or truncated:
                        break
                        
            except KeyboardInterrupt:
                print("\n⚠ Interrupted by user")
                break
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
                
    finally:
        if env:
            env.close()


def main():
    """Main entry point with mode selection"""
    print("="*70)
    print("CARLA SCENARIO RUNNER")
    print("="*70)
    print("\nSelect mode:")
    print("1. Manual scenario runner (basic CARLA API)")
    print("2. Gym environment runner (cycles through all scenarios)")
    print("3. Interactive scenario selector")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        # Manual scenario runner
        runner = CarlaScenarioRunner()
        
        if not runner.connect():
            print("Failed to connect to CARLA. Exiting.")
            return
        
        num_scenarios = int(input("Number of scenarios to run: "))
        scenario_duration = int(input("Duration per scenario (seconds): "))
        
        for i in range(num_scenarios):
            try:
                print(f"\n\nRunning scenario {i + 1}/{num_scenarios}")
                runner.run_scenario(
                    duration=scenario_duration,
                    num_vehicles=random.randint(10, 20),
                    num_pedestrians=random.randint(20, 30)
                )
                
                # Wait between scenarios
                if i < num_scenarios - 1:
                    print("Waiting 3 seconds before next scenario...")
                    time.sleep(3)
                    
            except KeyboardInterrupt:
                print("\n⚠ Interrupted by user")
                break
            except Exception as e:
                print(f"✗ Scenario {i + 1} failed: {e}")
                traceback.print_exc()
                continue
        
        print("\n" + "="*70)
        print("ALL SCENARIOS COMPLETED")
        print("="*70)
    
    elif choice == "2":
        # Gym environment runner
        num_episodes = int(input("Number of episodes: "))
        time_limit = int(input("Time limit per episode (seconds): "))
        scenario_filter = input("Filter scenarios by name (press Enter for all): ").strip()
        
        try:
            run_gym_environment_scenarios(
                num_episodes=num_episodes,
                time_limit=time_limit,
                scenario_filter=scenario_filter if scenario_filter else None
            )
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        except Exception as e:
            print(f"✗ Fatal error: {e}")
            traceback.print_exc()
    
    elif choice == "3":
        # Interactive selector
        try:
            run_interactive_scenario_selector()
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")
        except Exception as e:
            print(f"✗ Fatal error: {e}")
            traceback.print_exc()
    
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)