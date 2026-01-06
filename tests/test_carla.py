import carla
import random
import time

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)

    world = client.get_world()

    # OPTIONAL: force a map
    # world = client.load_world("Town05")

    blueprint_library = world.get_blueprint_library()

    # ------------------------
    # Spawn ego vehicle
    # ------------------------
    ego_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())

    ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
    ego_vehicle.set_autopilot(True)

    # ------------------------
    # Spawn NPC vehicles
    # ------------------------
    vehicle_bps = blueprint_library.filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()

    vehicles = []
    for sp in spawn_points[:20]:
        bp = random.choice(vehicle_bps)
        try:
            v = world.spawn_actor(bp, sp)
            v.set_autopilot(True)
            vehicles.append(v)
        except:
            pass

    # ------------------------
    # Spawn pedestrians
    # ------------------------
    walker_bps = blueprint_library.filter("walker.pedestrian.*")
    walkers = []
    spectator = world.get_spectator()

    for _ in range(30):

        loc = world.get_random_location_from_navigation()
        if loc:
            bp = random.choice(walker_bps)
            transform = carla.Transform(loc)
            walker = world.spawn_actor(bp, transform)
            walkers.append(walker)

    print(f"Spawned {len(vehicles)} vehicles and {len(walkers)} pedestrians")

    try:
        while True:
            world.wait_for_tick()
               # Update spectator every frame
            transform = ego_vehicle.get_transform()
            spectator.set_transform(
            carla.Transform(
                    transform.location + carla.Location(z=30),
                    carla.Rotation(pitch=-90)
                    )
                )
    finally:
        print("Destroying actors...")
        ego_vehicle.destroy()
        for v in vehicles:
            v.destroy()
        for w in walkers:
            w.destroy()

if __name__ == "__main__":
    main()
