import carla
import json

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0) 

# Maps you want to extract spawn points from
maps_to_extract = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05','Town10HD']

all_scenarios = {}

for map_name in maps_to_extract:
    print(f"\n=== Processing {map_name} ===")
    
    # Load the map
    world = client.load_world(f'D:\CARLA\CarlaUE4\Content\Carla\Maps\{map_name}')
    carla_map = world.get_map()
    
    # Get spawn points
    spawn_points = carla_map.get_spawn_points()
    
    print(f"Found {len(spawn_points)} spawn points")
    
    # Create scenarios using pairs of spawn points
    for i in range(min(5, len(spawn_points) - 1)):  # Create 5 scenarios per map
        spawn = spawn_points[i]
        target = spawn_points[i + 10]  # Use a spawn point further away as target
        
        scenario_name = f"{map_name}-ClearNoon-Road-{i}"
        
        all_scenarios[scenario_name] = {
            "map_name": map_name,
            "weather_condition": "Clear Noon",
            "initial_position": {
                "x": round(spawn.location.x, 1),
                "y": round(spawn.location.y, 1),
                "z": round(spawn.location.z, 1)
            },
            "initial_rotation": {
                "pitch": round(spawn.rotation.pitch, 1),
                "yaw": round(spawn.rotation.yaw, 1),
                "roll": round(spawn.rotation.roll, 1)
            },
            "target_position": {
                "x": round(target.location.x, 1),
                "y": round(target.location.y, 1),
                "z": round(target.location.z, 1)
            },
            "target_gnss": {"lat": 0, "lon": 0, "alt": 0},
            "traffic_density": "Low",
            "situation": "Road"
        }

# Save to file
with open('new_scenarios.json', 'w') as f:
    json.dump(all_scenarios, f, indent=2)

print("\n✅ Scenarios saved to 'new_scenarios.json'")