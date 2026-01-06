'''
MapControl:
    - Module that controls the current map of the simulation, and allows its customization
'''
import carla
import time
import src.config.configuration as config

class MapControl:
    def __init__(self, world, client):
        self.__world          = world
        self.__client         = client
        self.__available_maps = [m for m in self.__client.get_available_maps() if 'Opt' not in m]
        self.__map_dict       = {m.split("/")[-1]: idx for idx, m in enumerate(self.__available_maps)}
        
        # Handle edge case where map might not be in dict
        current_map_name = self.__world.get_map().name.split("/")[-1].split("_")[0]
        if current_map_name in self.__map_dict:
            self.__active_map = self.__map_dict[current_map_name]
        else:
            self.__active_map = 0
            
        self.__map = self.__world.get_map()
        
        # Debug: Print available maps
        print(f"MapControl initialized. Available maps: {list(self.__map_dict.keys())}")

    def get_active_map_name(self):
        return self.__map.name.split("/")[-1].split("_")[0]

    def get_map(self):
        return self.__map
    
    def print_available_maps(self):
        for idx, m in enumerate(self.__available_maps):
            print(f'{idx}: {m}')
    
    def set_active_map(self, map_name, reload_map=False):
        # Check if map exists
        if map_name not in self.__map_dict:
            available = list(self.__map_dict.keys())
            raise ValueError(f"Map '{map_name}' not available. Available maps: {available}")
        
        # Check if the map is already loaded
        if self.__map_dict[map_name] == self.__active_map and not reload_map:
            print(f"Map '{map_name}' already loaded, skipping...")
            return
        
        self.__active_map = self.__map_dict[map_name]
        
        print(f"Loading map: {map_name}")
        try:
            # Method 1: Simple load (like the test script)
            new_world = self.__client.load_world(map_name)
            
            # Update the world reference
            self.__world._World__world = new_world
            
            time.sleep(3)
            self.__map = new_world.get_map()
            print(f"Map '{map_name}' loaded successfully!")
            
        except RuntimeError as e:
            print(f"ERROR: Failed to load map '{map_name}'")
            print(f"Error details: {e}")
            
            # Fallback: Try with full CARLA path
            print(f"Trying alternative path format...")
            try:
                carla_map_path = f"/Game/Carla/Maps/{map_name}"
                new_world = self.__client.load_world(carla_map_path, reset_settings=False)
                self.__world._World__world = new_world
                time.sleep(3)
                self.__map = new_world.get_map()
                print(f"Map '{map_name}' loaded successfully with alternative method!")
            except RuntimeError as e2:
                print(f"Both methods failed. Available maps: {self.__available_maps}")
                raise

    def change_map(self):
        self.print_available_maps()
        map_idx = int(input('Choose a map index: '))
        map_name = list(self.__map_dict.keys())[map_idx]
        self.set_active_map(map_name)
    
    def reload_map(self):
        self.set_active_map(self.get_active_map_name(), reload_map=True)