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
        
        # Load map - this is blocking and returns new world
        new_world = self.__client.load_world(map_name)
        
        # CRITICAL: Wait for the world to be ready
        # Tick a few times to ensure episode is initialized
        for _ in range(5):
            new_world.tick()
            time.sleep(0.1)
        
        # Update internal references
        self.__world._World__world = new_world
        self.__map = new_world.get_map()
        
        # Additional wait for stability
        time.sleep(2)
        
        print(f"Map '{map_name}' loaded successfully!")
    def change_map(self):
        self.print_available_maps()
        map_idx = int(input('Choose a map index: '))
        map_name = list(self.__map_dict.keys())[map_idx]
        self.set_active_map(map_name)
    
    def reload_map(self):
        self.set_active_map(self.get_active_map_name(), reload_map=True)