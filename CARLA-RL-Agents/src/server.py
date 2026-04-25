import os
import subprocess
import time

'''
Server Module

This module contains the CarlaServer class that is responsible for starting and stopping the Carla server.

Requirements:
    - Environment variable CARLA_SERVER that contains the path to the Carla server directory
'''

class CarlaServer:
    @staticmethod
    def initialize_server(low_quality = False, offscreen_rendering = False, silent = False, sleep_time = 10):
        # Get environment variable CARLA_SERVER that contains the path to the Carla server directory
        carla_server = os.getenv('CARLA_SERVER')
        
        # If CARLA_SERVER is not set, raise an error with instructions
        if carla_server is None:
            error_msg = """
╔═══════════════════════════════════════════════════════════════╗
║ CARLA SERVER PATH NOT FOUND                                   ║
╚═══════════════════════════════════════════════════════════════╝

The CARLA_SERVER environment variable is not set.

TO FIX THIS, either:
  
  Option 1: Set environment variable in PowerShell (temporary)
    $env:CARLA_SERVER = "C:\\path\\to\\CARLA"
    
  Option 2: Set environment variable permanently (Windows)
    [Environment]::SetEnvironmentVariable(
        "CARLA_SERVER",
        "C:\\path\\to\\CARLA",
        [EnvironmentVariableTarget]::User
    )
    
  Option 3: Start CARLA server manually, then use --no_server flag
    python training/encoder_training.py --no_server
    
CARLA Installation:
  - Download from: https://github.com/carla-simulator/carla/releases
  - Extract to a directory (e.g., C:\\CARLA or C:\\carla-0.9.15)
  - Set CARLA_SERVER to that directory
  - Run CarlaUE4.exe or CarlaUE4.sh from that directory
            """
            raise RuntimeError(error_msg.strip())

        # If it is Unix add the CarlaUE4.sh to the path else add CarlaUE4.exe
        if os.name == 'posix':
            carla_server = os.path.join(carla_server, 'CarlaUE4.sh')
            command = f"bash {carla_server} {'--quality-level=Low' if low_quality else ''} {'--RenderOffScreen' if offscreen_rendering else ''}"
        else:
            carla_server = os.path.join(carla_server, 'CarlaUE4.exe')
            command = f"{carla_server} {'--quality-level=Low' if low_quality else ''} {'--RenderOffScreen' if offscreen_rendering else ''}"

        # Run the command
        if not silent:
            print('Starting Carla server, please wait...')
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
        # Wait for the server to start
        time.sleep(sleep_time)
        if not silent:
            print('Carla server started')

        return process
    
    @staticmethod
    def close_server(process, silent = False):
        if os.name == 'posix':
            os.killpg(os.getpgid(process.pid), 15)
            if not silent:
                print('Carla server closed')
        else:
            # On Windows, use taskkill to terminate the process and all its children
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if not silent:
                print('Carla server closed')
    
    @staticmethod
    def restart_server(process, low_quality = False, offscreen_rendering = False, silent = False, sleep_time = 10):
        CarlaServer.close_server(process, silent)
        return CarlaServer.initialize_server(low_quality, offscreen_rendering, silent, sleep_time)
    
    @staticmethod
    def kill_carla_linux():
        if os.name == 'posix':
            os.system('pkill -9 -f CarlaUE4')
            print('Carla server closed')
        else:
            print('This method is only for Unix systems! Please close the Carla server manually.')
