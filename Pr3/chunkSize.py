import pandas as pd
import time

csv_file = "vehicular_dataset_2025-01-01.csv"

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} виконано за {end_time - start_time:.2f} секунд")
        return result
    return wrapper

@measure_time
def load_full():
    return pd.read_csv(csv_file)

@measure_time
def load_with_chunks(chunk_size=100000):
    return pd.concat(pd.read_csv(csv_file, chunksize=chunk_size), ignore_index=True)

@measure_time
def load_with_optimized_types():
    dtypes = {
        "VehicleID": "int32", "Timestamp": "str", "Day": "int8", "Hour": "int8",
        "Minute": "int8", "Second": "int8", "Latitude": "float32", "Longitude": "float32",
        "Speed": "float32", "Direction": "float32", "StartingPointLatitude": "float32",
        "StartingPointLongitude": "float32", "DestinationLatitude": "float32", "DestinationLongitude": "float32",
        "CPU_Available": "float32", "Memory_Available": "float32", "BatteryLevel": "float32",
        "TaskType": "category", "TaskSize": "int32", "TaskPriority": "int8", "NetworkLatency": "float32",
        "SignalStrength": "float32", "TrafficDensity": "float32", "WeatherCondition": "category",
        "RoadCondition": "category", "VehicleType": "category", "VehicleAge": "int8",
        "EngineTemperature": "float32", "FuelLevel": "float32", "TirePressure": "float32",
        "BrakeFluidLevel": "float32", "CoolantLevel": "float32", "OilLevel": "float32",
        "WiperFluidLevel": "float32", "HeadlightStatus": "bool", "BrakeLightStatus": "bool",
        "TurnSignalStatus": "bool", "HazardLightStatus": "bool", "ABSStatus": "bool",
        "AirbagStatus": "bool", "GPSStatus": "bool", "WiFiStatus": "bool", "BluetoothStatus": "bool",
        "CellularStatus": "bool", "RadarStatus": "bool", "LidarStatus": "bool",
        "CameraStatus": "bool", "IMUStatus": "bool", "Odometer": "int32", "TaskOffloaded": "bool"
    }
    return pd.read_csv(csv_file, dtype=dtypes, parse_dates=["Timestamp"])

load_full()
load_with_chunks()
load_with_optimized_types()
