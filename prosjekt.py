import os
from typing import Any
import numpy as np
import pandas as pd # type: ignore
from pydantic import BaseModel
from scipy.spatial.transform import Rotation as R # type: ignore
import matplotlib.pyplot as plt # type: ignore
import colorama # type: ignore

class Vector3d(BaseModel):
    x: float
    y: float
    z: float

    @classmethod
    def from_list(cls, data: list[float]):
        return cls(x=data[0], y=data[1], z=data[2])
    
    def to_list(self):
        return [self.x, self.y, self.z]
    
    @property
    def magnitude(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    @property
    def declination(self):
        return np.rad2deg(np.atan(self.x / self.y))
    
    @property
    def inclination(self):
        return np.rad2deg(np.atan(self.z/np.sqrt(self.x**2 + self.y**2)))
    
    def rotate(self, rot: R):
        vec = self.to_list()
        new_vec = rot.apply(vec)
        return Vector3d.from_list(new_vec)
    
    def __add__(self, other):
        assert isinstance(other, Vector3d)
        return Vector3d(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __truediv__(self, other: Any):
        return Vector3d(x=self.x/other, y=self.y/other, z=self.z/other)


class MagneticFieldCalc(BaseModel):
    data_path: str
    _acceleration_data: dict[str, list[Vector3d]] = {}
    _magnet_data: dict[str, list[Vector3d]] = {}
    _aligned_magnet_vectors: dict[str, list[Vector3d]] = {}


    @property
    def acceleration_data(self):
        if self._acceleration_data == {}:
            self.get_data()
        return self._acceleration_data

    @property
    def magnet_data(self):
        if self._magnet_data == {}:
            self.get_data()
        return self._magnet_data

    @property
    def aligned_magnet_vectors(self):
        if self._aligned_magnet_vectors == {}:
            self.align_vectors()
        return self._aligned_magnet_vectors

    def get_dirs(self):
        return os.listdir(self.data_path)

    def get_data(self):
        for folder in self.get_dirs():
            path = os.path.join(self.data_path, folder)
            if not os.path.exists(os.path.join(path, "Accelerometer.csv")):
                continue
            if not os.path.exists(os.path.join(path, "Magnetometer.csv")):
                continue
            acceleration_data = pd.read_csv(os.path.join(path, "Accelerometer.csv"), header=1)
            magnet_data = pd.read_csv(os.path.join(path, "Magnetometer.csv"), header=1)
            acceleration_vecs = [Vector3d.from_list([d[1], d[2], d[3]]) for d in acceleration_data.values]
            magnet_vecs = [Vector3d.from_list([d[1], d[2], d[3]]) for d in magnet_data.values]
            self._acceleration_data[folder] = acceleration_vecs
            self._magnet_data[folder] = magnet_vecs
    
    def align_vector(self, gravity_vec: Vector3d, magnet_vec: Vector3d) -> Vector3d:
        rot = R.align_vectors(gravity_vec.to_list(), [0,0,1])[0]
        new_vec = magnet_vec.rotate(rot)
        return new_vec
    
    def align_vectors(self):
        for key, data in self.acceleration_data.items():
            aligned_magnet_vecs = [self.align_vector(g_vec, magnet_data) for g_vec, magnet_data in zip(data, self.magnet_data[key])]
            self._aligned_magnet_vectors[key] = aligned_magnet_vecs

    @staticmethod
    def average_vector(vectors: list[Vector3d]):
        vec = Vector3d(x=0,y=0,z=0)
        for v in vectors:
            vec = vec + v
        return vec/len(vectors)

    def average_declination(self, magnet_vectors: list[Vector3d]):
        average_vector = self.average_vector(magnet_vectors)
        angle = average_vector.declination
        return angle

    @property
    def average_declinations(self):
        average_declination_dict = {key: self.average_declination(magnet_vectors) for key, magnet_vectors in self.aligned_magnet_vectors.items()}
        return average_declination_dict

    def average_magnitude(self, magnet_vectors: list[Vector3d]):
        average_vector = self.average_vector(magnet_vectors)
        magnitude = average_vector.magnitude
        return magnitude
    
    @property
    def average_magnitudes(self):
        average_magnitude_dict = {key: self.average_magnitude(magnet_vectors) for key, magnet_vectors in self.aligned_magnet_vectors.items()}
        return average_magnitude_dict
    
    def average_inclination(self, magnet_vectors: list[Vector3d]):
        average_vector = self.average_vector(magnet_vectors)
        inclination = average_vector.inclination
        return inclination
    
    @property
    def average_inclinations(self):
        average_inclination_dict = {key: self.average_inclination(magnet_vectors) for key, magnet_vectors in self.aligned_magnet_vectors.items()}
        return average_inclination_dict
    
    def print_info(self):
        def print_red(text):
            print(colorama.Fore.RED + colorama.Style.BRIGHT + text + colorama.Style.RESET_ALL)
        
        def print_green(text):
            print(colorama.Fore.GREEN + colorama.Style.BRIGHT + text + colorama.Style.RESET_ALL)
        
        average_inclinations = self.average_inclinations
        average_declinations = self.average_declinations
        average_magnitudes = self.average_magnitudes

        print("\n")
        for key in self.get_dirs():
            print_red(f"Data set: {key}")
            print("Average inclination:", end=" ")
            print_green(f"{average_inclinations[key]}")
            print("Average declination:", end=" ")
            print_green(f"{average_declinations[key]}")
            print("Average magnitude:", end=" ")
            print_green(f"{average_magnitudes[key]}")
            print("\n")

    def normalize_vectors(self, vectors: list[Vector3d]):
        return [v/v.magnitude for v in vectors]

    def plot_magnet_vectors_3d(self):
        for key, magnet_vectors in self.aligned_magnet_vectors.items():
            fig = plt.figure(figsize=(10,10))
            vectors = self.normalize_vectors(magnet_vectors)
            zeros = [0]*len(vectors)
            ax = fig.add_subplot(projection='3d')
            ax.set_title(f"Data set: {key}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.quiver(zeros, zeros, zeros, [v.x for v in vectors], [v.y for v in vectors], [v.z for v in vectors], length=1)
            plt.show()
    
    def plot_declination(self):
        for key, magnet_vectors in self.aligned_magnet_vectors.items():
            fig = plt.figure(figsize=(10,10))
            vectors_2d = [Vector3d(x=v.x, y=v.y, z=0) for v in magnet_vectors]
            vectors = self.normalize_vectors(vectors_2d)
            zeros = [0]*len(vectors)
            ax = fig.add_subplot()
            ax.set_title(f"Declination, Data set: {key}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_xlim(-0.3, 0.7)
            ax.set_ylim(0, 1)
            ax.quiver(zeros, zeros, [v.x for v in vectors], [v.y for v in vectors], angles="xy", scale_units="xy", scale=1, label="Magnet vectors (x,y)")

            desired_declination = np.deg2rad(5.3)
            ax.quiver(0, 0, np.sin(desired_declination), np.cos(desired_declination), color="red", scale=2, angles="xy", scale_units="xy", label="Desired declination")

            average_vector = self.average_vector(vectors)
            average_vector = average_vector/average_vector.magnitude
            ax.quiver(0, 0, average_vector.x, average_vector.y, color="green", scale=2, angles="xy", scale_units="xy", label="Average magnet vector")
            ax.legend()

            plt.show()
    
    def plot_inclination(self):
        for key, magnet_vectors in self.aligned_magnet_vectors.items():
            fig = plt.figure(figsize=(10,10))
            vectors_2d = [Vector3d(x=np.sqrt(v.x**2 + v.y**2), y=v.z, z=0) for v in magnet_vectors]
            vectors = self.normalize_vectors(vectors_2d)
            zeros = [0]*len(vectors)
            ax = fig.add_subplot()
            ax.set_title(f"Inclination, Data set: {key}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-1, 0)
            ax.quiver(zeros, zeros, [v.x for v in vectors], [v.y for v in vectors], scale=1, angles="xy", scale_units="xy", label="Magnet vectors (sqrt(x**2 + y**2),z)")

            desired_inclination = np.deg2rad(74 + 55/60)
            ax.quiver(0, 0, np.cos(desired_inclination), -np.sin(desired_inclination), color="red", scale=2, angles="xy", scale_units="xy", label="Desired inclination")

            average_vector = self.average_vector(vectors)
            average_vector = average_vector/average_vector.magnitude
            ax.quiver(0, 0, average_vector.x, average_vector.y, color="green", scale=2, angles="xy", scale_units="xy", label="Average magnet vector")
            ax.legend()

            plt.show()



if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    magnet_calc = MagneticFieldCalc(data_path=file_path)
    magnet_calc.print_info()
    magnet_calc.plot_magnet_vectors_3d()
    magnet_calc.plot_declination()
    magnet_calc.plot_inclination()
