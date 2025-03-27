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
    _time_data: dict[str, list[float]] = {}
    _magnet_error_x: dict[str, float] = {}
    _magnet_error_y: dict[str, float] = {}
    _magnet_error_z: dict[str, float] = {}
    _magnet_errors: dict[str, float] = {}


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
            if not os.path.exists(os.path.join(path, "Magnetometer.csv")):
                continue
            magnet_data = pd.read_csv(os.path.join(path, "Magnetometer.csv"), header=1)

            if not os.path.exists(os.path.join(path, "Accelerometer.csv")):
                acceleration_vecs = [Vector3d(x=0, y=0, z=1) for _ in range(len(magnet_data))]
            else:
                acceleration_data = pd.read_csv(os.path.join(path, "Accelerometer.csv"), header=1)
                acceleration_vecs = [Vector3d.from_list([d[1], d[2], d[3]]) for d in acceleration_data.values]

            magnet_vecs = [Vector3d.from_list([d[1], d[2], d[3]]) for d in magnet_data.values]
            time_data = [d[0] for d in magnet_data.values]
            self._acceleration_data[folder] = acceleration_vecs
            self._magnet_data[folder] = magnet_vecs
            self._time_data[folder] = time_data
    
    def align_vector(self, gravity_vec: Vector3d, magnet_vec: Vector3d) -> Vector3d:
        rot = R.align_vectors(gravity_vec.to_list(), [0,0,1])[0]
        new_vec = magnet_vec.rotate(rot)
        return new_vec
    
    def align_vectors(self):
        for key, data in self.acceleration_data.items():
            aligned_magnet_vecs = [self.align_vector(g_vec, magnet_data) for g_vec, magnet_data in zip(data, self.magnet_data[key])]
            self._aligned_magnet_vectors[key] = aligned_magnet_vecs

    def magnet_error_x(self):
        if self._magnet_error_x:
            return self._magnet_error_x
        errors = {}
        for key, data in self.aligned_magnet_vectors.items():
            x_list = [d.x for d in data]
            errors[key] = np.std(x_list)
        self._magnet_error_x = errors
        return errors

    def magnet_error_y(self):
        if self._magnet_error_y:
            return self._magnet_error_y
        errors = {}
        for key, data in self.aligned_magnet_vectors.items():
            y_list = [d.y for d in data]
            errors[key] = np.std(y_list)
        self._magnet_error_y = errors
        return errors
    
    def magnet_error_z(self):
        if self._magnet_error_z:
            return self._magnet_error_z
        errors = {}
        for key, data in self.aligned_magnet_vectors.items():
            z_list = [d.z for d in data]
            errors[key] = np.std(z_list)
        self._magnet_error_z = errors
        return errors
    
    def magnet_error(self):
        if self._magnet_errors:
            return self._magnet_errors
        errors = {}
        for key, data in self.aligned_magnet_vectors.items():
            x_list = [d.x for d in data]
            y_list = [d.y for d in data]
            z_list = [d.z for d in data]
            errors[key] = np.sqrt(np.std(x_list)**2 + np.std(y_list)**2 + np.std(z_list)**2)
        self._magnet_errors = errors
        return errors
    
    def declination_error(self):
        errors = {}
        for key, data in self.aligned_magnet_vectors.items():
            average_vector = self.average_vector(data)
            x = average_vector.x
            y = average_vector.y

            y_error = (x / (x**2 + y**2)) * self.magnet_error_y()[key]
            x_error = (y / (x**2 + y**2)) * self.magnet_error_x()[key]
            errors[key] = np.sqrt(x_error**2 + y_error**2)
        return errors

    def inclination_error(self):
        errors = {}
        for key, data in self.aligned_magnet_vectors.items():
            average_vector = self.average_vector(data)
            x = average_vector.x
            y = average_vector.y
            z = average_vector.z
            r = np.sqrt(x**2 + y**2)

            x_error = (x*z / (r*(r**2 + z**2))) * self.magnet_error_x()[key]
            y_error = (y*z / (r*(r**2 + z**2))) * self.magnet_error_y()[key]
            z_error = (r / (r**2 + z**2)) * self.magnet_error_z()[key]
            errors[key] = np.sqrt(x_error**2 + y_error**2 + z_error**2)
        return errors

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

    def relative_error(self, value, error):
        return np.abs(error/value) * 100

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
            print("Average components:")
            average_vector = self.average_vector(self.aligned_magnet_vectors[key])
            x, y, z = average_vector.x, average_vector.y, average_vector.z
            ex, ey, ez = self.magnet_error_x()[key], self.magnet_error_y()[key], self.magnet_error_z()[key]
            rex, rey, rez = self.relative_error(x, ex), self.relative_error(y, ey), self.relative_error(z, ez)
            print("\t X: ", end="")
            print_green(f"{x:.3f} ± {ex:.3f} ({rex:.3f}%)")
            print("\t Y: ", end="")
            print_green(f"{y:.3f} ± {ey:.3f} ({rey:.3f}%)")
            print("\t Z: ", end="")
            print_green(f"{z:.3f} ± {ez:.3f} ({rez:.3f}%)")

            print("Average inclination:", end=" ")
            re_incl = self.relative_error(average_inclinations[key], self.inclination_error()[key])
            print_green(f"{average_inclinations[key]:.3f} ± {self.inclination_error()[key]:.3f} ({re_incl:.3f}%)")
            print("Average declination:", end=" ")
            re_decl = self.relative_error(average_declinations[key], self.declination_error()[key])
            print_green(f"{average_declinations[key]:.3f} ± {self.declination_error()[key]:.3f} ({re_decl:.3f}%)")
            print("Average magnitude:", end=" ")
            re_mag = self.relative_error(average_magnitudes[key], self.magnet_error()[key])
            print_green(f"{average_magnitudes[key]:.3f} ± {self.magnet_error()[key]:.3f} ({re_mag:.3f}%)")
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

    def plot_components_single(self):
        first_set = self.get_dirs()[1]
        magnet_vectors = self.aligned_magnet_vectors[first_set]
        time = self._time_data[first_set]
        fig = plt.figure(figsize=(10,10))
        ax = fig.subplots(3,1)

        ax[0].plot(time, [v.x for v in magnet_vectors], "r", label="X")
        ax[1].plot(time, [v.y for v in magnet_vectors], "g", label="Y")
        ax[2].plot(time, [v.z for v in magnet_vectors], "b", label="Z")

        for a in ax:
            a.set_xlabel("Time (s)")
            a.set_ylabel("Magnetic field (uT)")
            a.legend()

        plt.show()
    
if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    magnet_calc = MagneticFieldCalc(data_path=file_path)
    magnet_calc.print_info()
    magnet_calc.plot_magnet_vectors_3d()
    magnet_calc.plot_declination()
    magnet_calc.plot_inclination()
    magnet_calc.plot_components_single()
