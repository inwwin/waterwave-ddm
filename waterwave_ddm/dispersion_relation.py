import numpy as np


class BaseDispersionRelation:
    """A base class for all dispersion relation"""

    def phase_velocity(self, wavenumber):
        raise NotImplementedError

    def group_velocity(self, wavenumber):
        raise NotImplementedError

    def frequency(self, wavenumber):
        return wavenumber * self.phase_velocity(wavenumber)


class DeepStillWaterDispersionRelation(BaseDispersionRelation):
    """
    The theoretical dispersion relation for water-air interface wave
    under no wind and deep water approximation (i.e. kH >> 1)
    """

    def __init__(self, air_density=1.2466, water_density=999.70, surface_tension=7.42e-2):
        """
        Initialize the dispersion relation with water-air parameters.
        The default is for dry-air and pure water at 1 atm and 10 degrees celcius.
        """
        self.gravitational_acceleration = 9.81
        self.surface_tension = surface_tension
        self.air_density = air_density
        self.water_density = water_density

        self.specific_air_density = self.air_density / self.water_density
        self.gravity_term = self.gravitational_acceleration \
            * (1 - self.specific_air_density) / (1 + self.specific_air_density)
        self.surface_tension_term = self.surface_tension / (self.water_density + self.air_density)

    def phase_velocity(self, wavenumber):
        sign = np.sign(wavenumber)
        wavenumber = np.abs(wavenumber)
        return sign * np.sqrt(self.gravity_term / wavenumber + self.surface_tension_term * wavenumber)

    def group_velocity(self, wavenumber):
        sign = np.sign(wavenumber)
        wavenumber = np.abs(wavenumber)
        return sign * (self.gravity_term + 3 * self.surface_tension_term * wavenumber ** 2) \
            / (2 * self.frequency(wavenumber))
