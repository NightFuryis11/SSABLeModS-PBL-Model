from pint import Quantity
import numpy as np

von_Karman = np.float64(Quantity(0.35, "dimensionless").to_base_units().m)                            # von Karman constant
g_0 = np.float64(Quantity(9.81, "m s^(-2)").to_base_units().m)                                        # Standard gravity
c_p = np.float64(Quantity(1005.7, "J K^(-1) kg^(-1)").to_base_units().m)                              # Specific heat of air at constant pressure
e_s_tp = np.float64(Quantity(661, "Pa").to_base_units().m)                                            # Saturation vapor pressure at the triple point of water
R_v = np.float64(Quantity(461.5, "J K^(-1) kg^(-1)").to_base_units().m)                               # Water vapor gas constant
R_d = np.float64(Quantity(287.5, "J K^(-1) kg^(-1)").to_base_units().m)                               # Dry air gas constant
T_tp = np.float64(Quantity(273.16, "K").to_base_units().m)                                            # Temperature at the triple point of water
L_tp = np.float64(Quantity(2.501*10**6, "J kg^(-1)").to_base_units().m)                               # Latent heat of vaporization at the triple point of water
c_pv = np.float64(Quantity(1864, "J K^(-1) kg^(-1)").to_base_units().m)                               # Specific heat of water vapor at constant pressure
c_pl = np.float64(Quantity(4184, "J K^(-1) kg^(-1)").to_base_units().m)                               # Specific heat of liquid water at constant pressure
p_0 = np.float64(Quantity(100000, "Pa").to_base_units().m)                                            # Standard reference pressure
omega = np.float64(Quantity(-7.292*10**(-5), "s^(-1)").to_base_units().m)                             # Planetary rotation rate
k_m_z_i = np.float64(Quantity(0.001, "m^(2) s^(-1)").to_base_units().m)                               # Eddy momentum diffusivity at the top of the boundary layer
k_h_z_i = np.float64(Quantity(0.001, "m^(2) s^(-1)").to_base_units().m)                               # Eddy heat diffusivity at the top of the boundary layer
S_0 = np.float64(Quantity(1360, "W m^(-2)").to_base_units().m)                                        # Average incoming solar shortwave radiation at the top of the atmosphere
Stefan_Boltzmann = np.float64(Quantity(5.670 * 10**(-8), "W m^(-2) K^(-4)").to_base_units().m)        # Stefan-Boltzman radiative constant
Planck = np.float64(Quantity(6.62607015 * 10**(-34), "J s").to_base_units().m)                        # Planck's radiative constant
c = np.float64(Quantity(299792458, "m s^(-1)").to_base_units().m)                                     # The speed of light
Boltzmann = np.float64(Quantity(1.380649 * 10**(-23), "J K^(-1)").to_base_units().m)                  # Boltzman radiative constant
r_sun = np.float64(Quantity(695700000, "m").to_base_units().m)                                        # Radius of the Sun
D_sun = np.float64(Quantity(149600000000, "m").to_base_units().m)                                     # Mean distance from the Earth to the Sun