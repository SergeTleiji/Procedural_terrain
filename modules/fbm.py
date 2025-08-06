"""
fbm.py
------
Generates procedural heightmaps using Fractal Brownian Motion (FBM) noise
on the GPU via NVIDIA Warp.

Purpose in Pipeline:
    - Used in Step 3 of `main.py` to produce macro- and meso-scale terrain features.
    - Outputs a 2D NumPy array (.npy) representing the heightmap.

How It Works:
    1. A Warp kernel (`fbm_kernel`) computes noise values for each pixel coordinate.
    2. Two noise layers are generated:
        - Macro FBM: large-scale terrain variation (mountains, hills).
        - Meso FBM: smaller-scale variation (mounds, undulations).
    3. The macro and meso layers are summed to form the final heightmap.
    4. The heightmap is saved to disk as a `.npy` file for later mesh generation.

Inputs:
    - x_start_m, y_start_m: world-space starting coordinates (meters).
    - width_m: terrain width in meters.
    - res: resolution (pixels per side).
    - macro/meso parameters: octaves, frequency, amplitude, and dropoff factors.
    - seed: RNG seed for reproducibility.

Outputs:
    - `.npy` file containing summed macro + meso height values.

Dependencies:
    - numpy
    - NVIDIA Warp (wp)
    - Called by `main.py` → `NoiseGenerator.generate_and_save_area()`

Example:
    noise = NoiseGenerator(macro_octaves=3, meso_octaves=2, seed=42)
    noise.generate_and_save_area(0, 0, 500, 1024, "output/heightmap.npy")
"""

import numpy as np
import warp as wp


# === GPU Noise Kernel ===
# Each thread computes noise for one (i, j) pixel coordinate.
@wp.kernel
def fbm_kernel(
    kernel_seed: int,
    frequency: float,
    amplitude: float,
    x: wp.array(dtype=float),
    y: wp.array(dtype=float),
    z: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    state = wp.rand_init(kernel_seed)

    # Convert pixel indices to scaled noise coordinates
    p = frequency * wp.vec2(x[j], y[i])

    # Compute noise value and scale by amplitude
    n = amplitude * wp.noise(state, p)

    # Accumulate result (allows layering octaves)
    z[i, j] += n


class NoiseGenerator:
    """
    Generates macro + meso FBM noise heightmaps using Warp kernels.
    """

    def __init__(
        self,
        macro_octaves=2,
        meso_octaves=3,
        macro_freq=0.01,
        meso_freq=0.033,
        frequency_multiplier=1,
        macro_amplitude=15,
        meso_amplitude=5,
        macro_amplitude_dropoff=3,
        meso_amplitude_dropoff=2,
        seed=435,
    ):
        # Store noise configuration parameters
        self.macro_octaves = macro_octaves
        self.meso_octaves = meso_octaves
        self.macro_freq = macro_freq
        self.meso_freq = meso_freq
        self.frequency_multiplier = frequency_multiplier
        self.macro_amplitude = macro_amplitude
        self.meso_amplitude = meso_amplitude
        self.macro_amplitude_dropoff = macro_amplitude_dropoff
        self.meso_amplitude_dropoff = meso_amplitude_dropoff
        self.seed = seed

    # === Generate macro-scale terrain layer ===
    def generate_macro_fbm(self, x_start_m, y_start_m, width_m, res):
        """
        Produces a large-scale FBM noise layer (mountains/hills).
        """
        frequency = self.macro_freq
        amplitude = self.macro_amplitude

        # Coordinate grid in world space
        self.min_x, self.max_x = x_start_m, x_start_m + width_m
        self.min_y, self.max_y = y_start_m, y_start_m + width_m

        x = np.linspace(self.min_x, self.max_x, res)
        y = np.linspace(self.min_y, self.max_y, res)

        # Warp-compatible arrays
        self.x = wp.array(x, dtype=float)
        self.y = wp.array(y, dtype=float)
        self.macro_pixel_values = wp.zeros((res, res), dtype=float)

        # Accumulate octaves
        for _ in range(self.macro_octaves):
            wp.launch(
                kernel=fbm_kernel,
                dim=(res, res),
                inputs=[self.seed, frequency, amplitude, self.x, self.y],
                outputs=[self.macro_pixel_values],
            )
            frequency *= self.frequency_multiplier
            amplitude *= self.macro_amplitude_dropoff

    # === Generate meso-scale terrain layer ===
    def generate_meso_fbm(self, x_start_m, y_start_m, width_m, res):
        """
        Produces a medium-scale FBM noise layer (mounds/undulations).
        """
        frequency = self.meso_freq
        amplitude = self.meso_amplitude

        # Coordinate grid in world space
        self.min_x, self.max_x = x_start_m, x_start_m + width_m
        self.min_y, self.max_y = y_start_m, y_start_m + width_m

        x = np.linspace(self.min_x, self.max_x, res)
        y = np.linspace(self.min_y, self.max_y, res)

        # Warp-compatible arrays
        self.x = wp.array(x, dtype=float)
        self.y = wp.array(y, dtype=float)
        self.meso_pixel_values = wp.zeros((res, res), dtype=float)

        # Accumulate octaves
        for _ in range(self.meso_octaves):
            wp.launch(
                kernel=fbm_kernel,
                dim=(res, res),
                inputs=[self.seed, frequency, amplitude, self.x, self.y],
                outputs=[self.meso_pixel_values],
            )
            frequency *= self.frequency_multiplier
            amplitude *= self.meso_amplitude_dropoff

    # === Save noise layers to disk ===
    def save(self, filename="output/noise.npy"):
        """
        Saves the combined macro + meso noise to disk as .npy.
        """
        macro_pixels = self.macro_pixel_values.numpy()
        meso_pixels = self.meso_pixel_values.numpy()
        pixels = macro_pixels + meso_pixels
        np.save(filename, pixels)
        print(f"Saved FBM noise to {filename}")

    # === Combined generation convenience method ===
    def generate_and_save_area(self, x_start_m, y_start_m, width_m, res, output_path):
        """
        Generates both macro + meso layers, sums them, and saves result.
        """
        self.generate_macro_fbm(x_start_m, y_start_m, width_m, res)
        self.generate_meso_fbm(x_start_m, y_start_m, width_m, res)
        self.save(output_path)
