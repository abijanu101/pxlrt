from .color import (
    color_range,
    color_novelty,
    kmeans_palette_error,
)

from .edge import (
    neighbor_contrast,
    difference_of_gaussians,
)

from .misc import (
    channel_intensity,
    fft_lowpass,
    regional_isolation,
    pseudo_mutual_information,
)

from .structure import (
    edge_distance_transform,
    connected_components
)

from .variance import (
    block_variance,
    multiscale_block_variance,
)

FILTER_BANK = {
    # -------------------------
    # Color
    # -------------------------
    "color_range": color_range,
    "color_novelty": color_novelty,
    "kmeans_palette_error": kmeans_palette_error,

    # -------------------------
    # Edge
    # -------------------------
    "neighbor_contrast": neighbor_contrast,
    "difference_of_gaussians": difference_of_gaussians,

    # -------------------------
    # Misc
    # -------------------------
    "channel_intensity": channel_intensity,
    "fft_lowpass": fft_lowpass,
    "regional_isolation": regional_isolation,
    "pseudo_mutual_information": pseudo_mutual_information,

    # -------------------------
    # Structure
    # -------------------------
    "edge_distance_transform": edge_distance_transform,
    "connected_components": connected_components,

    # -------------------------
    # Variance
    # -------------------------
    "block_variance": block_variance,
    "multiscale_block_variance": multiscale_block_variance,
}

__all__ = [
    # color
    "color_range",
    "color_novelty",
    "kmeans_palette_error",

    # edge
    "neighbor_contrast",
    "difference_of_gaussians",

    # misc
    "channel_intensity",
    "fft_lowpass",
    "regional_isolation",
    "pseudo_mutual_information",

    # structure
    "edge_distance_transform",
    "connected_components",

    # variance
    "block_variance",
    "multiscale_block_variance",

    # registry
    "FILTER_BANK",
]