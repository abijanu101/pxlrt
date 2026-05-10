import numpy as np
import cv2
import src.preprocessing.masker.filter_bank as fb
import logging

def generate_priority_mask(img: np.ndarray, save_path: str = None) -> np.ndarray:
    """
    Returns a heatmap for areas of relative importance based on color novelty.
    Areas with colors far from the mean are considered more 'novel'.
    """
    weighted_maps = [
        (1.2, fb.fft_lowpass(img)),
        (1.3, fb.pseudo_mutual_information(img)),

        (1.0, fb.kmeans_palette_error(img)),
        (1.2, fb.color_novelty(img)),
        (1.2, fb.color_range(img)),

        (1.0, fb.difference_of_gaussians(img)),
        (1.8, fb.regional_isolation(img)),
        (1.7, fb.edge_distance_transform(img)),
    ]

    weighted_maps.extend([
        (0.8, c)
        for c in fb.channel_intensity(img)
    ])

    accum = np.zeros_like(weighted_maps[0][1], dtype=np.float32)
    for weight, filter in weighted_maps:
        accum += weight * fb.multiscale_block_variance(filter)
        
    final_mask = accum / (accum.max() + 1e-6)

    # Estimate detected regions by thresholding the mask
    threshold_val = 0.25 # arbitrary hot threshold
    binary_mask = (final_mask > threshold_val).astype(np.uint8) * 255
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(binary_mask)
    num_regions = max(0, num_labels - 1) # subtract 1 for background
    
    logging.info(f"Masker detected {num_regions} distinct priority regions above threshold.")

    if save_path:
        # Convert mask (0-1) to an 8-bit heatmap image
        heatmap = cv2.applyColorMap((final_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(save_path, heatmap)
        logging.info(f"Saved priority mask debug image to: {save_path}")

    return final_mask

# playground

if __name__ == '__main__':
    DIR = './resources/ashlord00/images'
    imgs = [
        DIR + '/464a4159-95fa-492e-9227-517ec3a425b9.png',
        DIR + '/3ed3c7ce-e0ca-4625-b792-2cc86a6632fc.png',
        DIR + "/648a448f-5584-4ec1-8cf3-237d3cbf5953.png",
        DIR + '/2b569f4c-ec87-4cd1-8481-2e5db5903369.png',
        DIR + '/7dc0dab6-7d41-41a7-a79b-4b7576378032.png',
        DIR + '/0b0611b4-fd91-48f3-a2b7-bb191aeb4e3d.png',
        DIR + '/1775215d-5a39-4cc7-a4e3-9ff9a032653d.png',
        DIR + '/6cbd9c21-4976-442e-b25f-67cb080e9004.png',
        DIR + '/30245fb3-15e3-4e74-a92f-d1c9dbca8ac6.png',
        DIR + '/89991a49-4986-4848-b02b-701f031504e2.png',
        DIR + '/1763714b-cf5f-405b-9eae-00fb22114227.png',
        DIR + '/849b63ae-b47b-4b65-a892-87a32e3ffcd8.png',
        DIR + '/228661ff-79ae-4c45-bff0-84d0e19207b3.png',
        DIR + '/288112c3-64c7-4fef-89f6-6e8e7da48e13.png',
        DIR + "/0b4f83ac-fc76-434a-95d8-853a472f386a.png",
        
        DIR + "/02ff04e4-f65c-44bd-8c7f-9d9a41ecdb79.png",
        DIR + "/e0d0c74b-d6ba-42d5-a6c1-bbaedc14ff7e.png",
        DIR + "/ffc6459a-b287-4707-bda3-31e8e8168d20.png",
        DIR + "/40675b7e-5c87-45b2-b3ef-dfd63dc5ea19.png",
        DIR + "/01119409-894e-4be0-a0a7-a804acbed38e.png"
    ]
    import os
    imgs = [DIR + '/' + i for i in os.listdir(DIR)]

    for path in imgs:
        img = cv2.imread(path)
        pm = generate_priority_mask(img)
        cv2.imshow('original', img)
        cv2.imshow('priority mask', pm)
        cv2.waitKey(0)
        cv2.destroyAllWindows()