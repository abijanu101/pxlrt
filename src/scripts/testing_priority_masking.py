import numpy as np
import cv2

DIR = './resources/ashlord00/images'


def playingAround(path, label, scaled_up_by, aoi):
    img = cv2.imread(path)

    canny_thing(img)
    k_means_thing(img, 4)
    cv2.imshow(label, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def k_means_thing(img, k):
    data = img.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
    )

    palette_img = centers[labels.flatten()].reshape(img.shape)

    cv2.imshow("k-means clustered", palette_img.astype(np.uint8))


def canny_thing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 200, 250)

    cv2.imshow("canny edges", edges)


imgs = {
    # Clear Characters and Objects
    DIR + '/464a4159-95fa-492e-9227-517ec3a425b9.png': ("Moyai", 5, []),
    DIR + '/3ed3c7ce-e0ca-4625-b792-2cc86a6632fc.png': ("Silhouette of Man", 5, []),
    DIR + '/648a448f-5584-4ec1-8cf3-237d3cbf5953.png': ("Centauruses", 5, []),

    # Challenging Characters and Objects
    DIR + '/2b569f4c-ec87-4cd1-8481-2e5db5903369.png': ("Zombie", 5, []),
    DIR + '/7dc0dab6-7d41-41a7-a79b-4b7576378032.png': ("Green Girl", 5.5, []),
    DIR + '/0b0611b4-fd91-48f3-a2b7-bb191aeb4e3d.png': ("Large Frog", 5, []),
    DIR + '/1775215d-5a39-4cc7-a4e3-9ff9a032653d.png': ("Sniper Girl", 5, []),

    # Concrete Landscapes
    DIR + '/6cbd9c21-4976-442e-b25f-67cb080e9004.png': ("Ice Blue", 5, []),
    DIR + '/30245fb3-15e3-4e74-a92f-d1c9dbca8ac6.png': ("Destroyed City", 5, []),
    DIR + '/89991a49-4986-4848-b02b-701f031504e2.png': ("Cave", 6, []),
    DIR + '/1763714b-cf5f-405b-9eae-00fb22114227.png': ("Hueco Mundo",6, []),

    # Challenging and Abstract Landscapes
    DIR + '/849b63ae-b47b-4b65-a892-87a32e3ffcd8.png': ("Abstract Dunes", 6, []),
    DIR + '/228661ff-79ae-4c45-bff0-84d0e19207b3.png': ("Abstract Noisy Landscape", 6, []),
    DIR + '/288112c3-64c7-4fef-89f6-6e8e7da48e13.png': ("Abstract Tree", 6, []),

    # Misc.
    DIR + '/01119409-894e-4be0-a0a7-a804acbed38e.png': ("Multi Panel", 6, [])
}

for path, (label, scaling_factor, aoi) in imgs.items():
    playingAround(path, label, scaling_factor, aoi)
