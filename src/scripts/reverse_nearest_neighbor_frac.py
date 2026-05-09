import numpy as np
import cv2

DIR = './resources/ashlord00/images'

def show_different_scales(path):
    img = cv2.imread(path)
    h, w, _ = img.shape

    def scale_rescale(img, scaled_up_by):
        return cv2.resize(
            cv2.resize(
                img,
                (round(w / scaled_up_by), round(h / scaled_up_by)),
                interpolation=cv2.INTER_NEAREST
            ),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    potential_sizes = [5.21]
    
    for s in potential_sizes:
        cv2.imshow(f'rescaled x{s}', scale_rescale(img, s))
    cv2.imshow('original', img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


imgs = [
    # Clear Characters and Objects
    DIR + '/464a4159-95fa-492e-9227-517ec3a425b9.png',
    DIR + '/3ed3c7ce-e0ca-4625-b792-2cc86a6632fc.png',
    DIR + "/648a448f-5584-4ec1-8cf3-237d3cbf5953.png",
    # Challenging Characters and Objects

    DIR + '/2b569f4c-ec87-4cd1-8481-2e5db5903369.png',
    DIR + '/7dc0dab6-7d41-41a7-a79b-4b7576378032.png',
    DIR + '/0b0611b4-fd91-48f3-a2b7-bb191aeb4e3d.png',
    DIR + '/1775215d-5a39-4cc7-a4e3-9ff9a032653d.png',

    # Concrete Landscapes
    DIR + '/6cbd9c21-4976-442e-b25f-67cb080e9004.png',
    DIR + '/30245fb3-15e3-4e74-a92f-d1c9dbca8ac6.png',
    DIR + '/89991a49-4986-4848-b02b-701f031504e2.png',
    DIR + '/1763714b-cf5f-405b-9eae-00fb22114227.png',

    # Challenging and Abstract Landscapes
    DIR + '/849b63ae-b47b-4b65-a892-87a32e3ffcd8.png',
    DIR + '/228661ff-79ae-4c45-bff0-84d0e19207b3.png',
    DIR + '/288112c3-64c7-4fef-89f6-6e8e7da48e13.png',

    # Misc.
    DIR + '/01119409-894e-4be0-a0a7-a804acbed38e.png'
]

for path in imgs:
    show_different_scales(path)
