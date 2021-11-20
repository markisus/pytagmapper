from hack_sys_path import *

import argparse
from pytagmapper.data import *
import cv2

def main():
    parser = argparse.ArgumentParser(description='Write tags.txt into a directory of images using opencv aruco tag detector.')
    parser.add_argument('image_dir', type=str, help='directory of image_{id}.png images')
    parser.add_argument('--show-detections', '-s', action='store_true', default=False, help='use cv2.imshow to show detected tags')
    args = parser.parse_args()
    
    image_paths = get_image_paths(args.image_dir)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters_create()

    # BGR format
    aruco_side_colors = [(0, 0, 255),
                         (0, 255, 0),
                         (255, 0, 0),
                         (0, 255, 255)]

    for file_id, image_path in image_paths.items():
        image = cv2.imread(image_path)
        aruco_corners, aruco_ids, aruco_rejected = \
            cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

        with open(os.path.join(args.image_dir, f"tags_{file_id}.txt"), "w") as f:
            for tag_idx, tag_id in enumerate(aruco_ids):
                tag_id = tag_id[0]
                acorners = aruco_corners[tag_idx][0]
                f.write(f"{tag_id}\n")
                f.write(f"{acorners[0][0]} {acorners[0][1]}\n")
                f.write(f"{acorners[1][0]} {acorners[1][1]}\n")
                f.write(f"{acorners[2][0]} {acorners[2][1]}\n")
                f.write(f"{acorners[3][0]} {acorners[3][1]}\n")

                if args.show_detections:
                    for i in range(4):
                        start = (int(acorners[i][0]), int(acorners[i][1]))
                        end = (int(acorners[(i+1)%4][0]), int(acorners[(i+1)%4][1]))
                        color = aruco_side_colors[i]
                        cv2.line(image, start, end, color, thickness=2)

                    center = np.sum(acorners, axis=0)/4
                    cv2.putText(image, str(tag_id), (int(center[0]),int(center[1])), 0, 2, (0, 255, 255), thickness=5)

        if args.show_detections:
            scale = 900 / image.shape[1]
            resize_shape = (int(image.shape[1] * scale),
                            int(image.shape[0] * scale))
            resized = cv2.resize(image, resize_shape)
            cv2.imshow(image_path, resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
