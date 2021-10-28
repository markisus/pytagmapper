import argparse
import cv2
from data import *
from geometry import *
from project import *
from inside_out_tracker import InsideOutTracker

def main():
    parser = argparse.ArgumentParser(description='Demo inside out tracking on a map.')
    parser.add_argument('--map-dir', type=str, help='map directory containing map.json', required=True)
    parser.add_argument('--camera-matrix-dir', type=str, help='directory containing camera_matrix.txt', default='.')
    parser.add_argument('--width', type=int, help='camera stream width', default=0)
    parser.add_argument('--height', type=int, help='camera stream width', default=0)
    parser.add_argument('--device', type=int, help='camera device', default=0)
    args = parser.parse_args()

    map_data = load_map(args.map_dir)
    tag_side_length = map_data['tag_side_length']
    camera_matrix = load_camera_matrix(args.camera_matrix_dir)
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    aruco_params = cv2.aruco.DetectorParameters_create()

    # BGR format
    aruco_side_colors = [(0, 0, 255),
                         (0, 255, 0),
                         (255, 0, 0),
                         (0, 255, 255)]


    axes_colors = [(0, 0, 255),
                   (0, 255, 0),
                   (255, 0, 0)]

    
    tag_min_x = float('inf')
    tag_min_y = float('inf')
    tag_max_x = -float('inf')
    tag_max_y = -float('inf')

    for pose_world_tag in map_data['tag_locations'].values():
            x = pose_world_tag[0]
            y = pose_world_tag[1]
            tag_min_x = min(x, tag_min_x)
            tag_min_y = min(y, tag_min_x)
            tag_max_x = max(x, tag_max_x)
            tag_max_y = max(y, tag_max_x)

    map_dimension = max(tag_max_x - tag_min_x, tag_max_y - tag_min_y)
    viz_height = 10
    x_center = (tag_min_x + tag_max_x) / 2
    y_center = (tag_min_y + tag_max_y) / 2
    tdcam_display_width = 800
    tx_world_tdcam = np.array([
        [1,0,0,x_center],
        [0,-1,0,y_center],
        [0,0,-1,viz_height],
        [0,0,0,1]
    ])
    tx_tdcam_world = SE3_inv(tx_world_tdcam)    
    
    f = 2*tdcam_display_width / map_dimension
    tdcam_matrix = np.array([
        [f, 0, tdcam_display_width/2],
        [0, f, tdcam_display_width/2],
        [0, 0, 1],
    ])

    # project camera axes to topdown
    axes = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 1]
    ], dtype=np.float64)
    axes[:3,:] *= 0.1
    
    camera = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    if args.width:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    tracker_initted = False
    tracker = InsideOutTracker(camera_matrix, map_data)

    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        aruco_corners, aruco_ids, aruco_rejected = \
            cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)

        if aruco_corners is None:
            aruco_corners = []
        if aruco_ids is None:
            aruco_ids = []
        else:
            aruco_ids = aruco_ids.flatten()

        # print(aruco_corners)
        # print(aruco_ids)

        aruco_corners_flat = [np.array([c.flatten()]).T for c in aruco_corners]
        for tag_id, acorners in zip(aruco_ids, aruco_corners_flat):
            for i in range(4):
                ni = (i+1)%4
                start = (int(acorners[2*i, 0]), int(acorners[2*i+1, 0]))
                end = (int(acorners[2*ni, 0]), int(acorners[2*ni + 1, 0]))
                color = aruco_side_colors[i]
                cv2.line(frame, start, end, color, thickness=1)

            center = np.sum(acorners.reshape((4,2)), axis=0)/4
            cv2.putText(frame, str(tag_id), (int(center[0]),int(center[1])), 0, 1, (0, 255, 255), thickness=1)

        if len(aruco_ids) and not tracker_initted:
            print("Initting...")
            while tracker.error < 100:
                improved = tracker.update(aruco_ids, aruco_corners_flat, force_update = False)
            tracker_initted = True
            print("Done")

        if tracker_initted:
            improved = tracker.update(aruco_ids, aruco_corners_flat)

        ptag_ids, ptag_corners = tracker.get_projections()
        for tag_id, acorners in zip(ptag_ids, ptag_corners):
            for i in range(4):
                ni = (i+1)%4
                start = (int(acorners[2*i, 0]), int(acorners[2*i+1, 0]))
                end = (int(acorners[2*ni, 0]), int(acorners[2*ni + 1, 0]))
                color = aruco_side_colors[i]
                cv2.line(frame, start, end, color, thickness=1)

            center = np.sum(acorners.reshape((4,2)), axis=0)/4
            cv2.putText(frame, str(tag_id), (int(center[0]),int(center[1])), 0, 1, (0, 0, 255), thickness=1)

        
        scale = 400 / frame.shape[1]
        resize_shape = (int(frame.shape[1] * scale),
                        int(frame.shape[0] * scale))
        resized = cv2.resize(frame, resize_shape)


        topdown_viz = np.zeros((int(tdcam_display_width),
                                int(tdcam_display_width),3), np.uint8)
        topdown_viz[-resize_shape[1]:,-resize_shape[0]:,:] = resized

        # project all tags to topdown
        for tag_id, tx_world_tag in tracker.txs_world_tag.items():
            tx_tdcam_tag = tx_tdcam_world @ tx_world_tag
            corners_td, _, _ = project(tdcam_matrix,
                                       tx_tdcam_tag,
                                       tracker.corners_mat)
            for i in range(4):
                ni = (i+1)%4
                start = (int(corners_td[2*i, 0]), int(corners_td[2*i+1, 0]))
                end = (int(corners_td[2*ni, 0]), int(corners_td[2*ni + 1, 0]))
                color = aruco_side_colors[i]
                cv2.line(topdown_viz, start, end, color, thickness=1)
                
            center = np.sum(corners_td.reshape((4,2)), axis=0)/4
            cv2.putText(topdown_viz, str(tag_id), (int(center[0]),int(center[1])), 0, 1, (0, 0, 255), thickness=1)


        axes_img = tdcam_matrix @ (tx_tdcam_world @ tracker.tx_world_viewpoint @ axes)[:3,:]
        axes_img[:2,:] /= axes_img[2,:]
        axes_origin = axes_img[:,0]

        for i in range(3):
            end = axes_img[:,i+1]
            cv2.line(topdown_viz,
                     (int(axes_origin[0]), int(axes_origin[1])), 
                     (int(end[0]), int(end[1])),
                     axes_colors[i],
                     thickness = 1)

        cv2.putText(topdown_viz, f"q = quit", (10, 30), 0, 0.5, (255, 255, 255), thickness=1)
        cv2.putText(topdown_viz, f"Tracker Error: {tracker.error}", (10, 60), 0, 0.5, (255, 255, 255), thickness=1)

        rotation_y = 90
        cv2.putText(topdown_viz, str("Rotation"), (10, rotation_y), 0, 0.5, (255, 255, 255), thickness=1)
        cv2.putText(topdown_viz, str(tracker.tx_world_viewpoint[0,:]), (10, rotation_y + 15), 0, 0.5, (255, 255, 255), thickness=1)
        cv2.putText(topdown_viz, str(tracker.tx_world_viewpoint[1,:]), (10, rotation_y + 2*15), 0, 0.5, (255, 255, 255), thickness=1)
        cv2.putText(topdown_viz, str(tracker.tx_world_viewpoint[2,:]), (10, rotation_y + 3*15), 0, 0.5, (255, 255, 255), thickness=1)

        xyz_y = rotation_y + 4*15
        cv2.putText(topdown_viz, str("XYZ"), (10, xyz_y), 0, 0.5, (255, 255, 255), thickness=1)
        cv2.putText(topdown_viz, str(tracker.tx_world_viewpoint[:3,3]), (10, xyz_y + 15), 0, 0.5, (255, 255, 255), thickness=1)

        
        cv2.imshow('Inside Out Tracking Demo', topdown_viz)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

