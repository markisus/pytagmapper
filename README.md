pytagmapper
====

pytagmapper is a python3 SLAM library for square fiducial tags in the style of AprilTag and ArUco tags. Given a set tags with unknown poses, and a set of pixel detections in images from unknown camera views, pytagmapper will back out both the pose of each tag and the pose of each camera view.  

**Important:** pytagmapper assumes zero distortion (simple pinhole model with 3x3 camera matrix) so if your camera has significant distortion, you will have to undistort images as a preprocessing step. See this [opencv tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html) for more information about how to produce an undistorted 3x3 camera matrix and undistorted images. Be careful to disable any auto-focus features of your camera.  

Demo
====
This [demo](https://github.com/markisus/pytagmapper/tree/main/DEMO.md) creates a map out of the images located in the [example_data](https://github.com/markisus/pytagmapper/tree/main/example_data) folder and runs inside out tracking.

Input Data Directory
====
See [example_data](https://github.com/markisus/pytagmapper/tree/main/example_data) folder for an example scene.
- `image_{id}.png` where id is an integer
- `tags_{id}.txt` where id is an integer, corresponding to `image_{id}.png` (see [Tags Txt Format](#tags-txt-format) below)
- `camera_matrix.txt` row major camera matrix for the camera used to take all the images
- `tag_side_length.txt` physical side length of the tags in meters

    
Prepare Your Own Input Data Using ArUco
====
Create a directory `mkdir ~/my_map_data`.  
Calibrate your camera and save its 3x3 calibration matrix into a file `~/my_map_data/camera_matrix.txt`.  
Print some tags from https://tn1ck.github.io/aruco-print/ and tape them down to a table in various positions. Save the tag side length in meters into a file `my_map_data/tag_side_length.txt`.  
Take undistorted images of this scene using the and save those images as `~/my_map_data/image_0.png`, `~/my_map_data/image_1.png`, ... etc.  
Run `python make_aruco_tag_txts.py ~/my_map_data --show-detections` to generate the `~/my_map_data/tags_0.txt`, `~/my_map_data/tags_1.txt`, .. etc.  
  
Then build and display the map. `build_map.py` takes optional argument `--mode` which can be one of 2d, 2.5d, or 3d.

    cd pytagmapper
    python pytagmapper_tools/build_map.py ~/my_map_data
    python pytagmapper_tools/show_map.py ~/my_map    
    
pytagmapper builds the map by adding in viewpoints to the optimizer one at a time. It's heuristics to know when to advance to the next viewpoint are currently very conservative. Help it along by pressing ctrl+c to advance to the next viewpoint when the current error gets low enough.

# Tags Txt Format
`tags_{id}.txt` is a file containing a list of all tags detected `image_{id}.png`. See [example_data/tags_0.txt](https://github.com/markisus/pytagmapper/blob/main/example_data/tags_0.txt) for an example. If you are using ArUco, you can use the `make_aruco_tag_txts.py` script to generate these tag txts.

    [tag id A]
    [tag top left pixel x] [tag top left pixel y]
    [tag top right pixel x] [tag right pixel y]
    [tag bottom right pixel x] [tag bottom right pixel y]
    [tag bottom left pixel x] [tag bottom left y]
    [tag id B]
    [tag top left pixel x] [tag top left pixel y]
    [tag top right pixel x] [tag right pixel y]
    [tag bottom right pixel x] [tag bottom right pixel y]
    [tag bottom left pixel x] [tag bottom left y]
    ...
    
 Output Map Format
 =====
 `build_map.py` generates a directory containing `map.json` and `viewpoints.json`.  
 
 map.json Schema
 ----
    {
        'tag_side_length': (float),
        'tag_locations': {
            tag_id (str): [x (float), y (float), yaw (float, radians)], # 2d mode
            tag_id (str): [x (float), y (float), z(float), yaw (float, radians)], # 2.5d mode
            tag_id (str): (row major 4x4 pose matrix as list of lists), # 3d mode
            ...
         }
    }
viewpoints.json Schema
----
    {
        image_id (str): (row major 4x4 pose matrix as list of list),
        ...
    }
 
