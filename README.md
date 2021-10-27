pytagmapper
====

pytagmapper is a python3 SLAM library for square fiducial tags in the style of AprilTag and ArUco tags. Given a set tags with unknown poses lying in a plane, and a set of pixel detections in images from unknown camera views, pytagmapper will back out both the pose of each tag and the pose of each camera view.

Input Data Directory
====
See [example_data](https://github.com/markisus/pytagmapper/tree/main/example_data) folder for an example scene.
- `image_{id}.png` where id is an integer
- `tags_{id}.txt` where id is an integer, corresponding to `image_{id}.png` (see [Tags Txt Format](#tags-txt-format) below)
- `camera_matrix.txt` row major camera matrix for the camera used to take all the images
- `tag_side_length.txt` physical side length of the tags in meters

Demo
====
    git clone https://github.com/markisus/pytagmapper
    cd pytagmapper
    python build_map.py --input-data-dir example_data --output-data-dir example_map
    python show_map.py --map-dir example_map
    
Prepare Your Own Input Data Using ArUco
====
Create a directory `mkdir ~/my_map_data`.  
Calibrate your camera and save its 3x3 calibration matrix into a file `~/my_map_data/camera_matrix.txt`. pytagmapper assumes zero distortion so if your camera has significant distortion, you will have to undistort as a preprocessing step.  
Print some tags from https://tn1ck.github.io/aruco-print/ and tape them down to a table in various positions. Save the tag side length in meters into a file `my_map_data/tag_side_length.txt`.  
Take undistorted images of this scene using the and save those images as `~/my_map_data/image_0.png`, `~/my_map_data/image_1.png`, ... etc.  
Run `python make_aruco_tag_txts.py --image-dir ~/my_map_data --show_tags` to generate the `~/my_map_data/tags_0.txt`, `~/my_map_data/tags_1.txt`, .. etc.  
  
Then build and display the map.  

    cd pytagmapper
    python build_map.py --input-data-dir ~/my_map_data --output-data-dir ~/my_map
    python show_map.py --map-dir ~/my_map    
    

# Tags Txt Format
`tags_{id}.txt` is a file containing a list of all tags detected `image_{id}.png`. See [example_data/tags_0.txt](https://github.com/markisus/pytagmapper/blob/main/example_data/tags_0.txt) for an example.

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
            tag_id (str): [x (float), y (float), yaw (float, radians)],
            ...
         }
    }
viewpoints.json Schema
----
    {
        image_id (str): (row major 4x4 pose matrix as list of list),
        ...
    }
 
