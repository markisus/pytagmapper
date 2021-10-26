pytagmapper
====

pytagmapper is a SLAM library for square fiducial tags in the style of AprilTag and ArUco tags. Given a set tags with unknown poses lying in a plane, and a set of pixel detections in images from unknown camera views, pytagmapper will back out both the pose of each tag and the pose of each camera view.

Input Data Directory
====
See [example_data](https://github.com/markisus/pytagmapper/tree/main/example_data) folder for an example scene.
- `image_{id}.png` where id is an integer
- `tags_{id}.txt` where id is an integer, corresponding to `image_{id}.png` (see Tags Txt Format below)
- `camera_matrix.txt` row major camera matrix for the rectified camera used to take all the images
- `tag_side_length.txt` physical side length of the tags in meters

Tags Txt Format
====
`tags_{id}.txt` is a file containing a list of all tags detected `image_{id}.png`

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

Quick Start
====
    git clone https://github.com/markisus/pytagmapper
    cd pytagmapper
    python build_map.py --input-data-dir example_data --output-data-dir example_map
    python show_map.py --map-dir example_map
    
    
