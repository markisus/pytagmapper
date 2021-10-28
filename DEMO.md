Demo
====
This demo creates a map out of the images located in the [example_data](https://github.com/markisus/pytagmapper/tree/main/example_data) folder.
![source images screenshot](https://github.com/markisus/pytagmapper/blob/main/source_images.png)  

Commands
----

    git clone https://github.com/markisus/pytagmapper
    cd pytagmapper
    python build_map.py --input-data-dir example_data --output-data-dir example_map
    python show_map.py --map-dir example_map  
    
 Result
 ----
    
![map screenshot](https://github.com/markisus/pytagmapper/blob/main/demo.png)

Inside Out Tracking
=====
After building the map, you can run inside out tracking, see [inside_out_tracker_demo.py](https://github.com/markisus/pytagmapper/blob/main/inside_out_tracker_demo.py).

https://user-images.githubusercontent.com/469689/139331966-fc9a8298-25d5-4b53-8dd2-faf51bb9b8ec.mp4

