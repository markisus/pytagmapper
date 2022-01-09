import argparse
import glob
import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reindex filenames {prefix}{idx}.{extension} => {prefix}{idx+offset}.{extension}')
    parser.add_argument('directory', type=str)
    parser.add_argument('offset', type=int)
    parser.add_argument('--extension', type=str, default='png')
    parser.add_argument('--prefix', type=str, default='image_')
    args = parser.parse_args()

    prefix_len = len(args.prefix)    

    glob_search = args.directory + "/" + args.prefix + "*." + args.extension
    print(f"globbing for {glob_search}")
    glob_results = glob.glob(glob_search)

    temp_files = []

    for path in glob_results:
        head, tail = os.path.split(path)
        image_name, ext = os.path.splitext(tail)
        image_idx = image_name[prefix_len:]
        parsed_idx = int(image_idx)
        new_idx = parsed_idx + args.offset
        result = os.path.join(head, f"{args.prefix}{new_idx}.{args.extension}.temp")
        print(f"renaming {path} => {result}")
        shutil.copyfile(path, result)
        temp_files.append(result)

    # remove original files
    for path in glob_results:
        os.remove(path)

    # remove temp suffix
    for temp_file in temp_files:
        untemp = temp_file[:-len(".temp")]
        print(f"renaming {temp_file} => {untemp}")
        os.rename(temp_file, untemp)

    
        
    


