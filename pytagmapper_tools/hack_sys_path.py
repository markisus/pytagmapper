# append repository home directory so that python can find the sibling pytagmapper package
from pathlib import Path
repo_home_dir = str(Path(__file__).parent.parent.absolute())
import sys
sys.path.append(repo_home_dir)
