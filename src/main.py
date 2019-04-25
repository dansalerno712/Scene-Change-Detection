import sys
from detector import SceneDetector
import os


def main():
    # get command line arguments
    if len(sys.argv) != 3:
        print("Usage: python main.py <path_to_video_file> <output_dir>")
        sys.exit(1)

    path = sys.argv[1]
    output_dir = sys.argv[2]

    # make sure directory exists
    if (not os.path.isdir(output_dir)):
        print("Error: output directory is not a directory")
        return

    # create a detector
    detector = SceneDetector(path, output_dir)

    # run the detector
    detector.detect(method="SSD", a=1, b=3, c=3, s=0.005, display=True, output=False)


if __name__ == '__main__':
    main()
