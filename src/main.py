import sys
from detector import SceneDetector


def main():
    # get command line arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_video_file>")
        sys.exit(1)

    path = sys.argv[1]

    # create a detector
    detector = SceneDetector(path)

    # run the detector
    detector.detect(method="SSD", a=1, b=3, c=3, s=0.005, display=True)


if __name__ == '__main__':
    main()
