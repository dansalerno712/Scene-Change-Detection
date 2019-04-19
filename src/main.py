import sys
import cv2


def main():
    # get command line arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_video_file>")
        sys.exit(1)

    path = sys.argv[1]

    # grab video
    cap = cv2.VideoCapture(path)

    # playback
    while cap.isOpened():
        ret, frame = cap.read()

        cv2.imshow("frame", frame)

        # how to check for keyboard input with opencv
        # if they press q stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
