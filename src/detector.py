import cv2
import matplotlib.pyplot as plt
import numpy
from datetime import datetime


class SceneDetector:

    """A class to implement scene change detection based on this
    paper, with some additions
    https://www.researchgate.net/profile/Anastasios_Dimou/publication/249863763_SCENE_CHANGE_DETECTION_FOR_H264_USING_DYNAMIC_THRESHOLD_TECHNIQUES/links/00b4952d3b4ae69753000000/SCENE-CHANGE-DETECTION-FOR-H264-USING-DYNAMIC-THRESHOLD-TECHNIQUES.pdf

    Attributes:
        input_path (string): The path to a video file
        output_dir (string): The directory to save output files
    """

    def __init__(self, input_path, output_dir):
        super(SceneDetector, self).__init__()
        self.input_path = input_path
        self.output_dir = output_dir

    def __calculate_mean(self, window):
        """Calculate the mean of the sldiding window of frame
        data. Ignores all NaNs

        Args:
            window (numpy Array): Array of frame data

        Returns:
            Floar: The mean of the window
        """
        # use nanmean because the first frames will have nans
        return numpy.nanmean(window)

    def __calculate_sd(self, window,):
        """Calculates the standard deviation of the sliding
        window of frame data

        Args:
            window (numpy Array): Array of frame data

        Returns:
            Float: The standard deviation of the frame data
        """
        # use nanstd because the first frames will have nans
        return numpy.nanstd(window, ddof=1)

    def __calculate_basic_threshold(self, window, a, b, c):
        """Calculates the threshold if there hasn't been a scene
        change detected recently

        Args:
            window (numpy Array): Array of frame data
            a (Float): a coefficient from the paper
            b (Float): b coefficient from the paper
            c (Float): c coefficient from the paper

        Returns:
            Float: The dynamic threshold to use to check for
            scene changes
        """
        mean = self.__calculate_mean(window)
        sd = self.__calculate_sd(window)

        # window[0] is the most recent frame's data
        return a * window[0] + b * mean + c * sd

    def __calculate_decay_threshold(self, last_change_frame_value, s, last_change_frame, current_frame):
        """Calculates the threshold if a scene change has recently
        been detected

        Args:
            last_change_frame_value (Float): The similarity value
            from the last detected frame pair
            s (Float): s coefficient from the paper
            last_change_frame (Int): The number of the last frame
            that a scene was detected on
            current_frame (Int): The number of the current frame

        Returns:
            Float: The dynamic theshold to use to check for scene
            changes
        """
        return last_change_frame_value * numpy.exp((s * -1) * (current_frame - last_change_frame))

    def __calculate_SAD(self, frame1, frame2):
        """Calculates the sum of absolute differences between
        two images

        Args:
            frame1 (numpy 2D Array): The first image
            frame2 (numpy 2D Array): The second image

        Returns:
            Float: A number that represents the similarity between
            the two images through SAD
        """
        return numpy.sum(numpy.abs(frame1[:, :] - frame2[:, :]))

    def __calculate_SSD(self, frame1, frame2):
        """Calculates the sum of squared differences between
        two images

        Args:
            frame1 (numpy 2D Array): The first image
            frame2 (numpy 2D Array): The second image

        Returns:
            Float: A number that represnet the similarity between
            the two images through SSD
        """
        return numpy.sum((frame1[:, :] - frame2[:, :])**2)

    def __calculate_MAD(self, frame1, frame2):
        """Calculates the mean absolute difference between
        two images

        Args:
            frame1 (numpy 2D Array): The first image
            frame2 (numpy 2D Array): The second image

        Returns:
            Float: A number that represnet the similarity between
            the two images through MAD
        """
        return self.__calculate_SAD(frame1, frame2) / (frame1.size ** 2)

    def __calculate_CORR(self, frame1, frame2):
        """Calculates the pearson's correlation coefficient between
        two images

        Args:
            frame1 (numpy 2D Array): The first image
            frame2 (numpy 2D Array): The second image

        Returns:
            Float: A number that represnet the similarity between
            the two images through correlation
        """
        # use flat images and [0][1] gets the actual coefficient
        # from the returned matrix
        return numpy.corrcoef(frame1.flat, frame2.flat)[0][1]

    def __get_difference_method(self, key):
        """Given a key, returns the function that calculates
        image similarity that corresponds to that key

        Args:
            key (String): which similarity function you want to use

        Returns:
            Function: A similarity function
        """
        methods = {
            "SAD": self.__calculate_SAD,
            "SSD": self.__calculate_SSD,
            "MAD": self.__calculate_MAD,
            "CORR": self.__calculate_CORR
        }
        return methods[key]

    def __output(self, output_data):
        """Outputs detectiond data to a text file

        Args:
            output_data (Array): An array of tuples in the form of
            (frame before change, frame of change)
        """
        file_name = self.output_dir + "/detection-" + \
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".txt"

        with open(file_name, "w") as f:
            f.write("Scene changes:\n")
            for pair in output_data:
                f.write("Scene change detected between frames " +
                        str(pair[0]) + " and " + str(pair[1]) + "\n")

    def detect(self, window_size=20, method="SAD", a=-1, b=2, c=2, s=0.02, k=20, display=False, output=False):
        """Detects scene changes in a video

        Args:
            window_size (int, optional): How many frames should be kept
            in the sliding window array
            method (str, optional): Which image similarity method should be used
            Options: SAD, SSD, MAD, CORR
            a (float, optional): a coefficient from the paper
            b (float, optional): b coefficient from the paper
            c (float, optional): c coefficient from the paper
            s (float, optional): s coefficient from the paper
            k (int, optional): k coefficient from the paper
            AKA how many frames to use the decay threshold for after
            detecting a change
            display (bool, optional): Whether or not to display data
            while/after detecting. If true, this will show
            a) The current frame from the video
            b) The preceding frame and the frame that a change
            was detected on
            c) A graph plotting frame similarity and threshold data
            output (bool, optional): whether or not to output data.
            This will output a text file saying between which frames
            a scene change was found
        """
        # create frame buffer
        window = numpy.full(window_size, numpy.nan)

        # create arrays to hold graphing values if necessary
        if display:
            thresh_vals = []
            frame_vals = []

        if output:
            output_data = []

        # we need to keep the previous_frame
        previous_frame = None

        # keep track of last detected frame information
        change_detected = False
        decay_counter = 0
        last_change_frame = -1
        last_change_frame_value = 0

        # keep track of how many frames weve seen so far
        frame_count = 0

        # grab video
        cap = cv2.VideoCapture(self.input_path)

        # playback
        while cap.isOpened():
            # grab new frame
            frame_count += 1
            ret, frame = cap.read()

            # break if no frame was grabbed
            if not ret:
                break

            # convert frame to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # show current frame if necessary
            if display:
                cv2.imshow("f", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # make sure we've seen at least one frame so far
            if previous_frame is not None:
                # calculate threshold
                if change_detected:
                    threshold = self.__calculate_decay_threshold(
                        last_change_frame_value, s, last_change_frame, frame_count)

                    # decrement decay counter
                    decay_counter -= 1

                    # if we reach the end, stop using decay
                    if decay_counter <= 0:
                        change_detected = False
                else:
                    threshold = self.__calculate_basic_threshold(
                        window, a, b, c)

                # calculate value for current frame
                frame_val = self.__get_difference_method(
                    method)(frame, previous_frame)

                # append graphing data if necessary
                if display:
                    thresh_vals.append(threshold)
                    frame_vals.append(frame_val)

                # scene change was detected
                if frame_val > threshold:
                    # show differing frames if necessary
                    if display:
                        cv2.imshow("previous_frame", previous_frame)
                        cv2.imshow("current_frame", frame)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    if output:
                        output_data.append((frame_count - 1, frame_count))

                    # set values for the decay function
                    change_detected = True
                    decay_counter = k
                    last_change_frame = frame_count
                    last_change_frame_value = frame_val

                # remove oldest frame
                window = numpy.delete(window, -1)

                # add current frame to buffer
                window = numpy.insert(window, 0, frame_val)

            # update previous frame
            previous_frame = frame

        # clean up
        cap.release()
        cv2.destroyAllWindows()

        # graph if necessary
        if display:
            plt.plot(thresh_vals)
            plt.plot(frame_vals)
            plt.show()

        # output if necessary
        if output:
            self.__output(output_data)
