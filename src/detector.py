import cv2
import matplotlib.pyplot as plt
import numpy


class SceneDetector:
    def __init__(self, path):
        super(SceneDetector, self).__init__()
        self.path = path

    def __calculate_mean(self, window):
        return numpy.nanmean(window)

    def __calculate_sd(self, window, mean):
        return numpy.nanstd(window, ddof=1)

    def __calculate_basic_threshold(self, window, a, b, c):
        mean = self.__calculate_mean(window)
        sd = self.__calculate_sd(window, mean)

        return a * window[0] + b * mean + c * sd

    def __calculate_decay_threshold(self, last_change_frame_value, s, last_change_frame, current_frame):
        return last_change_frame_value * numpy.exp((s * -1) * (current_frame - last_change_frame))

    def __calculate_SAD(self, frame1, frame2):
        return numpy.sum(numpy.abs(frame1[:, :] - frame2[:, :]))

    def __calculate_SSD(self, frame1, frame2):
        return numpy.sum((frame1[:, :] - frame2[:, :])**2)

    def __calculate_MAD(self, frame1, frame2):
        return self.__calculate_SAD(frame1, frame2) / frame1.size

    def __calculate_CORR(self, frame1, frame2):
        return numpy.corrcoef(frame1.flat, frame2.flat)[0][1]

    def __get_difference_method(self, key):
        methods = {
            "SAD": self.__calculate_SAD,
            "SSD": self.__calculate_SSD,
            "CORR": self.__calculate_CORR
        }
        return methods[key]

    def detect(self, window_size=20, method="SAD", a=-1, b=2, c=2, s=0.02, k=20, display=False):
        # create frame buffer
        window = numpy.full(window_size, numpy.nan)

        if display:
            thresh_vals = []
            frame_vals = []

        # we need to keep the previous_frame
        previous_frame = None

        # keep track of last detected frame
        change_detected = False
        decay_counter = 0
        last_change_frame = -1
        last_change_frame_value = 0
        frame_count = 0

        # grab video
        cap = cv2.VideoCapture(self.path)

        # playback
        while cap.isOpened():
            frame_count += 1
            ret, frame = cap.read()

            # convert frame to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if display:
                cv2.imshow("f", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if previous_frame is not None:
                # remove oldest frame
                window = numpy.delete(window, -1)

                # calculate threshold
                if change_detected:
                    threshold = self.__calculate_decay_threshold(
                        last_change_frame_value, s, last_change_frame, frame_count)
                    decay_counter -= 1

                    if decay_counter <= 0:
                        change_detected = False
                else:
                    threshold = self.__calculate_basic_threshold(
                        window, a, b, c)

                # calculate value for current frame
                frame_val = self.__get_difference_method(
                    method)(frame, previous_frame)

                if display:
                    thresh_vals.append(threshold)
                    frame_vals.append(frame_val)

                if frame_val > threshold:
                    # do something
                    if display:
                        cv2.imshow("previous_frame", previous_frame)
                        cv2.imshow("current_frame", frame)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                    change_detected = True
                    decay_counter = k
                    last_change_frame = frame_count
                    last_change_frame_value = frame_val

                # add current fram to buffer
                window = numpy.insert(window, 0, frame_val)

            previous_frame = frame

        # cleanup
        if display:
            plt.plot(thresh_vals)
            plt.plot(frame_vals)
            plt.show()
        cap.release()
