class SceneDetector:
    """Detects scene changes in an input video"""

    def __init__(self, path):
        """
        @input path <String>: Path to a video file
        """
        super(SceneDetector, self).__init__()
        self.path = path
