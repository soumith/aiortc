import asyncio
import errno
import fractions
import logging
import threading
import time
from typing import Dict, Optional, Set

import numpy as np
import av
from av import AudioFrame, VideoFrame
from av.frame import Frame

from aiortc.contrib.media import PlayerStreamTrack

logger = logging.getLogger(__name__)

VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 30  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)

def player_worker(
    loop, video_track, quit_event
):
    start = time.time()
    timestamp = 0
    color = 0
    while not quit_event.is_set():
        try:
            frame = VideoFrame.from_ndarray(np.full((640, 480, 3), color, dtype=np.uint8))
            timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
            color += 1
            frame.time_base = VIDEO_TIME_BASE
            frame.pts = timestamp
            props = ['height', 'width', 'index', 'interlaced_frame', 'is_corrupt', 'key_frame', 'pts', 'time', 'time_base', ]
        except (av.AVError, StopIteration) as exc:
            if isinstance(exc, av.FFmpegError) and exc.errno == errno.EAGAIN:
                time.sleep(0.01)
                continue
            asyncio.run_coroutine_threadsafe(video_track._queue.put(None), loop)
            break

        asyncio.run_coroutine_threadsafe(video_track._queue.put(frame), loop)

        # show a stable 30fps
        wait = start + (timestamp / VIDEO_CLOCK_RATE) - time.time()
        time.sleep(wait)

class NumpyPlayer:

    def __init__(self, ):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()
        self.__video = PlayerStreamTrack(self, kind="video")
        self._throttle_playback = False

    @property
    def video(self):
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker,
                args=(
                    asyncio.get_event_loop(),
                    self.__video,
                    self.__thread_quit,
                ),
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        if not self.__started and self.__container is not None:
            self.__container.close()
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        logger.debug(f"MediaPlayer {msg}", *args)
