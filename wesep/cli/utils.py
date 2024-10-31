import argparse


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t",
        "--task",
        choices=[
            "extraction",
        ],
        default="extraction",
        help="task type",
    )
    parser.add_argument(
        "-l",
        "--language",
        choices=[
            # "chinese",
            "english",
        ],
        default="english",
        help="language type",
    )
    parser.add_argument(
        "--bsrnn",
        action="store_true",
        help="whether to use the bsrnn model",
    )
    parser.add_argument(
        "-p", "--pretrain", type=str, default="", help="model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device type (most commonly cpu or cuda,"
        "but also potentially mps, xpu, xla or meta)"
        "and optional device ordinal for the device type.",
    )
    parser.add_argument("--audio_file", help="mixture's audio file")
    parser.add_argument("--audio_file2", help="enroll's audio file")
    parser.add_argument(
        "--resample_rate", type=int, default=16000, help="resampling rate"
    )
    parser.add_argument(
        "--vad", action="store_true", help="whether to do VAD or not"
    )
    parser.add_argument(
        "--output_file",
        default='./extracted_speech.wav',
        help="extracted speech saved in .wav"
    )
    parser.add_argument(
        "--normalize_output",
        default=True,
        help="Control if normalize the ouput audio in .wav"
    )
    args = parser.parse_args()
    return args
