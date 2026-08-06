# Lazy imports to allow sub-packages (e.g. kce) to work without
# installing the full TSE dependency tree (silero_vad, etc.).
try:
    from wesep.cli.extractor import load_model  # noqa
    from wesep.cli.extractor import load_model_local  # noqa
except ImportError:
    pass