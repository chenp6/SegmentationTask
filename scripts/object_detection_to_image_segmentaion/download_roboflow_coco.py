"""
Backward-compatible wrapper for the shared Roboflow COCO downloader.

Prefer:
    python -m scripts.tools.download_roboflow_coco ...

This legacy module path is kept so existing commands and README examples
continue to work.
"""

from scripts.tools.download_roboflow_coco import (  # noqa: F401
    CREDENTIALS_FILE,
    DEFAULT_API_URL,
    download_dataset,
    get_download_url,
    load_credentials,
    main,
    parse_universe_url,
)


if __name__ == "__main__":
    main()
