from pathlib import Path

from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

dir = Path("datasets/coco")

urls = [ASSETS_URL + "/coco2017labels-segments.zip"]
download(urls, dir=dir.parent)

urls = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
]
download(urls, dir=dir / "images", threads=3)
