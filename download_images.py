from bing_image_downloader import downloader

# options: set adult_filter_off=True to avoid “safe search”
# force_replace=False so it won’t re‐download existing files
downloader.download(
    "peanut butter dish",
    limit=20,
    output_dir="data/train/peanuts",
    adult_filter_off=True,
    force_replace=False,
    timeout=60
)

downloader.download(
    "slice of cheesecake",
    limit=20,
    output_dir="data/train/dairy",
    adult_filter_off=True,
    force_replace=False,
    timeout=60
)

downloader.download(
    "plain bagel",
    limit=20,
    output_dir="data/train/gluten",
    adult_filter_off=True,
    force_replace=False,
    timeout=60
)

# You can duplicate the above for the `data/val/...` folders (use a different query or slice of the same results).
