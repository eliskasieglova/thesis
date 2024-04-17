import icepyx as ipx
from pathlib import Path
import management
from vars import label, spatial_extent, date_range

def downloadICESat(product, outpath):
    """
    Downloades ICESat-2 granules from EarthData and saves them to the output path.

    Necessary to have EarthData login saved as environmental variables under EARTHDATA_USERNAME
    and EARTHDATA_PASSWORD.

    Params:
    - product
        short name for ICESat-2 product ('ATL03', 'ATL06', 'ATL08')
    - spatial_extent
        bounding box of area ([16.65, 77.65, 18.4, 78])
    - date_range
        list of tuple of begin and end dates (['2018-10-14', '2024-02-02'])
    - outpath
        Path object to where the downloaded data should be saved
    """

    #management.createFolder(Path(f'downloads/{label}/'))

    region_a = ipx.Query(product, spatial_extent, date_range)
    region_a.order_granules(verbose=True, subset=False, email=False)
    region_a.download_granules(outpath)


def downloadSvalbard():

    # download specifications
    products = ['ATL08', 'ATL06']

    for product in products:
        outpath = Path(f'downloads/{product}/')
        downloadICESat(product, outpath)

    return

