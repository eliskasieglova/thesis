from scripts import downloads
from pathlib import Path

# download specifications
spatial_extent = [16.65, 77.65, 18.4, 78]  # area of Scheelebreen
date_range = ['2018-10-14', '2024-02-02']

# download ATL03, ATL06 and ATL08 data for same region and dates
products = ['ATL03', 'ATL06', 'ATL08']
for product in products:
    outpath = Path(f'downloads/{product}')
    downloads.downloadICESat(product, spatial_extent, date_range, outpath)



