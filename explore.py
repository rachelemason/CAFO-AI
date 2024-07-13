"""
Contains code used for data exploration
"""

import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def describe_data(df, animal_type, size_range, geom_kw, quantity_kw):

  print(f"There are {df.shape[0]} entries in the {animal_type} spreadsheet")

  x = df[~df[geom_kw].isnull()]
  print(f"{len(x)} farms have valid coordinates")

  x = df[df[quantity_kw] == 0]
  print(f"{len(x)} farms report 0 {animal_type}")

  print(f"The largest farm reports {df[quantity_kw].max()} {animal_type}")

  x = df[(df[quantity_kw] < size_range[0]) & (df[quantity_kw] > 0)]
  print(f"There are {len(x)} farms with 0 < {animal_type} < {size_range[0]}")
  print(f"  - They account for {x[quantity_kw].sum()} {animal_type}")

  x = df[(df[quantity_kw] > size_range[0]) & (df[quantity_kw] < size_range[1])]
  print(f"There are {x.shape[0]} farms with  {size_range[0]} < {animal_type} < {size_range[1]}")
  print(f"  - They account for {x[quantity_kw].sum()} {animal_type} \n")


def histo(df, ax_a, ax_b, animal, quantity_kw):

    # Histogram
    to_show = df[df[quantity_kw] > 0]
    n, bins, _ = ax_a.hist(to_show[quantity_kw], bins=[1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],\
                          log=True, histtype='step', color='k')
    ax_a.set_xscale('log')
    ax_a.set_xlabel(f'Number of {animal} on farm')
    ax_a.set_ylabel('Frequency')

    # Cumulative numbers of animals on small --> large farms
    cumulative = []
    for n, bin in enumerate(bins[:-1]):
      temp = df[(df[quantity_kw] >= bin) & (df[quantity_kw] < bins[n+1])]
      temp = temp[quantity_kw].sum()
      if n == 0:
        cumulative.append(temp)
      else:
        cumulative.append(temp+cumulative[n-1])
    xpos = range(len(cumulative))
    ax_b.plot(xpos, cumulative, marker='o', ms=6, ls='--', color='k')
    ax_b.set_xticks(xpos, [r"10$^0$-10$^1$", "10$^1$-10$^2$", "10$^2$-10$^3$",\
                          "10$^3$-10$^4$", "10$^4$-10$^5$", "10$^5$-10$^6$", "10$^6$-10$^7$"],\
                    rotation=45, ha='center')
    ax_b.set_xlabel(f'Number of {animal} on farm')
    ax_b.set_ylabel(f'Cumulative number of {animal}')


def join_farms_and_buildings(farms, buildings, dist):

  buildings['buildings_geom'] = buildings['geometry']

  farms = farms.to_crs("EPSG:32643")
  farms['geometry'] = farms['geometry'].buffer(dist)
  farms = farms.to_crs("EPSG:4326")

  joined = farms.sjoin(buildings, how='inner', predicate='intersects')
  joined['index_right'] = joined['index_right'].astype(int)

  return joined


def calc_length_etc(row):

    x, y = row.exterior.coords.xy
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])),\
                Point(x[1], y[1]).distance(Point(x[2], y[2])))
    length = max(edge_length)
    width = min(edge_length)
    aspect_ratio = length / width
    return pd.Series({'Length':length, 'Aspect Ratio':aspect_ratio})


def get_dimensions(gdf, crs):

  temp = gdf['buildings_geom'].minimum_rotated_rectangle().to_crs(crs)
  dimensions = temp.apply(calc_length_etc).reset_index(drop=False)
  new = gdf.copy().reset_index(drop=False)#.drop(columns=["index_right"])
  new[['Length', 'Aspect Ratio']] = dimensions[['Length', 'Aspect Ratio']]

  return new


