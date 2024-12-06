"""
Contains code used for exploring potential training/test/application datasets
and outputting consistently-formatted files for use by other notebooks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import ee
import geemap.foliumap as geemap

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


def get_buildings(area, min_building_size, country_code):
  """
  Return a feature collection of buildings > min_building_size within the
  specified boundary, along with the region boundary in ee format.
  """

  gdf = gpd.GeoDataFrame(crs="EPSG:4326", geometry=[area["Boundary"]])
  geom = geemap.geopandas_to_ee(gdf[['geometry']])

  buildings_fc = (
      ee.FeatureCollection(f"projects/sat-io/open-datasets/VIDA_COMBINED/{country_code}")
      .filter(ee.Filter.gt('area_in_meters', min_building_size))
  )

  # For some reason, if there are no buildings in the filtered and this command
  # is incorporated in the previous statement, it fails and can't be caught
  # using a try/except block. Seems to work as a stand-alone command though??

  buildings_fc = buildings_fc.filterBounds(geom)

  return buildings_fc, geom


def join_farms_and_buildings(farms, buildings, farm_dist, not_farm_dist, crs):
  """
  Return a df containing the farms in <farms> and the buildings in <buildings>
  that are within <farm_dist> of each farm location. Also, return a df
  containing the buildings that are at least <not_farm_dist> away from any farm
  location.
  """

  def buffer_gdf(farms, dist):
    temp = farms.to_crs(crs)
    temp['geometry'] = temp['geometry'].buffer(dist)
    temp = temp.to_crs("EPSG:4326")

    return temp

  # Find the buildings within <farm_distance> of farm coords

  # Buffer the farm coordinates
  buffered = buffer_gdf(farms, farm_dist)
  # Create a column to preserve building coords
  buildings.loc[:, 'buildings_geom'] = buildings.loc[:, 'geometry']
  # Inner join on buffered farms with buildings
  farm_buildings = buffered.sjoin(buildings, how='inner', predicate='intersects')
  # Make the building geometry the primary geometry, retain farm geom in a
  # separate column, rename the CRS column
  farm_buildings.loc[:, 'Parent coords'] = farm_buildings.loc[:, 'geometry']
  farm_buildings.loc[:, 'geometry'] = farm_buildings.loc[:, 'buildings_geom']
  # This is just for datasets like Chile that need multiple CRSs
  try:
    farm_buildings.loc[:, 'CRS'] = farm_buildings.loc[:, 'CRS_left']
  except KeyError:
    pass
  # Keep any of these cols, if present, because they are useful; drop a load of
  # other columns which just make it hard to inspect the df
  cols_to_keep = ['geometry', 'Parent coords', 'Area (sq m)', 'Farm type',\
                  'Number of animals', 'Number of pigs', 'Number of poultry',\
                  'CAFO class', 'Animal units (pigs)', 'Animal units (poultry)',\
                  'Animal units (unspecified/other)', 'CRS', 'countryName',\
                  'Description']
  farm_buildings = farm_buildings.filter(cols_to_keep).reset_index(drop=True)

  # Find the buildings more than <not_farm_dist> from farm coords

  buildings.to_crs(crs, inplace=True)
  farms.to_crs(crs, inplace=True)
  near_farms = buildings.sjoin_nearest(farms, how='inner',\
                                       max_distance=not_farm_dist)
  not_farm_buildings = buildings[~buildings.index.isin(near_farms.index)].copy()
  not_farm_buildings.to_crs("EPSG:4326", inplace=True)

  # Tidy the gdf. For these buildings there is no farm info to preserve
  not_farm_buildings = not_farm_buildings.filter(['geometry', 'Area (sq m)',\
                                                  'CRS'])

  return farm_buildings, not_farm_buildings


def calc_length_etc(row):

    x, y = row.exterior.coords.xy
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])),\
                Point(x[1], y[1]).distance(Point(x[2], y[2])))
    length = max(edge_length)
    width = min(edge_length)
    aspect_ratio = length / width
    return pd.Series({'Length (m)':length, 'Aspect ratio':aspect_ratio})


def get_dimensions(gdf, crs):

  temp = gdf['geometry'].minimum_rotated_rectangle().to_crs(crs)
  dimensions = temp.apply(calc_length_etc).reset_index(drop=False)
  new = gdf.copy().reset_index(drop=True)
  new[['Length (m)', 'Aspect ratio']] = dimensions[['Length (m)', 'Aspect ratio']]

  return new


def define_bins():
  """
  Define a consistent set of bins for histograms and stratified sampling, 
  to be used across all datasets
  """

  bins = {}
  bins['Area (sq m)'] = np.linspace(200, 6000, 30)
  bins['Length (m)'] = np.linspace(0, 280, 30)
  bins['Aspect ratio'] = np.linspace(0, 20, 30)

  return bins


def stratified_sample(gdf_farm, gdf_nonfarm, property_name, bins,\
                      factor=1):
  """
  Return a sample from <gdf_nonfarm> that matches <gdf_farm> in
  terms of the distribution of <property_name>.
  """

  # Create histograms for the farm buildings
  farm_hist, bin_edges = np.histogram(gdf_farm[property_name], bins=bins)

  # Initialize an empty dataframe for the sampled non-farm buildings
  sampled_nonfarm = gpd.GeoDataFrame()

  # Stratified sampling
  for i in range(len(bin_edges)-1):

      # Select rows with areas that lie in this bin
      bin_mask = (gdf_nonfarm[property_name] >= bin_edges[i]) &\
                  (gdf_nonfarm[property_name] < bin_edges[i+1])
      bin_nonfarm = gdf_nonfarm[bin_mask]

      # Sample the same number of non-farm buildings as there are farm
      # buildings in this bin, multiplied by <factor>; add to overall non-farm
      # gdf
      if len(bin_nonfarm) > 0:
          num = int(farm_hist[i] * factor)
          sampled_bin_nonfarm = bin_nonfarm.sample(num,\
                                                   random_state=42,\
                                                   replace=False)
          sampled_nonfarm = pd.concat([sampled_nonfarm, sampled_bin_nonfarm])

  return sampled_nonfarm


def plot_notfarms(farms, notfarms, bins, dataset_name):
  """
  Make histograms and aspect-area plots for farm and notfarm samples.
  """

  _ , axes = plt.subplots(1, 3, figsize=(10, 3.5))

  for ax, col in zip(axes, ["Area (sq m)", "Length (m)", "Aspect ratio"]):
    ax.hist(farms[col], bins=bins[col], histtype='step', color='k',\
            label="Farm")
    ax.hist(notfarms[col], bins=bins[col], histtype='step', color='b',\
            alpha=0.5, lw=1.5, label="Not-farm")
    if col == "Area (sq m)":
      ax.legend(fontsize=8)
    ax.set_ylabel("Frequency")
    ax.set_xlabel(col)
  plt.tight_layout()
  plt.show()
  plt.savefig("/content/drive/MyDrive/CAFO_data/Analysis/\
  {dataset_name}_notfarm_histos.png")

  def axis_stuff(ax, label):
    ax.set_xlim(150, 5000)
    ax.set_ylim(0.8, 20)
    ax.set_xlabel(f'Area (sq m)')
    ax.set_ylabel(f'Aspect ratio')
    ax.set_title(label, fontsize=10)

  _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
  ax1.plot(farms["Area (sq m)"], farms["Aspect ratio"], 'ko', ms=2)
  axis_stuff(ax1, "Farm buildings")
  ax2.plot(notfarms["Area (sq m)"], notfarms["Aspect ratio"], 'bo', ms=2,\
   alpha=0.3)
  axis_stuff(ax2, "Not-farm buildings")
  plt.tight_layout()
  plt.show()
  plt.savefig("/content/drive/MyDrive/CAFO_data/Analysis/\
  {dataset_name}_notfarm_aspect_area.png")


def loop_over_buildings(to_check, column="geometry", sentinel=None, radius=240):
  """
  # Loops over all the rows in <to_check> and shows each one in turn on a map.
  # Type <reject> to add the row to a list of rows to reject
  """

  os.environ["HYBRID"] = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'

  viz = {
      'color': 'yellow',
      'width': 2,
      'fillColor': '00000000'
  }

  viz2 = {
      'color': 'red',
      'width': 2,
      'fillColor': '00000000'
  }

  sentinel_viz = {
    'min': 0.0,
    'max': 2500,
    'bands': ['B4', 'B3', 'B2'],
  }

  rejects = []
  for n in range(len(to_check)):
    feature = gpd.GeoDataFrame(to_check.iloc[n]).T.set_geometry("geometry")\
                                .set_crs("EPSG:4326")

    if column != "geometry":
      feature = feature.drop(columns=["geometry"]).rename(columns={column:"geometry"}) 
      feature['geometry'] = gpd.GeoSeries.from_wkt(feature['geometry'])
      feature = feature.set_geometry("geometry").set_crs("EPSG:4326")

    print(f"Working on feature {n+1} of {len(to_check)}")
    display(feature)
    fc = geemap.geopandas_to_ee(feature[["geometry"]])

    #get the image extent
    def buffer_and_bound(feature, buffer_radius=radius):
      return feature.centroid().buffer(buffer_radius, 2).bounds()
    img_extent = fc.map(buffer_and_bound)

    Map = geemap.Map()
    Map.centerObject(fc.first().geometry(), 17)
    Map.add_basemap("HYBRID")
    if sentinel is not None:
      print('here')
      Map.addLayer(sentinel, sentinel_viz, "Sentinel")
    Map.addLayer(fc.style(**viz), {}, "Building")
    Map.addLayer(img_extent.style(**viz2), {}, "Image extent")
    display(Map)

    response = input("Enter reject to reject, exit to exit, or any key to continue  ")
    if response == 'reject':
      rejects.append(feature.index[0])
    if response == 'exit':
      break

  return rejects


def merge_and_make_box(area, buildings_fc, within=40):
  """
  For use in defining farms- and not-farms in test and model application regions.
  Combine closely-spaced buildings into a single polygon, and select the largest
  building within that polygon. This is analogous to what we do when selecting
  the largest building per set of farm coordinates when creating the training
  datasets, except here we don't have any actual farm coordinates, so we use
  clusters of buildings instead. Returns the merged area boundaries for
  visualization, and the geometry (polygon) of each one's largest building.
  """

  buildings = geemap.ee_to_gdf(buildings_fc)
  buildings = buildings.to_crs(area["CRS"])
  saved_geoms = buildings["geometry"]

  # create a polygon for each cluster of buildings by buffering and merging
  geoms = buildings.buffer(within).geometry.union_all()
  merged = gpd.GeoDataFrame(geometry=[geoms]).set_crs(area["CRS"])
  merged = merged.explode().reset_index(drop=True)

  # identify all the buildings within each polygon and select the largest
  intersecting = gpd.sjoin(merged, buildings, how="left", predicate="intersects")
  largest = intersecting.sort_values(by=["area_in_meters"], ascending=False)\
                        .groupby(level=0).head(1)

  # give each building its correct geometry (as opposed to the cluster polygon)
  temp = largest.join(saved_geoms, on="index_right", rsuffix='orig')
  largest = temp.drop(columns=['geometry', 'index_right']).\
            rename(columns={'geometryorig': 'geometry'}).\
            set_geometry('geometry').to_crs("EPSG:4326")

  # remove buildings that intersect with boundary edges, as they'd be discarded
  # when the Sentinel images are created, anyway
  largest = largest.sjoin(area["Boundary gdf"], how="inner", predicate="within")
  largest = largest.drop(columns=["index_right"]).reset_index(drop=True)

  print(f"Went from {len(buildings)} buildings to {len(largest)} boxes")

  return merged, largest


def get_sentinel(area, boundary, sentinel_bands, year=2023):
  """
  Get Sentinel data for an area
  """

  gdf = gpd.GeoDataFrame(crs="EPSG:4326", geometry=[area["Boundary"]])
  geom = geemap.geopandas_to_ee(gdf[['geometry']])

  sentinel = (
      ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterDate(f'{year}-01-01', f'{year}-12-31')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
      .select(sentinel_bands)
      .median() #crude cloud filter
      .clip(boundary)
  )

  return sentinel


def re_order(df):
  """
  Return a version of df in which the columns are in a standard order. Also,
  convert the Parent coords column from geometry to wkt, to avoid problems
  with multiple geometry columns in later steps.
  """

  columns = ["geometry", "Area (sq m)", "Length (m)", "Aspect ratio",\
             "Parent coords", "Farm type", "Number of animals", "Dataset name"]
  df = df[columns].reset_index(drop=True)
  if df["Parent coords"].dtype == 'geometry':
    df['Parent coords'] = df['Parent coords'].to_wkt()
  
  return df
