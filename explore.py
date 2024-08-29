"""
Contains code used for exploring potential training datasets and outputting
consistently-formatted files for use by other notebooks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
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


def farms_per_region(df, ax, num, location_kw):
  """
  Create a histogram showing the number of farms per <location_kw> (e.g.
  Comuna, Municipio)
  """

  df = df.groupby([location_kw]).size().sort_values(ascending=False).reset_index(name="Num")
  ax.bar(x=df[location_kw], height=df['Num'], color='w', edgecolor='0.5',
                    width=1)
  for n, comuna in enumerate(df[location_kw][:10]):
    ax.text(0.98, 0.96-0.05*n, comuna, ha='right', va='center',\
            fontsize=8, transform=ax.transAxes)
  ax.axvline(num, color='blue', ls='--')
  ax.set_xticks([])
  ax.set_xlabel(f'{location_kw}')
  ax.set_ylabel(f'Number of farms')

  total = df['Num'].sum()
  in_top_n = df['Num'].head(num).sum()
  percent = in_top_n / total * 100
  print(f'The top {num} {location_kw} account for {in_top_n}/{total} farms ({percent:.0f}%)')


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
  # separate column
  farm_buildings.loc[:, 'Parent coords'] = farm_buildings.loc[:, 'geometry']
  farm_buildings.loc[:, 'geometry'] = farm_buildings.loc[:, 'buildings_geom']
  # Keep any of these cols, if present; drop a load of other columns which
  # just make it hard to inspect the df
  cols_to_keep = ['geometry', 'Parent coords', 'Area (sq m)', 'Farm type',\
                  'Number of animals', 'Number of pigs', 'Number of poultry',\
                  'CAFO class', 'Animal units (pigs)', 'Animal units (poultry)',\
                  'Animal units (unspecified/other)', 'CRS', 'countryName',\
                  'Description']
  farm_buildings = farm_buildings.filter(cols_to_keep).reset_index(drop=True)

  # Find the buildings more than <not_farm_dist> from farm coords

  buffered = buffer_gdf(farms, not_farm_dist)
  # Left join on buildings with buffered farms
  joined = buildings.sjoin(buffered, how="left", predicate='intersects')
  # ID rows with no matches, i.e. buildings outside farm buffer
  # Don't bother removing duplicates, isn't worth it
  not_farm_buildings = joined[joined['index_right'].isnull()]
  # Tidy the gdf. For these buildings there is no farm/lagoon info to preserve
  not_farm_buildings = not_farm_buildings.filter(['geometry', 'Area (sq m)'])

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


def show_n_save(df, bins, fig_name=None, df_name=None, title=''):
  """
  Plots histograms for a single dataset. Intended for use with the DMV and NC
  building datasets
  """

  _, axes = plt.subplots(1, 3, figsize=(9, 3.5))

  cols = ["Area (sq m)", 'Length (m)', 'Aspect ratio']
  stats = {}
  for ax, col, in zip(axes, cols):
    ax.hist(df[col], bins=bins[col], color='k', histtype='step')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    stats[col] = (df[col].min(), df[col].median(), df[col].max())

  plt.suptitle(title, fontsize=10)
  plt.tight_layout()
  if fig_name is not None:
    plt.savefig(f'/content/drive/MyDrive/CAFO_data/Analysis/{fig_name}.png')

  stats_df = pd.DataFrame(stats, index=["Min", "Med", "Max"])
  display(stats_df)
  if df_name is not None:
    stats_df.to_pickle(f'/content/drive/MyDrive/CAFO_data/Analysis/{df_name}.pkl')


def building_scatterplots(df, dist, animal_type, compare, fname, axes):

  def axis_stuff(ax):
    ax.set_xlim(100, 2e6)
    ax.set_xscale('log')
    ax.set_xlabel(f'Reported number of {animal_type.lower()}')

  def plot_compare(animal_type, item, ax):
    # Indicate DMV poultry or NC pigs stats
    if animal_type == 'Poultry' and compare is not None:
      ax.axhline(compare[item].median(), ls="--", lw=1, color='b')
      ax.axhline(compare[item].min(), ls=":", lw=1, color='b')
      ax.axhline(compare[item].max(), ls=":", lw=1, color='b')
    elif animal_type == "Pig" and compare is not None:
      ax.axhline(compare[item].median(), ls="--", lw=1, color='b')

  farms = df.groupby('Parent coords')

  # Max building size vs quantity of animals
  for farm in farms:
    n_animals = farm[1]['Number of animals'].unique()[0]
    med_area = farm[1]['Area (sq m)']
    axes[0].scatter(n_animals, med_area, marker='o', s=8, color='k')
  plot_compare(animal_type, 'Area (sq m)', axes[0])
  axis_stuff(axes[0])
  axes[0].set_ylim(0, 7000)
  axes[0].set_ylabel(f'Max area (sq m) of buildings within {dist} m')

  # Max building length vs quantity of animals
  for farm in farms:
    n_animals = farm[1]['Number of animals'].unique()[0]
    med_length = farm[1]['Length (m)'].max()
    axes[1].scatter(n_animals, med_length, marker='o', s=8, color='k')
  plot_compare(animal_type, 'Length (m)', axes[1])
  axis_stuff(axes[1])
  axes[1].set_ylim(0, 275)
  axes[1].set_ylabel(f'Max length (m) of buildings within {dist} m')

  # Max building aspect ratio vs quantity of animals
  for farm in farms:
    n_animals = farm[1]['Number of animals'].unique()[0]
    med_aspect = farm[1]['Aspect ratio'].max()
    axes[2].scatter(n_animals, med_aspect, marker='o', s=8, color='k')
  plot_compare(animal_type, 'Aspect ratio', axes[2])
  axis_stuff(axes[2])
  axes[2].set_ylim(0, 18)
  axes[2].set_ylabel(f'Max aspect ratio of buildings within {dist} m')

  plt.tight_layout()
  plt.savefig(f'/content/drive/MyDrive/CAFO_data/Analysis/{fname}_buildings.png')


def stratified_sample(gdf_farm, gdf_nonfarm, property_name, bins):
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
      # buildings in this bin, add to overall non-farm gdf
      if len(bin_nonfarm) > 0:
          sampled_bin_nonfarm = bin_nonfarm.sample(farm_hist[i],\
                                                   random_state=42,\
                                                   replace=True)
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



def loop_over_buildings(to_check):
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

  rejects = []
  for n in range(len(to_check)):
    feature = gpd.GeoDataFrame(to_check.iloc[n]).T.set_geometry('geometry')\
                                .set_crs("EPSG:4326")

    print(f"Working on feature {n+1} of {len(to_check)}")
    display(feature)
    fc = geemap.geopandas_to_ee(feature[['geometry']])

    Map = geemap.Map()
    Map.centerObject(fc.first().geometry(), 17)
    Map.add_basemap("HYBRID")
    Map.addLayer(fc.style(**viz), {}, "Building")
    display(Map)

    response = input("Enter reject to reject, exit to exit, or any key to continue  ")
    if response == 'reject':
      rejects.append(feature.index[0])
    if response == 'exit':
      break

  return rejects


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
