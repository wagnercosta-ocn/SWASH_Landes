import os
import glob
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from scipy.stats import pearsonr

# SWOT_HR files
# List of GeoJSON file paths
geojson_files = glob.glob(os.path.join('/scratch/work/langercostaw/HR_SWOT', '*.geojson'))


# Load all files into a list of GeoDataFrames
gdf_list = [gpd.read_file(file) for file in geojson_files]
combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))# Concatenate all GeoDataFrames
print('SWOT_HR files concatenated!')

# transects lat lon
df2 = pd.read_csv('/scratch/work/langercostaw/swash_landes/transects_points_with_depth_filtered_v2/processed_transects_10.csv')
#df2 = df2.loc[df2.elevation<=-5]
id_tr = df2['transect_id'].unique()
print('transect names found!')
GDF=gpd.GeoDataFrame({})
for id in id_tr:
  lon=df2.loc[df2.transect_id==id].longitude.values
  lat=df2.loc[df2.transect_id==id].latitude.values
  # SWASH free surface results
  folder_results = '/scratch/work/langercostaw/swash_landes/swash_cases/results_per_transect'
  csv_files = glob.glob(os.path.join(folder_results, f'case_2025012603*_{id}.csv'))


  for f in csv_files:
      df=pd.read_csv(f)
      df['longitude']=lon
      df['latitude']=lat
      df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1) # Create geometry column as Point objects

      gdf = gpd.GeoDataFrame(df, geometry='geometry')
      gdf = gdf.set_crs(epsg=4326)

      # convert to metric coordinate. epsg:3857(Web mercator)
      gdf_metric=gdf.to_crs("EPSG:3857")
      combined_gdf_metric = combined_gdf.to_crs("EPSG:3857")

      # calculate the buffer around the transect points
      gdf_metric['buffer'] = gdf_metric.buffer(50)

      # join spatialy transect buffers and SWOT HR data
      joined = gpd.sjoin(combined_gdf_metric,gdf_metric, how='inner', predicate='within')

      # calculate average sea_surface height of the points within the buffer for each transect point
      result_metric = joined.groupby('index_right').agg({'geometry': lambda x: Point(x.x.mean(), x.y.mean())}).reset_index()
      print(result_metric.head())
      result_metric = result_metric.set_crs(epsg=3857)
      # rename column
      result_metric = result_metric.rename(columns={'geometry': 'avg_point_metric'})

      # reproject coordinates

      result = result_metric.copy()
      result['avg_point'] = result['avg_point_metric'].set_crs(epsg=3857)
      result['avg_point'] = result['avg_point_metric'].to_crs("EPSG:4326")
      result = result.drop(columns=['avg_point_metric'])
      print(result[['index_right', 'avg_point']])

      # Find the nearest point in the cloud for each transect point
      #nearest_points = gdf.sjoin_nearest(combined_gdf, how="left", distance_col="distance")


      #GDF=pd.concat([GDF,nearest_points])
      GDF=pd.concat([GDF,result])
  #plt.plot(nearest_points2.setup_noassim,nearest_points2.surface,'.k');plt.show()
GDF2 = GDF.copy()
GDF2=GDF2.iloc[::5]
var1='watlev_noassim'

var2='surface'
#GDF2=GDF2.loc[(GDF2['depth']<-5)&(GDF2['position']>1000)]
GDF2=GDF2.loc[(GDF2['position']>=1000)] #& (GDF2['position']<1000)]
GDF2=GDF2.loc[GDF2[var2]<16]
GDF2=GDF2.loc[abs(GDF2[var2]-GDF2[var1])<3]


fig,ax = plt.subplots()
ax.plot(GDF2[var2],GDF2[var1],'.k')
ax.plot(np.arange(-2,12),np.arange(-2,12),'--k');plt.grid('on')
ax.set_xlabel('SWOT_HR_SSH (m)')
ax.set_ylabel(f'SWASH {var1} (m)')
ax.set_title ('Scatter-plot')

# calculating errors
corr, p_value = pearsonr(GDF2[var1], GDF2[var2])
bias = GDF2[var1].mean()- GDF2[var2].mean()
squared_errors = (GDF2[var1] - GDF2[var2])**2
mse = np.mean(squared_errors)
rmse = np.sqrt(mse)
mae = np.sum(abs(GDF2[var1] - GDF2[var2]))/GDF2.shape[0]

std= np.std(GDF2[var1] - GDF2[var2])
print('Results Scatter')
print(corr)
print(bias)
print(rmse)
print(mae)
print(std)

if p_value<0.05:
   ax.text(0.7,0.4,f"corr = {corr:.2f}",transform=ax.transAxes)
   ax.text(0.7,0.35,f"bias = {bias:.2f} m",transform=ax.transAxes)
   ax.text(0.7,0.3,f"rmse = {rmse:.2f} m", transform=ax.transAxes)
   ax.text(0.7,0.25,f"mae = {mae:.2f} m",transform=ax.transAxes)
   ax.text(0.7,0.2,f"std = {std:.2f} m",transform=ax.transAxes)
else:
   print('no valid results')

fig.show();
fig.savefig(f'scatter_swash_swotHR_{var1}.png')

fig2,ax2 = plt.subplots()
q1= GDF2[var2].quantile(np.arange(0.01,0.99,0.01))
q2= GDF2[var1].quantile(np.arange(0.01,0.99,0.01))

ax2.plot(q1,q2 ,'.b')
ax2.plot(np.arange(-1,7),np.arange(-1,7),'--k');plt.grid('on')
ax2.set_xlabel('SWOT_HR_SSH (m)')
ax2.set_ylabel(f'SWASH {var1} (m)')
ax2.set_title ('QQ-plot (1%-99)')

bias = np.mean(q2)- np.mean(q1)
squared_errors = (q2 - q1)**2
mse = np.mean(squared_errors)
rmse = np.sqrt(mse)
mae = np.sum(abs(q2 - q1))/q1.shape[0]
std= np.std(q2 - q1)

#ax.text(0.7,0.4,f"corr = {corr:.2f}",transform=ax.transAxes)
ax2.text(0.7,0.35,f"bias = {bias:.2f} m",transform=ax.transAxes)
ax2.text(0.7,0.3,f"rmse = {rmse:.2f} m", transform=ax.transAxes)
ax2.text(0.7,0.25,f"mae = {mae:.2f} m",transform=ax.transAxes)
ax2.text(0.7,0.2,f"std = {std:.2f} m",transform=ax.transAxes)

fig2.show();

print('Results QQplot')
print(bias)
print(rmse)
print(mae)
print(std)

fig2.savefig(f'qqplot_swash_swotHR_{var1}.png')
#GDF.to_file(f"output.geojson", driver="GeoJSON")
