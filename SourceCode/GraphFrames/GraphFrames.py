import os

import pyspark.sql.functions as f
from graphframes import *
from pyspark.sql import SparkSession


os.environ["SPARK_HOME"] = "C:\Installations\spark-2.4.5-bin-hadoop2.7"
os.environ["HADOOP_HOME"] = "C:\Installations\Hadoop"
os.environ["PYSPARK_SUBMIT_ARGS"]  = ("--packages  graphframes:graphframes:0.5.0-spark2.0-s_2.11 pyspark-shell")

spark = SparkSession \
    .builder \
    .appName("graphFrame") \
    .getOrCreate()

#1
#Reading two datasets directly into dataFrames and creating a graphframe
station_df = spark.read.csv(r"201508_station_data.csv", header=True)
trip_df = spark.read.csv(r"201508_trip_data.csv", header=True)
station_df.show()
trip_df.show()

station_vertices = station_df.withColumnRenamed("name", "id").select("id").distinct()
station_vertices.show()

trip_edges = trip_df.withColumnRenamed("Start Station", "src").withColumnRenamed("End Station", "dst").select("src", "dst")
trip_edges.show()

g = GraphFrame(station_vertices, trip_edges)

#2 traingle count

g.triangleCount().show()

#3 shortest path

g.shortestPaths(landmarks=["MLK Library", "Townsend at 7th"]).show()

#4 page rank

results = g.pageRank(resetProbability=0.15, tol=0.01)
results.vertices.show()
results.edges.show()

#5 Save vertices and edges as Parquet to some location.
g.vertices.write.parquet("vertices")
g.edges.write.parquet("edges")

#bonus

#1 Label propogation algorithm
g.labelPropagation(maxIter=5).show()

#2 BFS algorithm
g.bfs("id='2nd at Folsom'", "id='Market at 10th'").show()