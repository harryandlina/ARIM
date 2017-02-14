package AccurateML.kmeans

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by zhangfan on 17/1/11.
  */
object TestKmeans {
  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test nonlinear")
    val sc = new SparkContext(conf)
    // Load and parse the data
//    val data = sc.textFile("data/mllib/kmeans_data.txt")
//    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

    val numClusters = args(0).toInt
    val numIterations = args(1).toInt
    val dataPath =args(2)
    val minPartN = args(3).toInt
    val parsedData = MLUtils.loadLibSVMFile(sc, dataPath, 102660, minPartN).map(point => point.features)
    // Cluster the data into two classes using KMeans
//    val numClusters = 2
//    val numIterations = 2
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    // Evaluate clustering by computing Within Set Sum of Squared Errors
    val WSSSE = clusters.computeCost(parsedData)
    println("numClusters,"+numClusters+",itN,"+numIterations+",Within Set Sum of Squared Errors = " + WSSSE)

  }

}
