package AccurateML.lsh

import breeze.linalg.DenseMatrix
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import AccurateML.blas.ZFBLAS

/**
  *
  * Created by zhangfan on 16/9/21.
  */
class ZFLSH(
             n: Int,
             m: Int) {
  val normal01 = breeze.stats.distributions.Gaussian(0, 1)
  val nmat = DenseMatrix.rand(m, n, normal01)

  def hashVector(vector: linalg.Vector): String = {

    val r = new Array[Int](n)
    for (i <- 0 until n) {
      val mc = nmat(::, (i))
      val ans = ZFBLAS.dot(vector, Vectors.dense(mc.toArray))
      if (ans > 0)
        r(i) = 1
    }
    r.mkString("")
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("test lsh")
    val sc = new SparkContext(conf)
    val numBits = 4
    val numFeatures = 1000
    val lsh = new ZFLSH(numBits, numFeatures)
    val data: RDD[Vector] = sc.objectFile("") //eg, the first element in data is dfirst=Vector(1.0,2.0,...,1000.0)
    val mapData:RDD[(String,Vector)]=data.map(vec=>(lsh.hashVector(vec),vec))
    //eg,the first element in mapData mdfirst=Tuple2("1010",Vector(1.0,2.0,...,100.0))
    //"1010" is the sketch of dfirst=Vector(1.0,2.0,...,1000.0)
    //the instances with the same sketch will belong to the same cluster

  }

}
