package AccurateML.lsh


import java.io._

import breeze.linalg.svd.SVD
import breeze.linalg.{DenseMatrix, _}
import breeze.numerics._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangfan on 16/12/12.
  */
class IncreSVD1(
                 val rating: DenseMatrix[Double],
                 val nf: Int,
                 val round: Int,
                 val initValue: Double = 1.0,
                 val lrate: Double = 1.0,
                 val const_init: Boolean = true,
                 val k: Double = 0.0) extends Serializable {
  val n = rating.rows
  val m = rating.cols
  val cache = DenseMatrix.zeros[Double](n, m)
  var movieFeas = DenseMatrix.rand[Double](nf, m)
  //new DenseMatrix[Double](nf, m, Array.fill(nf * m)(initValue))
  var userFeas = DenseMatrix.rand[Double](nf, n)
  //new DenseMatrix[Double](nf, n, Array.fill(nf * n)(initValue))
  var cacheMovieFeas: DenseMatrix[Double] = movieFeas.copy
  var cacheUserFeas: DenseMatrix[Double] = userFeas.copy

  var pcav: DenseMatrix[Double] = DenseMatrix.zeros[Double](nf, m)
  //  var pcav:DenseMatrix[Double] = DenseMatrix.zeros[Double](nf, m)

  def setPCAV(pv: DenseMatrix[Double]): Unit = {
    pcav = pv
    println("set pcav size, " + pcav.rows + "," + pcav.cols)

  }

  if (const_init == true) {
    movieFeas = new DenseMatrix[Double](nf, m, Array.fill(nf * m)(initValue))
    userFeas = new DenseMatrix[Double](nf, n, Array.fill(nf * n)(initValue))
  }


  var rate = lrate

  def zfmse(a: DenseMatrix[Double], b: DenseMatrix[Double]): Double = {
    if (a.rows != b.rows || a.cols != b.cols) {
      System.err.println("mismatch error, " + a.rows + "," + a.cols + ",\t" + b.rows + "," + b.cols)
      1 / 0.0
    }
    val diff: DenseMatrix[Double] = a - b
    var mse: Double = 0.0
    for (i <- 0 until diff.rows) {
      for (j <- 0 until diff.cols) {
        mse += diff(i, j) * diff(i, j)
      }
    }
    // Calculate the root mean error per element. This rules out
    //     the dimensionality of the matrices
    mse = sqrt(mse / (diff.rows.toDouble * diff.cols.toDouble))
    mse
  }

  def calcFeatures(): Unit = {
    val dot: DenseMatrix[Double] = userFeas.t * movieFeas

    println("Init MSE," + zfmse(rating, dot))

    for (i <- 0 until nf) {

      val rmse = new ArrayBuffer[Double]()
      rate = lrate
      for (r <- 0 until round) {
        for (ku <- 0 until n) {
          for (km <- 0 until m) {
            val p = predictRating(km, ku)

            val err = rating(ku, km) - p
            val cf = userFeas(i, ku)
            val mf = movieFeas(i, km)


            userFeas(i, ku) += rate * (err * mf - k * cf)
            movieFeas(i, km) += rate * (err * cf - k * mf)
            if (userFeas(i, ku).equals(Double.NaN) || userFeas(i, ku).equals(Double.PositiveInfinity) || userFeas(i, ku).equals(Double.NegativeInfinity)) {
              System.err.println("Double.NaN")
              return
            }

          }
        }
        println("rate = " + rate)
        rate = 0.9 * rate

        rmse += zfmse(rating, userFeas.t * movieFeas)
      }

      val dot: DenseMatrix[Double] = userFeas.t * movieFeas
      println("Feature " + i + " MSE,\t" + zfmse(rating, dot) + " ,each iter mse [, " + rmse.mkString(","))


    }
  }


  def calcFeaturesBacktrack(): Unit = {
    val dot: DenseMatrix[Double] = userFeas.t * movieFeas
    var rmse = zfmse(rating, dot)
    println("MSE(m vs increSVD), " + zfmse(rating, dot) + ", MSE(svd VS increSVD), " + zfmse(dot, pcav))


    for (i <- 0 until nf) {

      rate = lrate
      for (r <- 0 until round) {

        var backtrack: Boolean = true
        var backtrack_count: Int = 0
        while (backtrack && backtrack_count < 20) {

          for (ku <- 0 until n) {
            for (km <- 0 until m) {
              val p = predictRatingBacktrack(km, ku)

              val err = rating(ku, km) - p
              val cf = cacheUserFeas(i, ku)
              val mf = cacheMovieFeas(i, km)


              cacheUserFeas(i, ku) += rate * (err * mf - k * cf)
              cacheMovieFeas(i, km) += rate * (err * cf - k * mf)

            } // m
          } // n
          //println("rate = " + rate)
          val dot: DenseMatrix[Double] = cacheUserFeas.t * cacheMovieFeas
          var rmse_new = zfmse(rating, dot)
          // println("i = " + i + " r = " + r + " rmse = "  + rmse_new)

          if (rmse_new.equals(Double.NaN) || rmse_new > rmse) {
            rate = 0.5 * rate
            backtrack_count = backtrack_count + 1
            println("F" + i + ",r" + r + ",new MSE(m vs increSVD)," + rmse_new + " ,rate, " + rate + ",backN, " + backtrack_count)
            //
            // Please check that this copy is the fasted way to assign matrix to matrix.
            // A simple loop would do....
            cacheMovieFeas = movieFeas.copy
            cacheUserFeas = userFeas.copy
          } else {
            backtrack = false
            movieFeas = cacheMovieFeas.copy
            userFeas = cacheUserFeas.copy
            rmse = rmse_new

          }
        } // Backtrack
      } // r


      //println("Feature "+i+" MSE,\t" + zfmse(rating, dot)+" ,each iter mse [, "+rmse.map(_.toDouble).mkString(","))

      println("F " + i + ", MSE(m vs increSVD), " + rmse + ", MSE(svd VS increSVD), " + zfmse(pcav, userFeas.t * movieFeas))
      println()
    }
  }

  def predictRating(mid: Int, uid: Int): Double = {
    var p = 0.0
    for (f <- 0 until nf) {
      p += userFeas(f, uid) * movieFeas(f, mid)
    }
    p
  }

  def predictRatingBacktrack(mid: Int, uid: Int): Double = {
    var p = 0.0
    for (f <- 0 until nf) {
      p += cacheUserFeas(f, uid) * cacheMovieFeas(f, mid)
    }
    p
  }

}

object IncreSVD1 {

  def getListOfFiles(dir: String): List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(_.isFile).toList
    } else {
      List[File]()
    }
  }

  def main(args: Array[String]) {

    /**
      * Massive data set!!! Read all files and concat into one doto DenseMatrix
      */
    //    var files = getListOfFiles("/Users/zhangfan/Downloads/ghg_data.zf.scaler/part-00000")
    //    var N:Int = 0
    //    var M:Int = 0
    //    var doto:DenseMatrix[Double] = DenseMatrix.zeros[Double](2098, 5232)
    //    for (file <- files){
    //      println(file.getPath())
    //      var part:DenseMatrix[Double]  = csvread(file, ' ' )
    //      var row:DenseVector[Double] = part.reshape(1, 16*part.cols).toDenseVector
    //      doto(N, ::) := row.t
    //      N = N + 1
    //      M = max(M, 16*part.cols)
    //    }
    //
    //    println("N = "  + N + " M = " + M)
    //    println(doto.rows + " " + doto.cols)
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val conf = new SparkConf().setAppName("IncreSVDvoglis")
    val sc = new SparkContext(conf)

    val bitN = args(0).toInt
    val itN = args(1).toInt
    val partN = args(2).toInt
    val dataSrc = args(3) //"/Users/zhangfan/Downloads/ghg_data.zf.scaler"
    val data = sc.textFile(dataSrc).map(line => {
        val vs = line.split(",").map(_.toDouble)
        val features = vs.slice(0, vs.size - 1)
        LabeledPoint(vs.last, Vectors.dense(features))
      })
    // use data to build "doto", acutally I build "doto" in each partition(data.mapPartition)
    val dataArr = new ArrayBuffer[Double]()
    for (inc <- data.collect()) {
      dataArr ++= inc.features.toArray
    }
    val numInc = data.count().toInt //=2921
    val numFeatures = data.first().features.size //5232
    println("dataRows,dataCols,(" + numInc + "," + numFeatures)
    val doto = new DenseMatrix[Double](numInc, numFeatures, dataArr.toArray, 0, numFeatures, true)


    /**
      * Create a random matrix
      */
    //var doto =  DenseMatrix.rand[Double](100, 1000) :* 1.0

    /**
      * Read a small sized real problem (CASP)
      */
    // var doto = csvread(new File("C:/eclipseKepler/workspace/StochasticGN/CASP.csv"), ',' )
    // doto = doto(0 until 1000, ::)

    /**
      * Set these variables K  is the number of singular values
      * donormalize: is set to true if we want columnwise normalization
      * performBreezeSVD: is set to true if we want to compare to Breeze SVD function.
      * used only for small datasets.
      * round: how many rounds of optimization should be performed for each singular value
      * lrate: Initial learning rate. This will be reduced in case of divergence
      * const_init: If set true it initializes the matrices to the user provided value  initValue
      * initValue: If const_init = true we initialize the matrices to this value
      */
    var K: Int = bitN//6
    /**
      * because ghg_data.zf.scaler has been scalered by
      * "val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(data.map(x => x.features))",
      * which means ghg_data.zf.scaler has 0 mean and 1 stv. so I think it's not reasonable to re-normalize the data to [0,1],
      * so I think should set " donormalize=false"
      */
    var donormalize: Boolean = true
    var performBreezeSVD: Boolean = true
    var round: Integer = itN //50
    var lrate: Double = 1.0
    var const_init: Boolean = true
    var initValue: Double = 1.0
    println("donormalize," + donormalize + ",round," + round + ",lrate," + lrate + ",const_init," + const_init + ",initValue," + initValue)



    if (donormalize == true) {
      //doto = normalize(doto(*, ::), 1)
      var rr: Int = doto.rows
      for (i <- 0 until doto.cols) {
        var ma = max(doto(::, i))
        var mi = min(doto(::, i))
        var maVec = DenseVector.zeros[Double](rr)
        var miVec = DenseVector.zeros[Double](rr)
        maVec(0 until rr) := ma
        miVec(0 until rr) := mi


        doto(::, i) := (maVec - doto(::, i)) / (maVec - miVec)
      }
    }



    println(doto.rows + "," + doto.cols)
    val time = System.currentTimeMillis()
    val zfsvd = new IncreSVD1(doto, K, round, lrate, initValue, const_init)
    zfsvd.setPCAV(DenseMatrix.zeros[Double](doto.rows, doto.cols))
    zfsvd.calcFeaturesBacktrack()
    //    val v = zfsvd.userFeas.t
    println("Incr. svd  err = " + zfsvd.zfmse(doto, zfsvd.userFeas.t * zfsvd.movieFeas) + ",increSVDT," + (System.currentTimeMillis() - time) / 1000 + " secs")


    val v = zfsvd.userFeas.t
    val writer = new java.io.PrintWriter(new java.io.File("increSVD1.v"))
    for (i <- 0 until v.rows) {
      val a = new Array[Double](v.cols)
      for (j <- 0 until a.size) {
        a(j) = v(i, j)
      }
      writer.write(a.mkString(",") + "\n")
    }
    writer.close()
    println("write done")
    if (performBreezeSVD == true) {
      val time = System.currentTimeMillis()
      val SVD(u, s, vt) = svd(doto)
      val ss = DenseMatrix.zeros[Double](K, K)
      diag(ss(0 until K, 0 until K)) := s(0 until K)
      val reM: DenseMatrix[Double] = u(::, 0 until K) * ss * vt(0 until K, ::)
      println("svd err = " + zfsvd.zfmse(doto, reM) + ",svdTime," + (System.currentTimeMillis() - time) / 1000 + " secs")
      println("svd VS increSVD=, " + zfsvd.zfmse(reM, zfsvd.userFeas.t * zfsvd.movieFeas))
    }


    //    println(zfsvd.userFeas(::, 0))

  }

}
