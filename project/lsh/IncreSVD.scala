package AccurateML.lsh

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangfan on 16/12/8.
  */
class IncreSVD(
                val rating: ArrayBuffer[LabeledPoint],
                val indexs: Array[Int],
                val nf: Int,
                val round: Int,
                val ratioN: Int, //10 min is 1
                val initValue: Double = 0.1,
                val lrate: Double = 0.001, //0.001
                val k: Double = 0.015 //0.015
              ) extends Serializable {
  val n = indexs.size
  val m = rating.last.features.size
  val movieFeas = new DenseMatrix[Double](nf, m, Array.fill(nf * m)(initValue))
  val userFeas = new DenseMatrix[Double](nf, n, Array.fill(nf * n)(initValue))
  val zfInf = 30 //3

  def zfmse(a: DenseMatrix[Double], b: DenseMatrix[Double]): Double = {
    val diff: DenseMatrix[Double] = a - b
    var mse = 0.0
    for (i <- 0 until diff.rows) {
      for (j <- 0 until diff.cols) {
        mse += diff(i, j) * diff(i, j)
      }
    }
    mse = sqrt(mse / (diff.rows.toDouble * diff.cols.toDouble))
    mse

  }

  def calcFeaaturesSparse(hashIt: Int): Unit = {

    for (f <- 0 until nf) {
      for (r <- 0 until round) {
        for (ku <- 0 until n) {
          var cnt = 0
          rating(indexs(ku)).features.foreachActive((i, value) => {
            if (cnt % ratioN == 0) {
              val km = i
              val p = predictRating(km, ku)

              val err = rating(indexs(ku)).features.apply(km) - p
              val cf = userFeas(f, ku)
              val mf = movieFeas(f, km)

              userFeas(f, ku) += lrate * (err * mf - k * cf)
              if (userFeas(f, ku).equals(Double.NaN) || userFeas(f, ku).equals(Double.PositiveInfinity) || userFeas(f, ku).equals(Double.NegativeInfinity)) {
                System.err.println("Double.NaN")
              }
              movieFeas(f, km) += lrate * (err * cf - k * mf)
            }
            cnt += 1
          })
        }
      }

    }
  }

  def calcFeaatures(hashIt: Int, isSparse: Boolean = false): Unit = {
    if(isSparse){
      calcFeaaturesSparse(hashIt)
    }else{
      val sliceM: Int = math.round((m / ratioN).toFloat)
      //    val itIndex = sliceM*hashIt
      for (f <- 0 until nf) {
        for (r <- 0 until round) {
          for (ku <- 0 until n) {
            var kmN = 0
            while (kmN < sliceM) {
              val km = (f + kmN * sliceM) % m
              //            val km = (itIndex+kmN)%m
              val p = predictRating(km, ku)

              val err = rating(indexs(ku)).features.apply(km) - p
              val cf = userFeas(f, ku)
              val mf = movieFeas(f, km)

              userFeas(f, ku) += lrate * (err * mf - k * cf)
              if (userFeas(f, ku).equals(Double.NaN) || userFeas(f, ku).equals(Double.PositiveInfinity) || userFeas(f, ku).equals(Double.NegativeInfinity)) {
                System.err.println("Double.NaN")
              }
              movieFeas(f, km) += lrate * (err * cf - k * mf)

              kmN += 1
            }
          }
        }
      }
    }


  }

  def zfScaler(a: Double): Double = {
    var sum = a
    if (math.abs(sum) > zfInf) {
      //      println("zfInf",+sum)
      sum = if (sum > 0) zfInf else -zfInf
    }
    sum

  }

  def predictRating(mid: Int, uid: Int): Double = {
    var p = 0.0
    for (f <- 0 until nf) {
      p += userFeas(f, uid) * movieFeas(f, mid)
    }
    zfScaler(p)
  }

  def predictRating(mid: Int, uid: Int, fi: Int, acache: Double, bTrailing: Boolean): Double = {
    //    var sum: Double = if (acache > 0) acache else 1.0
    var sum = acache
    sum += movieFeas(fi, mid) * userFeas(fi, uid)
    sum = zfScaler(sum)
    if (bTrailing) {
      sum += (nf - fi - 1) * (initValue * initValue)
      sum = zfScaler(sum)
    }
    sum
  }

}

object IncreSVD {
  def main(args: Array[String]) {

    //    Logger.getLogger("org").setLevel(Level.WARN)
    //    Logger.getLogger("akka").setLevel(Level.WARN)
    //    val conf = new SparkConf().setAppName("IncreSVDvoglis")
    //    val sc = new SparkContext(conf)
    //
    //    val bitN = args(0).toInt
    //    val itN = args(1).toInt
    //    val ratioN = args(2).toInt
    //    val partN = args(3).toInt
    //    val dataSrc = args(4)
    //
    //    val data = sc.textFile(dataSrc).map(line => {
    //      val vs = line.split(",").map(_.toDouble)
    //      val features = vs.slice(0, vs.size - 1)
    //      LabeledPoint(vs.last, Vectors.dense(features))
    //    })
    //    // use data to build "doto", acutally I build "doto" in each partition(data.mapPartition)
    //    val dataArr = new ArrayBuffer[Double]()
    //    for (inc <- data.collect()) {
    //      dataArr ++= inc.features.toArray
    //    }
    //    val numInc = data.count().toInt //=2921
    //    val numFeatures = data.first().features.size //5232
    //    val doto = new DenseMatrix[Double](numInc, numFeatures, dataArr.toArray, 0, numFeatures, true)
    //    //    val doto = DenseMatrix.rand[Double](2000, 5000)
    //
    //
    //    //normalize
    //    val times = new ArrayBuffer[String]()
    //    var time = System.currentTimeMillis()
    //    if (true) {
    //      //doto = normalize(doto(*, ::), 1)
    //      var rr: Int = doto.rows
    //      for (i <- 0 until doto.cols) {
    //        var ma = max(doto(::, i))
    //        var mi = min(doto(::, i))
    //        var maVec = DenseVector.zeros[Double](rr)
    //        var miVec = DenseVector.zeros[Double](rr)
    //        maVec(0 until rr) := ma
    //        miVec(0 until rr) := mi
    //
    //
    //        doto(::, i) := (maVec - doto(::, i)) / (maVec - miVec)
    //      }
    //    }
    //    times += ",normT," + (System.currentTimeMillis() - time)
    //
    //
    //
    //    time = System.currentTimeMillis()
    //    val zfsvd = new IncreSVD(doto, bitN, itN,ratioN) //data bit round
    //    zfsvd.calcFeaatures(0)
    //    val v = zfsvd.userFeas.t
    //    times += ",increSVDT," + (System.currentTimeMillis() - time)
    //
    //    println(times.toArray.mkString("\t"))
    //
    //    val writer = new java.io.PrintWriter(new java.io.File("vIncreSVD." + bitN + "." + itN))
    //    for (i <- 0 until v.rows) {
    //      val a = new Array[Double](v.cols)
    //      for (j <- 0 until a.size) {
    //        a(j) = v(i, j)
    //      }
    //      writer.write(a.mkString(",") + "\n")
    //    }
    //    writer.close()
    //    println("v(" + v.rows + "," + v.cols + "),write done!")
    //
  }
}
