package AccurateML.svm.shun.hash

/**
  * ICT
  * Created by douyishun on 12/7/16.
  */

import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression._
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * Train SVM Model by multi-layer SVMs. Each layer contains multiple parallel SMOs.
  * The middle results between layers will filter the training example and generate the
  * initial parameter of next layer.
  *
  * At each layer, the partial results(alpha and bias) from parallel SMOs are combined
  * and used for filtering the point which may cause great impact to the model. And the
  * filtered point is the training example of next layer, significantly reduce training
  * size.
  */
class SVM(objectData: RDD[Array[ArrayBuffer[LabeledPoint]]]) extends java.io.Serializable {

  def train(zfratio: Double,
            maxLayer: Int,
            maxIterations: Int,
            C: Double,
            tol: Double,
            eps: Double,
            kernelFunc: (Vector, Vector) => Double): SVMModel = {

    val indexObject = objectData.zipWithIndex()
    var zipWithIndex = indexObject.map(t => {
      val index = t._2
      val ziparr = t._1(0)
      Tuple2(index, ziparr.last)
    })
    var pointWithZipIndex = indexObject.flatMap(t => {
      val index = t._2
      val pointarr = t._1(1)
      pointarr.map(point => Tuple2(index, point))
    })
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(pointWithZipIndex.map(_._2).map(x => x.features))

    zipWithIndex = zipWithIndex.map(t => {
      val x = t._2
      val scalePoint = LabeledPoint(if (x.label < 0.9) -1 else 1, scaler.transform(Vectors.dense(x.features.toArray)))
      Tuple2(t._1, scalePoint)
    })
    pointWithZipIndex = pointWithZipIndex.map(t => {
      val x = t._2
      val scalePoint = LabeledPoint(if (x.label < 0.9) -1 else 1, scaler.transform(Vectors.dense(x.features.toArray)))
      Tuple2(t._1, scalePoint)
    })
    val pointWithTwoIndex = pointWithZipIndex.zipWithIndex()
    val pointWithIndex = pointWithTwoIndex.map(t => (t._2, t._1._2))

    zipWithIndex.cache()
    pointWithTwoIndex.cache()
    val zipn: Int = zipWithIndex.count().toInt //total number of training data
    val trainn: Int = pointWithZipIndex.count().toInt
    var zipIndexMap = zipWithIndex.collectAsMap()
    val indexedDataMap = pointWithIndex.collectAsMap()

    var topIndexedData: RDD[(Long, LabeledPoint)] = pointWithIndex
    val alpha = new mutable.HashMap[Long, Double]
    val zipalpha = new mutable.HashMap[Long, Double]
    for (i <- 0 until trainn) alpha += (i.toLong -> 0.0) //init alpha with 0
    for (i <- 0 until zipn) zipalpha += (i.toLong -> 0.0) //init alpha with 0
    var intercept: Double = 0
    var zipintercept: Double = 0

    //training begin

    for (i <- 1 to maxLayer) {
      val smo = new SMO(alpha, maxIterations, C, tol, eps, kernelFunc)
      val res = topIndexedData.mapPartitionsWithIndex(smo.train)
      res.persist()

      val zipsmo = new SMO(zipalpha, maxIterations, C, tol, eps, kernelFunc)
      val zipres = zipWithIndex.mapPartitionsWithIndex(zipsmo.train)
      zipres.persist()

      //update alpha and intercept
      val updatedAlpha = res.filter(x => x._1 == 0 || x._1 > 0).collectAsMap()
      for (i <- updatedAlpha.keys) alpha(i) = updatedAlpha(i)
      intercept = res.filter(x => x._1 < 0).map(x => x._2).mean()
      res.unpersist()

      val zipupdatedAlpha = zipres.filter(x => x._1 == 0 || x._1 > 0).collectAsMap()
      for (i <- zipupdatedAlpha.keys) zipalpha(i) = zipupdatedAlpha(i)
      zipintercept = zipres.filter(x => x._1 < 0).map(x => x._2).mean()
      zipres.unpersist()

      //      select most important k%
      var score: Double = 0
      val topIndex = zipWithIndex.map(
        x => {
          score = -1 * zipintercept
          for (j <- zipalpha.keys if zipalpha(j) > 0) {
            score += zipalpha(j) * zipIndexMap(j).label *
              kernelFunc(zipIndexMap(j).features, zipIndexMap(x._1).features)
          }
          (Math.abs(zipIndexMap(x._1).label * score - 1), x._1.toInt)
        }
      ).top((zipn * zfratio).toInt)
//      topIndexedData = pointWithZipIndex.filter(x => topIndex.map(_._2).contains(x._1))
      topIndexedData = pointWithTwoIndex.filter(x => topIndex.map(_._2).contains(x._1._1)).map(t => (t._2, t._1._2))
    }
    //training end

    //collect support vector
    val svIndex: Array[Long] = alpha.filter(x => x._2 > 0).keys.toArray
    val svIndexedData = indexedDataMap.filter(x => svIndex.contains(x._1))
    val svAlpha = alpha.filter(x => svIndex.contains(x._1))

    //build SVMModel with support vector
    new SVMModel(svIndexedData.values.toArray,
      svAlpha.values.toArray, intercept, kernelFunc)
  }

  def train(zfratio: Double,
            maxLayer: Int,
            maxIterations: Int,
            C: Double,
            gamma: Double): SVMModel = {
    val kernel = new Kernel
    train(zfratio, maxLayer, maxIterations, C, 1e-3, 1e-3, kernel.rbf(gamma))
  }
}
