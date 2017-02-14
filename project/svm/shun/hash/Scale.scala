package AccurateML.svm.shun.hash

/**
  * ICT
  * Created by douyishun on 12/9/16.
  */

import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object Scale extends java.io.Serializable{

  /**
    * Centers the data with mean and Scales the data to unit standard deviation.
    */
  def scale(data: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val scaler = new StandardScaler(withMean = true, withStd = true).
      fit(data.map(x => x.features))

    data.map(
      x => LabeledPoint(
        if(x.label < 0.9) -1 else 1,
        scaler.transform(Vectors.dense(x.features.toArray)))
    )
  }
}
