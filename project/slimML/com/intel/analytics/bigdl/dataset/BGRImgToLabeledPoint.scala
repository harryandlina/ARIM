package com.intel.analytics.bigdl.dataset.image

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}

import scala.collection.Iterator

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

object BGRImgToLabeledPoint{
  def apply(toRGB: Boolean = true): BGRImgToLabeledPoint = new BGRImgToLabeledPoint(toRGB)
}

/**
  * transform labeled bgr image to labeledpoint
  * @param toRGB
  */
class BGRImgToLabeledPoint(toRGB: Boolean = true) extends Transformer[LabeledBGRImage,LabeledPoint] {
  override def apply(prev: Iterator[LabeledBGRImage]): Iterator[LabeledPoint] = {
    prev.map(img => {
      val label = img.label().toDouble
      val features = new Array[Float](img.width() * img.height() * 3)
      img.copyTo(features,0,toRGB)
      LabeledPoint(label,Vectors.dense(features.map(_.toDouble)))
    })
  }
}
