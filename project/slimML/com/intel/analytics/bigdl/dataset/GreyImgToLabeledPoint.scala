package com.intel.analytics.bigdl.dataset.image

import com.intel.analytics.bigdl.dataset.{Sample, Transformer}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}

import scala.collection.Iterator

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

/**
  * transform labeled grey image to LabeledPoint
  */
class GreyImgToLabeledPoint() extends Transformer[LabeledGreyImage,LabeledPoint] {
  override def apply(prev: Iterator[LabeledGreyImage]): Iterator[LabeledPoint] = {
    prev.map(img => {
      val label = img.label().toDouble
      val features = img.content.map(_.toDouble)

      LabeledPoint(label,Vectors.dense(features))
    })
  }
}

object GreyImgToLabeledPoint{
  def apply(): GreyImgToLabeledPoint = new GreyImgToLabeledPoint()
}