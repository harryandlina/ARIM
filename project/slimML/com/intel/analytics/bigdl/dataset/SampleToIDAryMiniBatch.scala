package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import org.apache.commons.lang3.SerializationUtils
import java.util

import com.intel.analytics.bigdl.utils.{RandomGenerator, T}

import scala.collection.Iterator
import scala.reflect.ClassTag

class SampleToIDAryMiniBatch[T: ClassTag](totalBatch: Int,miniBatch: Option[MiniBatch[T]] = None,
      partitionNum: Option[Int] = None)
      (implicit ev: TensorNumeric[T]) extends Transformer[(Long,Array[Sample[T]]),IDAryMiniBatch[T]]{
  private val batchPerPartition = Utils.getBatchSize(totalBatch,partitionNum)
  var idAryMiniBatchBuffer = miniBatch.orNull
  private val batchSize = batchPerPartition
  private val aryBatch = new Array[(Long,Array[Sample[T]])](batchSize)

  override def apply(prev: Iterator[(Long, Array[Sample[T]])]): Iterator[IDAryMiniBatch[T]] = {
    new Iterator[IDAryMiniBatch[T]] {
      override def hasNext: Boolean = prev.hasNext

      override def next(): IDAryMiniBatch[T] = {
        if(prev.hasNext){
          var i = 0
          while(i < batchSize && prev.hasNext){
            val sample = prev.next()
            aryBatch(i) = sample
            i += 1
          }
          var cur = i
          while(i < batchSize){
            val idx = (RandomGenerator.RNG.uniform(0,1.0) * cur).toInt
            val sample = aryBatch(idx)
            aryBatch(i) = sample
            i += 1
          }
          new IDAryMiniBatch[T](aryBatch)
        }
        else{
          null
        }
      }
    }
  }
}

object SampleToIDAryMiniBatch{
  def apply[T: ClassTag](totalBatch: Int, miniBatch: Option[MiniBatch[T]] = None,
            partitionNum: Option[Int] = None)
           (implicit ev: TensorNumeric[T]): SampleToIDAryMiniBatch[T] = new SampleToIDAryMiniBatch[T](totalBatch, miniBatch,None)
}
