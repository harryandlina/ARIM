package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

private[bigdl] class IDAryMiniBatch[T: ClassTag](var idArySamples: Array[(Long,Array[Sample[T]])])
(implicit ev: TensorNumeric[T]) extends MiniBatch[T] {
  require(idArySamples.length > 0,"Input data in idArySamples is empty")
  protected var batchSize = idArySamples.length
  protected var arySamples = idArySamples.map(p => {if(p._2.length > 1) p._2.tail else p._2}).flatMap(ary => ary)
  protected var headIdSamples = idArySamples.map(p => (p._1,p._2.head))
  protected var totalBatchSize = arySamples.length

  protected var miniBatch = SampleToMiniBatch[T](totalBatchSize,partitionNum = Some(1))
    .apply(arySamples.toIterator)
    .next() // get miniBatch

  protected var input: Activity = miniBatch.getInput()
  protected var target: Activity = miniBatch.getTarget()

  override def getInput(): Activity = input

  override def getTarget(): Activity = target

  override def size(): Int = batchSize

  override def slice(offset: Int, length: Int): MiniBatch[T] = {
    require(offset >0 && offset <= batchSize,s"offset should large than 0, found ${offset}")
    new IDAryMiniBatch[T](idArySamples.slice(offset - 1,offset - 1 + length))
  }

  override def set(samples: Seq[Sample[T]])(implicit ev: TensorNumeric[T]): this.type = {
    require(samples.length > 0,"sample is empty")
    require(batchSize == 0 || samples.length <= batchSize,"set Value: samples's size doesn't match mini batch size," +
      s"expected ${size()} got ${samples.length}")
    val idedSamples = samples.zipWithIndex.map(p => (p._2.toLong,Array(p._1)))
    this
  }

  /**
    * update content of this IDAryMiniBatch
    * @param nidArySamples
    * @return
    */
  def set(nidArySamples: Seq[(Long,Array[Sample[T]])]): this.type = {
    require(nidArySamples.length > 0,"nidArySamples is empty")
    require(batchSize == 0 || nidArySamples.length <= batchSize,"set Value: nidArySamples's size doesnt' mathc mini batch size," +
      s"expected ${size()} get ${nidArySamples.length}")
    idArySamples = nidArySamples.toArray
    batchSize = idArySamples.length
    arySamples = idArySamples.map(p => {if(p._2.length > 1) p._2.tail else p._2}).flatMap(ary => ary)
    totalBatchSize  = arySamples.length
    headIdSamples = idArySamples.map(p => (p._1,p._2.head))
    miniBatch = SampleToMiniBatch[T](totalBatchSize,partitionNum = Some(1))
        .apply(arySamples.toIterator).next()
    input = miniBatch.getInput()
    target = miniBatch.getTarget()
    this
  }

  /**
    * get the first samples of each Array[Sample[T]]
    * @return
    */
  def getHeadSamples() = {
    headIdSamples
  }

  /**
    * filter samples by long id
    * @param idAry
    * @return
    */
  def selectById(idAry: Array[Long]) = {
    idArySamples.filter(s => idAry.contains(s._1))
  }
}

object IDAryMiniBatch{
  def apply[T: ClassTag](idArySamples: Array[(Long, Array[Sample[T]])]
   )(implicit ev: TensorNumeric[T]): IDAryMiniBatch[T] = new IDAryMiniBatch[T](idArySamples)
}