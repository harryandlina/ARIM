package com.intel.analytics.bigdl.optim

import scala.util.Random

object SampleUtils {
  /**
    * sampling with reservoir algorithm
    * @param weights
    * @param size
    * @return
    */
  def reservoirMultinomial(weights: Array[Double],size: Int) = {
    assert(weights.length > 0 && size > 0)
    implicit val ord: Ordering[(Int,Double)] = Ordering.by(_._2)
    val selectedIndex = scala.collection.mutable.PriorityQueue[(Int,Double)]()(ord.reverse)
    val length = weights.length
    val random = new scala.util.Random()
    val eps = 1e-9
    for(i <- 0 until length){
      val elem = weights(i)
      var rand = random.nextDouble()
      rand = if(rand <= 0.0) eps else rand
      val ki = Math.log(rand) / elem

      if(i < size){
        selectedIndex.enqueue(Tuple2(i,ki))
      }
      else{
        val mk = selectedIndex.head
        if(ki > mk._2){
          selectedIndex.dequeue()
          selectedIndex.enqueue(Tuple2(i,ki))
        }
      }
    }
    selectedIndex.map(s => s._1 ).toArray
  }

  def uniformIndex(total: Int,size: Int) = {
    require(total > 0 && size > 0)
    val selIndex = new Array[Int](size)
    val random = new Random()
    for(i <- 0 until size){
      selIndex(i) = random.nextInt(total)
    }
    selIndex
  }

  def reservoirUniformIndex(total: Int,size: Int) = {
    require(total > 0 && size > 0)
    val selIndex = new Array[Int](size)
    val random = new Random()
    var i: Int = 0
    while (i < size){
      selIndex(i) = i
    }
    i = size
    while(i < total){
      val cur = random.nextInt(i+1)
      if(cur < size){
        selIndex(cur) = i
      }
      i += 1
    }
    selIndex
  }
}
