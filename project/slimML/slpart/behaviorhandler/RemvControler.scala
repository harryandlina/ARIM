package com.intel.analytics.bigdl.optim

import scala.beans.BeanProperty

class RemvControler extends Serializable {
  var isExtraCompute: Boolean = false

  var layerName: String = "default"
  var gradName: String = "gradWeight"

  var removeCriterion: String = "loss" // the criterion of judging non-critical samples,default is loss

  var excludeEpoch: Int = 5
  var excludeIteration: Int = 200

  var limitRemoveFraction: Double = 1.0

  // task non-critical related
  var isTaskRemove: Boolean = false
  var taskStrategy: String = "default" // default is mean of loss
  var taskFraction: Double = 1.0  // default taskStrategy set different value
  var excludeTaskFraction: Double = 0.0 // the fraction to exclude when remove

  // epoch non-critical related
  var isEpochRemove: Boolean = false
  var epochStrategy: String = "default" // default is mean of loss
  var epochFraction: Double = 1.0  // default taskStrategy set different value
  var excludeEpochFraction: Double = 0.0 // the fraction to exclude when remove

  var isAggregatedSamples: Boolean = false  // weather generated aggregated samples
  var isTrainWithAggregatedSamples: Boolean = false // use aggregated sample to train model

  @BeanProperty
  var infoOfIteration: Array[Int] = Array[Int]()  // the iteration to collect criterion information

  override def toString: String = {
    s"${getClass.getName}:{" +
      s"\nisExtraCompute: ${isExtraCompute}," +
      s"\nlayerName: ${layerName}," +
      s"\ngradName: ${gradName}," +
      s"\nremoveCriterion: ${removeCriterion}," +
      s"\nexcludeEpoch: ${excludeEpoch}," +
      s"\nexcludeIteration: ${excludeIteration}," +
      s"\nlimitRemoveFraction: ${limitRemoveFraction}," +
      s"\nisTaskRemove: ${isTaskRemove}," +
      s"\ntaskStrategy: ${taskStrategy}," +
      s"\ntaskFraction: ${taskFraction}," +
      s"\nexcludeTaskFraction: ${excludeTaskFraction}," +
      s"\nisEpochRemove: ${isEpochRemove}," +
      s"\nepochStrategy: ${epochStrategy}," +
      s"\nepochFraction: ${epochFraction}," +
      s"\nexcludeEpochFraction: ${excludeEpochFraction}," +
      s"\nisAggregatedSamples: ${isAggregatedSamples}," +
      s"\nisTrainWithAggregatedSamples: ${isTrainWithAggregatedSamples}," +
      s"\ninfoOfIteration: ${infoOfIteration.mkString(",")}\n}"
  }

  def this(isExtraCompute: Boolean,layerName: String,gradName: String,removeCriterion: String,
           isTaskRemove: Boolean,taskStrategy: String,taskFraction: Double){
    this()
    this.isExtraCompute = isExtraCompute

    this.layerName = layerName
    this.gradName = gradName

    this.removeCriterion = removeCriterion

    this.isTaskRemove = isTaskRemove
    this.taskStrategy = taskStrategy
    this.taskFraction = taskFraction
  }

  def this(isExtraCompute: Boolean,layerName: String,gradName: String,removeCriterion: String,
           excludeEpoch: Int,excludeIteration: Int,
           isTaskRemove: Boolean,taskStrategy: String,taskFraction: Double){
    this(isExtraCompute,layerName,gradName,removeCriterion,
      isTaskRemove,taskStrategy,taskFraction)

    this.excludeEpoch = excludeEpoch
    this.excludeIteration = excludeIteration
  }

  def this(isExtraCompute: Boolean,layerName: String,gradName: String,removeCriterion: String,
           excludeEpoch: Int,excludeIteration: Int,
           isTaskRemove: Boolean,taskStrategy: String,taskFraction: Double,
           isAggregatedSamples: Boolean,isTrainWithAggregatedSamples: Boolean){
    this(isExtraCompute,layerName,gradName,removeCriterion,
      excludeEpoch,excludeIteration,
      isTaskRemove,taskStrategy,taskFraction)

    this.isAggregatedSamples = isAggregatedSamples
    this.isTrainWithAggregatedSamples = isTrainWithAggregatedSamples
  }

}
