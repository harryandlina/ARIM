package slpart.models.lenet

import java.text.SimpleDateFormat
import java.util.Date

import com.intel.analytics.bigdl.dataset.{DataSet, DistributedDataSet, MiniBatch, SampleToIDAryMiniBatch}
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Module}
import com.intel.analytics.bigdl.optim._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import slpart.dataloader.MnistLoader

object TrainMnist {
  val logger = Logger.getLogger("org")
  logger.setLevel(Level.OFF)

  import slpart.models.OptionUtils._

  def main(args: Array[String]): Unit = {
    trainParser.parse(args,new TrainParams()).map(param =>{
      val curTime = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date)
      val logFile = s"${curTime}-${param.appName}-bigdl.log"
      LoggerFilter.redirectSparkInfoLogs(logFile)
      val conf = Engine.createSparkConf()
        .setAppName(param.appName)
      val sc = new SparkContext(conf)
      Engine.init

      val virtualCores = vCores(conf)
      val start = System.nanoTime()
      val trainSamples = MnistLoader.trainingSamplesAry(sc,param.dataset,param.zeroScore,isAggregate = param.aggregate,category = param.classes,
        itqbitN = param.itqbitN,itqitN = param.itqitN,itqratioN = param.itqratioN,upBound = param.upBound,splitN = param.minPartN,isSparse = param.isSparse)

      val validationSamples = MnistLoader.validateSamples(sc,param.dataset,param.zeroScore)
      System.out.println(s"generate Aggregate Samples: ${(System.nanoTime()-start) * 1.0 / 1e9} seconds")

      val model = if(param.loadSnapshot && param.model.isDefined){
        Module.load[Float](param.model.get)
      }
      else{
        LeNet5(param.classes)
      }

      val optimMethod = if(param.loadSnapshot && param.state.isDefined){
        OptimMethod.load[Float](param.state.get)
      }else{
        param.optMethod match {
          case "adam" => new Adam[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay)
          case "adadelta" => new Adadelta[Float]()
          case "rmsprop" => new RMSprop[Float](param.learningRate,param.learningRateDecay)
          case "ftrl" => new Ftrl[Float](param.learningRate)
          case _ => new SGD[Float](learningRate = param.learningRate,learningRateDecay = param.learningRateDecay,
            weightDecay = param.weightDecay,momentum = param.momentum,
            dampening = param.dampening,nesterov = param.nesterov)
        }
      }

      if(param.storeInitModel && param.initModel.isDefined){
        System.out.println(s"save initial model in ${param.initModel.get}")
        model.save(param.initModel.get,true)
      }

      val trainDataset = (DataSet.rdd(trainSamples) -> SampleToIDAryMiniBatch(param.batchSize)).asInstanceOf[DistributedDataSet[MiniBatch[Float]]]
      val criterion = ClassNLLCriterion()
      val optimizer = new IDAryDistriOptimizer(model,trainDataset,_criterion = criterion)

      if(param.checkpoint.isDefined){
        optimizer.setCheckpoint(param.checkpoint.get,Trigger.severalIteration(param.checkpointIteration))
        if(param.overwriteCheckpoint) optimizer.overWriteCheckpoint()
      }

      val remvControler = new RemvControler()
      remvControler.isExtraCompute = param.extraCompute
      remvControler.layerName = param.layerName
      remvControler.gradName = param.gradName
      remvControler.removeCriterion = param.removeCriterion
      remvControler.excludeEpoch = param.excludeEpoch
      remvControler.excludeIteration = param.excludeIteration
      remvControler.limitRemoveFraction = param.limitRemoveFraction

      remvControler.isTaskRemove = param.isTaskRemove
      remvControler.taskStrategy = param.taskStrategy
      remvControler.taskFraction = param.taskFraction
      remvControler.excludeTaskFraction = param.excludeTaskFraction

      val prestate = T(
        ("removecontroler",remvControler)
      )
      // set user defined state
      optimizer.setState(prestate)

      optimizer.setValidation(
        trigger = Trigger.everyEpoch,
        sampleRDD = validationSamples,
        vMethods = Array(new Top1Accuracy(),new Top5Accuracy(),new Loss()),
        batchSize = param.batchSize
      )
      optimizer.setOptimMethod(optimMethod)
      optimizer.setEndWhen(Trigger.maxEpoch(param.maxEpoch))

      val trainedModel = optimizer.optimize()

      if(param.storeTrainedModel && param.trainedModel.isDefined){
        System.out.println(s"save trained model in ${param.trainedModel.get}")
        trainedModel.save(param.trainedModel.get,overWrite = true)
        if(param.trainedState.isDefined) {
          System.out.println(s"save trained state in ${param.trainedState.get}")
          optimMethod.save(param.trainedState.get,overWrite = true)
        }
      }

      sc.stop()
    })
  }

}
