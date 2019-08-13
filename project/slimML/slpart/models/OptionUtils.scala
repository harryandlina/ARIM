package slpart.models

import org.apache.log4j.{ConsoleAppender, Level, PatternLayout}
import org.apache.spark.SparkConf
import scopt.OptionParser
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import java.io.{File, PrintWriter}
import scala.io.Source

object OptionUtils {

  def main(args: Array[String]): Unit = {
    trainParser.parse(args,new TrainParams()).map(param => {
      System.out.println(s"appName: ${param.appName}")
      System.out.println(s"dataset: ${param.dataset}")
      System.out.println(s"zeroScore: ${param.zeroScore}")

      System.out.println(s"checkpoint: ${param.checkpoint}")
      System.out.println(s"checkpoint: ${param.checkpointIteration}")
      System.out.println(s"overwriteCheckpoint: ${param.overwriteCheckpoint}")
      System.out.println(s"loadShapshot: ${param.loadSnapshot}")
      System.out.println(s"model: ${param.model}")
      System.out.println(s"state: ${param.state}")
      System.out.println(s"summary: ${param.summary}")

      System.out.println(s"classes: ${param.classes}")
      System.out.println(s"optMethod: ${param.optMethod}")
      System.out.println(s"learningRate: ${param.learningRate}")
      System.out.println(s"learningRateDecay: ${param.learningRateDecay}")
      System.out.println(s"weightDecay: ${param.weightDecay}")
      System.out.println(s"momentum: ${param.momentum}")
      System.out.println(s"dampening: ${param.dampening}")
      System.out.println(s"nesterov: ${param.nesterov}")
      System.out.println(s"lrScheduler: ${param.lrScheduler}")

      System.out.println(s"optnet: ${param.optnet}")
      System.out.println(s"depth: ${param.depth}")
      System.out.println(s"shortcutType: ${param.shortcutType}")

      System.out.println(s"graphModel: ${param.graphModel}")
      System.out.println(s"maxLearningRate: ${param.maxLearningRate}")
      System.out.println(s"warmupEpoch: ${param.warmupEpoch}")
      System.out.println(s"hasDropout: ${param.hasDropout}")
      System.out.println(s"hasBN: ${param.hasBN}")

      System.out.println(s"batchSize: ${param.batchSize}")
      System.out.println(s"maxEpoch: ${param.maxEpoch}")
      System.out.println(s"maxIteration: ${param.maxIteration}")
      System.out.println(s"validateIteration: ${param.validateIteration}")

      System.out.println(s"storeInitModel: ${param.storeInitModel}")
      System.out.println(s"initModel: ${param.initModel}")
      System.out.println(s"initState: ${param.initState}")
      System.out.println(s"storeTrainedModel: ${param.storeTrainedModel}")
      System.out.println(s"trainedModel: ${param.trainedModel}")
      System.out.println(s"trainedState: ${param.trainedState}")

      System.out.println(s"aggregate: ${param.aggregate}")
      System.out.println(s"itqbitN: ${param.itqbitN}")
      System.out.println(s"itqitN: ${param.itqitN}")
      System.out.println(s"itqratioN: ${param.itqratioN}")
      System.out.println(s"minPartN: ${param.minPartN}")
      System.out.println(s"upBound: ${param.upBound}")
      System.out.println(s"isSparse: ${param.isSparse}")

      System.out.println(s"extraCompute: ${param.extraCompute}")
      System.out.println(s"layerName: ${param.layerName}")
      System.out.println(s"gradName: ${param.gradName}")
      System.out.println(s"removeCriterion: ${param.removeCriterion}")
      System.out.println(s"excludeEpoch: ${param.excludeEpoch}")
      System.out.println(s"excludeIteration: ${param.excludeIteration}")
      System.out.println(s"limitRemoveFraction: ${param.limitRemoveFraction}")

      System.out.println(s"isTaskRemove: ${param.isTaskRemove}")
      System.out.println(s"taskStrategy: ${param.taskStrategy}")
      System.out.println(s"taskFraction: ${param.taskFraction}")
      System.out.println(s"excludeTaskFraction: ${param.excludeTaskFraction}")

      System.out.println(s"isEpochRemove: ${param.isEpochRemove}")
      System.out.println(s"epochStrategy: ${param.epochStrategy}")
      System.out.println(s"epochFraction: ${param.epochFraction}")
      System.out.println(s"excludeEpochFraction: ${param.excludeEpochFraction}")

      System.out.println(s"isAggregatedSamples: ${param.isAggregatedSamples}")
      System.out.println(s"isTrainWithAggregatedSamples: ${param.isTrainWithAggregatedSamples}")
      System.out.println(s"infoOfIteration: ${param.infoOfIteration}")

      System.out.println(s"lipLambda: ${param.lipLambda}")
      System.out.println(s"kradius: ${param.kradius}")
      System.out.println(s"kk: ${param.kk}")
      System.out.println(s"kmaxIter: ${param.kmaxIter}")
      System.out.println(s"kinitMethod: ${param.kinitMethod}")
      System.out.println(s"outputSizeProportion: ${param.outputSizeProportion}")
      System.out.println(s"ksubsampleSize: ${param.ksubsampleSize}")

    })
  }


  case class TrainParams(
                        appName: String = "appName",
                        dataset: String = "./",
                        zeroScore: Boolean = true,

                        checkpoint: Option[String] = None,
                        checkpointIteration: Int = 10000,
                        overwriteCheckpoint: Boolean = true,
                        loadSnapshot: Boolean = false,
                        model: Option[String] = None,
                        state: Option[String] = None,
                        summary: Option[String] = None,

                        classes: Int = 10,
                        optMethod: String = "sgd",
                        learningRate: Double = 0.001,
                        learningRateDecay: Double = 0.0,
                        weightDecay: Double = 0.0,
                        momentum: Double = 0.0,
                        dampening: Double = Double.MaxValue,
                        nesterov: Boolean = false,
                        lrScheduler: Option[String] = None,

                        optnet: Boolean = false,
                        depth: Int = 20,
                        shortcutType: String = "A",

                        graphModel: Boolean = false,
                        maxLearningRate: Double = 0.5,
                        warmupEpoch: Int = 0,
                        hasDropout: Boolean = false,
                        hasBN: Boolean = false,

                        batchSize: Int = 128,
                        maxEpoch: Int = 10,
                        maxIteration: Int = 1000,
                        validateIteration: Int = 100,

                        storeInitModel: Boolean = false,
                        initModel: Option[String] = None,
                        initState: Option[String] = None,
                        storeTrainedModel: Boolean = false,
                        trainedModel: Option[String] = None,
                        trainedState: Option[String] = None,

                        aggregate: Boolean = false,
                        itqbitN: Int = 1,
                        itqitN: Int = 20,
                        itqratioN: Int = 100,
                        minPartN: Int = 2,
                        upBound: Int = 20,
                        isSparse: Boolean = false,

                        extraCompute: Boolean = false,
                        layerName: String = "conv_1",
                        gradName: String = "gradWeight",
                        removeCriterion: String = "loss",
                        excludeEpoch: Int = 5,
                        excludeIteration: Int = 200,
                        limitRemoveFraction: Double = 1.01,

                        isTaskRemove: Boolean = false,
                        taskStrategy: String = "mean",
                        taskFraction: Double = 1.0,
                        excludeTaskFraction: Double = 0.0,

                        isEpochRemove: Boolean = false,
                        epochStrategy: String = "mean",
                        epochFraction: Double = 1.0,
                        excludeEpochFraction: Double = 0.0,

                        isAggregatedSamples: Boolean = false,
                        isTrainWithAggregatedSamples: Boolean = false,
                        infoOfIteration: Option[String] = None,

                        lipLambda: Double = 0.5,
                        kradius: Double = 1.0,
                        kk: Int = 10,
                        kmaxIter: Int = 20,
                        kinitMethod: String = "k-means||",
                        outputSizeProportion: Double = 0.025,
                        ksubsampleSize: String = "auto auto"
                        )
  val trainParser = new OptionParser[TrainParams]("train option parser") {
    opt[String]("appName")
      .text("the application name")
      .action((x,c) => c.copy(appName = x))
    opt[String]('f',"dataset")
      .text("where you put training data")
      .action((x,c) => c.copy(dataset = x))
    opt[Boolean]("zeroScore")
      .text("normalize dataset")
      .action((x,c) => c.copy(zeroScore = x))

    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x,c) => c.copy(checkpoint = Some(x)))
    opt[Int]("checkpointIteration")
      .text("when to cache the model")
      .action((x,c) => c.copy(checkpointIteration = x))
    opt[Boolean]("overwriteCheckpoint")
      .text("overwrite checkpoint")
      .action((x,c) => c.copy(overwriteCheckpoint = x))
    opt[Boolean]("loadSnapshot")
      .text("load model if exists")
      .action((x,c) => c.copy(loadSnapshot = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x,c) => c.copy(model = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x,c) => c.copy(state = Some(x)))
    opt[String]("summary")
      .text("where to save summary")
      .action((x,c) => c.copy(summary = Some(x)))

    opt[Int]('c',"classes")
      .text("the number of category")
      .action((x,c) => c.copy(classes = x))
    opt[String]("optMethod")
      .text("which optimize method to use")
      .action((x,c) => c.copy(optMethod = x))
    opt[Double]("learningRate")
      .text("initial learning rate")
      .action((x,c) => c.copy(learningRate = x))
    opt[Double]("learningRateDecay")
      .text("initial learning rate decay")
      .action((x,c) => c.copy(learningRateDecay = x))
    opt[Double]("weightDecay")
      .text("weight decay")
      .action((x,c) => c.copy(weightDecay = x))
    opt[Double]("momentum")
      .text("momentum")
      .action((x,c) => c.copy(momentum = x))
    opt[Double]("dampening")
      .text("dampening")
      .action((x,c) => c.copy(dampening = x))
    opt[Boolean]("nesterov")
      .text("nesterov")
      .action((x,c) => c.copy(nesterov = x))
    opt[String]("lrScheduler")
      .text("learning rate scheduler")
      .action((x,c) => c.copy(lrScheduler = Some(x)))

    opt[Boolean]("optnet")
      .text("shared gradients and caches to reduce memory usage")
      .action((x,c) => c.copy(optnet = x))
    opt[Int]("depth")
      .text("depth of ResNet, 18 | 20 | 34 | 50 | 101 | 152 | 200")
      .action((x,c) => c.copy(depth = x))
    opt[String]("shortcutType")
      .text("shortcutType of ResNet, A | B | C")
      .action((x,c) => c.copy(shortcutType = x))

    opt[Boolean]("graphModel")
      .text("use graph model")
      .action((x,c) => c.copy(graphModel = x))
    opt[Double]("maxLearningRate")
      .text("max learning rate")
      .action((x,c) => c.copy(maxLearningRate = x))
    opt[Int]("warmupEpoch")
      .text("warm up Epoch")
      .action((x,c) => c.copy(warmupEpoch = x))
    opt[Boolean]("hasDropout")
      .text("use dropout")
      .action((x,c) => c.copy(hasDropout = x))
    opt[Boolean]("hasBN")
      .text("use BN layer")
      .action((x,c) => c.copy(hasBN = x))

    opt[Int]('b',"batchSize")
      .text("batchSize")
      .action((x,c) => c.copy(batchSize = x))
    opt[Int]("maxEpoch")
      .text("number of epoch to train")
      .action((x,c) => c.copy(maxEpoch = x))
    opt[Int]("maxIteration")
      .text("number of iteration to train")
      .action((x,c) => c.copy(maxIteration = x))
    opt[Int]("validateIteration")
      .text("number of iteration to validate")
      .action((x,c) => c.copy(validateIteration = x))

    opt[Boolean]("storeInitModel")
      .text("save init model")
      .action((x,c) => c.copy(storeInitModel = x))
    opt[String]("initModel")
      .text("path of initial model")
      .action((x,c) => c.copy(initModel = Some(x)))
    opt[String]("initState")
      .text("path of initial state")
      .action((x,c) => c.copy(initState = Some(x)))
    opt[Boolean]("storeTrainedModel")
      .text("save trained model")
      .action((x,c) => c.copy(storeTrainedModel = x))
    opt[String]("trainedModel")
      .text("path of trained model")
      .action((x,c) => c.copy(trainedModel = Some(x)))
    opt[String]("trainedState")
      .text("path of trained state")
      .action((x,c) => c.copy(trainedState = Some(x)))

    opt[Boolean]("aggregate")
      .text("aggregate dataset")
      .action((x,c) => c.copy(aggregate = x))
    opt[Int]("itqbitN")
      .text("itqbitN")
      .action((x,c) => c.copy(itqbitN = x))
    opt[Int]("itqitN")
      .text("iteration of aggregator")
      .action((x,c) => c.copy(itqitN = x))
    opt[Int]("itqratioN")
      .text("select one attribution per itqratioN attribution")
      .action((x,c) => c.copy(itqratioN = x))
    opt[Int]("minPartN")
      .text("slice minPartN Part every attribution")
      .action((x,c) => c.copy(minPartN = x))
    opt[Int]("upBound")
      .text("the upperBound of each Aggregated point")
      .action((x,c) => c.copy(upBound = x))
    opt[Boolean]("isSparse")
      .text("libsvm format dataset")
      .action((x,c) => c.copy(isSparse = x))

    opt[Boolean]("extraCompute")
      .text("to compute non-critical samples")
      .action((x,c) => c.copy(extraCompute = x))
    opt[String]("layerName")
      .text("which layer to obtain gradient")
      .action((x,c) => c.copy(layerName = x))
    opt[String]("gradName")
      .text("name of gradient")
      .action((x,c) => c.copy(gradName = x))
    opt[String]("removeCriterion")
      .text("criterion of judging non-critical samples")
      .action((x,c) => c.copy(removeCriterion = x))
    opt[Int]("excludeEpoch")
      .text("do not remove non-critical at first excludeEpoch epoch")
      .action((x,c) => c.copy(excludeEpoch = x))
    opt[Int]("excludeIteration")
      .text("do not remove non-critical at first excludeIteration iteration")
      .action((x,c) => c.copy(excludeIteration = x))
    opt[Double]("limitRemoveFraction")
      .text("limitation of most fraction of non-critical")
      .action((x,c) => c.copy(limitRemoveFraction = x))

    opt[Boolean]("isTaskRemove")
      .text("remove non-critical every iteration")
      .action((x,c) => c.copy(isTaskRemove = x))
    opt[String]("taskStrategy")
      .text("iteration remove strategy")
      .action((x,c) => c.copy(taskStrategy = x))
    opt[Double]("taskFraction")
      .text("fraction of criterion")
      .action((x,c) => c.copy(taskFraction = x))
    opt[Double]("excludeTaskFraction")
      .text("the fraction need to exclude per iteration")
      .action((x,c) => c.copy(excludeTaskFraction = x))

    opt[Boolean]("isEpochRemove")
      .text("remove non-critical every epoch")
      .action((x,c) => c.copy(isEpochRemove = x))
    opt[String]("epochStrategy")
      .text("epoch remove strategy")
      .action((x,c) => c.copy(epochStrategy = x))
    opt[Double]("epochFraction")
      .text("fraction of criterion")
      .action((x,c) => c.copy(epochFraction = x))
    opt[Double]("excludeEpochFraction")
      .text("the fraction need to exclude per epcoh")
      .action((x,c) => c.copy(excludeEpochFraction = x))

    opt[Boolean]("isAggregatedSamples")
      .text("train with only aggregated samples")
      .action((x,c) => c.copy(isAggregatedSamples = x))
    opt[Boolean]("isTrainWithAggregatedSamples")
      .text("train with aggregated samples and original samples")
      .action((x,c) => c.copy(isTrainWithAggregatedSamples = x))
    opt[String]("infoOfIteration")
      .text("the iteration to collect criterion information")
      .action((x,c) => c.copy(infoOfIteration = Some(x)))

    opt[Double]("lipLambda")
      .text("lambda in lipschitz")
      .action((x,c) => c.copy(lipLambda = x))
    opt[Double]("kradius")
      .text("the ball radius of coreset")
      .action((x,c) => c.copy(kradius = x))
    opt[Int]("kk")
      .text("the k of k-means")
      .action((x,c) => c.copy(kk = x))
    opt[Int]("kmaxIter")
      .text("max iteration of k-means")
      .action((x,c) => c.copy(kmaxIter = x))
    opt[String]("kinitMethod")
      .text("initial method of k-means")
      .action((x,c) => c.copy(kinitMethod = x))
    opt[Double]("outputSizeProportion")
      .text("output proportion of coreset")
      .action((x,c) => c.copy(outputSizeProportion = x))
    opt[String]("ksubsampleSize")
      .text("the size used to calculate k-means centers")
      .action((x,c) => c.copy(ksubsampleSize = x))

  }

  case class TestParams(
                       dataset: String = "./",
                       model: String = "",
                       batchSize: Int = 128
                       )
  val testParser = new OptionParser[TestParams]("test option parser") {
    opt[String]('f',"dataset")
      .text("where you put test data")
      .action((x,c) => c.copy(dataset = x))
    opt[String]("model")
      .text("path of model")
      .action((x,c) => c.copy(model = x))
    opt[Int]('b',"batchSize")
      .text("batchSize")
      .action((x,c) => c.copy(batchSize = x))
  }

  def getNumMachineCores: Int = {
    val coreNum = Runtime.getRuntime().availableProcessors()
    require(coreNum > 0, "Get a non-positive core number")
    // We assume the HT is enabled
    // Todo: check the Hyper threading
    if (coreNum > 1) coreNum / 2 else 1
  }
  def dynamicAllocationExecutor(conf: SparkConf): Option[Int] = {
    if (conf.get("spark.dynamicAllocation.enabled", null) == "true") {
      val maxExecutors = conf.get("spark.dynamicAllocation.maxExecutors", "1").toInt
      val minExecutors = conf.get("spark.dynamicAllocation.minExecutors", "1").toInt
      require(maxExecutors == minExecutors, "Engine.init: " +
        "spark.dynamicAllocation.maxExecutors and " +
        "spark.dynamicAllocation.minExecutors must be identical " +
        "in dynamic allocation for BigDL")
      Some(minExecutors)
    } else {
      None
    }
  }
  def parseExecutorAndCore(conf: SparkConf): Option[(Int, Int)] = {
    val master = conf.get("spark.master", null)
    if (master.toLowerCase.startsWith("local")) {
      // Spark local mode
      val patternLocalN = "local\\[(\\d+)\\]".r
      val patternLocalStar = "local\\[\\*\\]".r
      master match {
        case patternLocalN(n) => Some(1, n.toInt)
        case patternLocalStar(_*) => Some(1, getNumMachineCores)
        case _ => throw new IllegalArgumentException(s"Can't parser master $master")
      }
    } else if (master.toLowerCase.startsWith("spark")) {
      // Spark standalone mode
      val coreString = conf.get("spark.executor.cores", null)
      val maxString = conf.get("spark.cores.max", null)
      require(coreString != null, "Engine.init: Can't find executor core number" +
        ", do you submit with --executor-cores option")
      require(maxString != null, "Engine.init: Can't find total core number" +
        ". Do you submit with --total-executor-cores")
      val core = coreString.toInt
      val nodeNum = dynamicAllocationExecutor(conf).getOrElse {
        val total = maxString.toInt
        require(total >= core && total % core == 0, s"Engine.init: total core " +
          s"number($total) can't be divided " +
          s"by single core number($core) provided to spark-submit")
        total / core
      }
      Some(nodeNum, core)
    } else if (master.toLowerCase.startsWith("yarn")) {
      // yarn mode
      val coreString = conf.get("spark.executor.cores", null)
      require(coreString != null, "Engine.init: Can't find executor core number" +
        ", do you submit with " +
        "--executor-cores option")
      val core = coreString.toInt
      val node = dynamicAllocationExecutor(conf).getOrElse {
        val numExecutorString = conf.get("spark.executor.instances", null)
        require(numExecutorString != null, "Engine.init: Can't find executor number" +
          ", do you submit with " +
          "--num-executors option")
        numExecutorString.toInt
      }
      Some(node, core)
    } else if (master.toLowerCase.startsWith("mesos")) {
      // mesos mode
      require(conf.get("spark.mesos.coarse", null) != "false", "Engine.init: " +
        "Don't support mesos fine-grained mode")
      val coreString = conf.get("spark.executor.cores", null)
      require(coreString != null, "Engine.init: Can't find executor core number" +
        ", do you submit with --executor-cores option")
      val core = coreString.toInt
      val nodeNum = dynamicAllocationExecutor(conf).getOrElse {
        val maxString = conf.get("spark.cores.max", null)
        require(maxString != null, "Engine.init: Can't find total core number" +
          ". Do you submit with --total-executor-cores")
        val total = maxString.toInt
        require(total >= core && total % core == 0, s"Engine.init: total core " +
          s"number($total) can't be divided " +
          s"by single core number($core) provided to spark-submit")
        total / core
      }
      Some(nodeNum, core)
    } else if (master.toLowerCase.startsWith("k8s")) {
      // Spark-on-kubernetes mode
      val coreString = conf.get("spark.executor.cores", null)
      val maxString = conf.get("spark.cores.max", null)
      require(coreString != null, "Engine.init: Can't find executor core number" +
        ", do you submit with --conf spark.executor.cores option")
      require(maxString != null, "Engine.init: Can't find total core number" +
        ". Do you submit with --conf spark.cores.max option")
      val core = coreString.toInt
      val nodeNum = dynamicAllocationExecutor(conf).getOrElse {
        val total = maxString.toInt
        require(total >= core && total % core == 0, s"Engine.init: total core " +
          s"number($total) can't be divided " +
          s"by single core number($core) provided to spark-submit")
        total / core
      }
      Some(nodeNum, core)
    } else {
      throw new IllegalArgumentException(s"Engine.init: Unsupported master format $master")
    }
  }
  def vCores(conf: SparkConf) = {
    val opt = parseExecutorAndCore(conf)
    val node = opt.get._1
    val core = opt.get._2
    node * core
  }

  private val pattern = "%d{yyyy-MM-dd HH:mm:ss} %-5p %c{1}:%L - %m%n"
  def consoleAppender(level: Level = Level.INFO): ConsoleAppender = {
    val console = new ConsoleAppender
    console.setLayout(new PatternLayout(pattern))
    console.setThreshold(level)
    console.activateOptions()
    console.setTarget("System.out")

    console
  }

  def saveBDV(filePath: String,weights: BDV[Double]) = {
    val outWriter = new PrintWriter(new File(filePath))
    outWriter.write(weights.toArray.mkString(","))
    outWriter.close()
  }
  def loadBDV(filePath: String) = {
    val iter = Source.fromFile(new File(filePath)).getLines()
    val weights = iter.next().split(",").map(_.toDouble)
    new BDV(weights)
  }
}
