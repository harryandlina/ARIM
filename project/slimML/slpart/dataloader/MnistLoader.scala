package slpart.dataloader

import java.nio.ByteBuffer
import java.nio.file.{Files, Path, Paths}

import com.intel.analytics.bigdl.dataset.{ByteRecord, LabeledPointToSample}
import com.intel.analytics.bigdl.dataset.image.{BytesToGreyImg, GreyImgNormalizer, GreyImgToLabeledPoint, GreyImgToSample}
import com.intel.analytics.bigdl.utils.File
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.numeric.NumericFloat

object MnistLoader {
  private val hdfsPrefix: String = "hdfs:"
  val trainMean = 0.13066047740239506
  val trainStd = 0.3081078

  val testMean = 0.13251460696903547
  val testStd = 0.31048024

  /**
    * load mnist data.
    * read mnist from hdfs if data folder starts with "hdfs:", otherwise form local file.
    * @param featureFile
    * @param labelFile
    * @return
    */
  def load(featureFile: String, labelFile: String): Array[ByteRecord] = {

    val featureBuffer = if (featureFile.startsWith(hdfsPrefix)) {
      ByteBuffer.wrap(File.readHdfsByte(featureFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(featureFile)))
    }
    val labelBuffer = if (featureFile.startsWith(hdfsPrefix)) {
      ByteBuffer.wrap(File.readHdfsByte(labelFile))
    } else {
      ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)))
    }
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[ByteRecord](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum))
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = ByteRecord(img, labelBuffer.get().toFloat + 1.0f) // change 0-base label to 1-base label
      i += 1
    }
    result
  }

  /**
    * get the training data
    * @param path the directory where dataset in
    * @param isZeroScore weather to zeroScore training data
    * @return
    */
  def trainingLabeledPoint(path: String,isZeroScore: Boolean = true) = {
    val trainImagePath = path + "/train-images-idx3-ubyte"
    val trainLabelPath = path + "/train-labels-idx1-ubyte"
    val byteRecords = load(trainImagePath,trainLabelPath)
    if(isZeroScore){
      System.out.println("zero Score training labeledPoint")
      val trans = BytesToGreyImg(28,28) -> GreyImgNormalizer(trainMean,trainStd) -> GreyImgToLabeledPoint()
      trans.apply(byteRecords.toIterator).toArray
    }
    else{
      val trans = BytesToGreyImg(28,28) -> GreyImgToLabeledPoint()
      trans.apply(byteRecords.toIterator).toArray
    }
  }

  /**
    * get the validate data
    * @param path the directory where dataset in
    * @param isZeroScore weather to zeroScore validate data
    * @return
    */
  def validationSamples(path: String,isZeroScore: Boolean = true) = {
    val trainImagePath = path + "/t10k-images-idx3-ubyte"
    val trainLabelPath = path + "/t10k-labels-idx1-ubyte"
    val byteRecords = load(trainImagePath,trainLabelPath)
    if(isZeroScore){
      System.out.println("zero Score validate labeledPoint")
      val trans = BytesToGreyImg(28,28) -> GreyImgNormalizer(testMean,testStd) -> GreyImgToSample()
      trans.apply(byteRecords.toIterator).toArray
    }
    else{
      val trans = BytesToGreyImg(28,28) -> GreyImgToSample()
      trans.apply(byteRecords.toIterator).toArray
    }
  }
  def trainingSamples(sc: SparkContext,path: String,zScore: Boolean = true) = {
    val trainImagePath = path + "/train-images-idx3-ubyte"
    val trainLabelPath = path + "/train-labels-idx1-ubyte"
    val byteRecords = load(trainImagePath,trainLabelPath)
    if(zScore){
      System.out.println("zero Score training labeledPoint")
      val trans = BytesToGreyImg(28,28) -> GreyImgNormalizer(trainMean,trainStd) -> GreyImgToSample()
      trans.apply(byteRecords.toIterator).toArray
    }
    else{
      val trans = BytesToGreyImg(28,28) -> GreyImgToSample()
      trans.apply(byteRecords.toIterator).toArray
    }
  }

  def trainingSamplesAry(sc: SparkContext,path: String,zScore: Boolean = true,isAggregate: Boolean = false,category: Int = 10,
                      itqbitN: Int = 1,
                      itqitN: Int = 20, //压缩算法迭代次数
                      itqratioN: Int = 100, //压缩算法每itqratioN个属性中使用一个属性进行计算
                      upBound: Int = 20, //每个压缩点包含原始点个数上限
                      splitN: Double = 2.0,// svd产生itqBitN个新的属性,对每个新的属性划分成splitN份
                      isSparse: Boolean = false //输入数据是否为libsvm格式
                     ) = {
    if(isAggregate){
      val trainlp = trainingLabeledPoint(path,isZeroScore = zScore)
      System.out.println("generate compressed training Samples  ...\n +" +
        s"category: ${category} itqbitN: ${itqbitN} itqitN: ${itqitN} itqratioN: ${itqratioN} upBound: ${upBound}" +
        s" splitN: ${splitN} isSparse: ${isSparse}")
      val tp = Aggregator.singleLayerAggregateAry(category,sc.parallelize(trainlp),
        itqbitN = itqbitN, itqitN = itqitN, itqratioN = itqratioN, upBound = upBound,
        splitN = splitN, isSparse = isSparse)
      tp.zipWithUniqueId().map(x => (x._2,x._1))
    }
    else{
      val trainsp = trainingSamples(sc,path,zScore)
      val arySap = trainsp.zipWithIndex.map(x => (x._2.toLong,Array(x._1)))
      val tp = sc.parallelize(arySap)
      tp
    }
  }
  def validateSamples(sc: SparkContext,path: String,zScore: Boolean = true) = {
    val validatesp = validationSamples(path,isZeroScore = zScore)
    sc.parallelize(validatesp)
  }
}
