package AccurateML.nonLinearRegression

import java.io.{PrintWriter, File}

import breeze.linalg.DenseVector

import scala.io.Source

/**
  * Created by zhangfan on 16/12/19.
  */
class NNCreateWeights {

}
object NNCreateWeights{
  def main(args: Array[String]) {
    val numFeature=5231
    val hiddenNodesN=5
    val dim = new NeuralNetworkModel(numFeature, hiddenNodesN).getDim()
    val writer = new PrintWriter(new File(""+numFeature+"."+hiddenNodesN))
    val w=DenseVector.rand[Double](dim)
    writer.write(w.toArray.mkString(",")+"\n")
    writer.close()
  }
}
