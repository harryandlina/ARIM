package AccurateML.nonLinearRegression


import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.Vectors

/**
 * @author Nodalpoint
 * A simple application to test the objective function implementation
 * Training data set:
 * <-----  Input ----->  Target
 * -1.000000,-1.000000,-0.389030
 * -1.000000,-0.800000,-0.479909
 * -1.000000,-0.600000,-0.550592
 *  ...
 *  ...
 *  1.000000,1.000000,-0.389030
 */

object SumOfSquareTest {

	def main(args: Array[String]) = {
    /* Input dimension is 2*/
    var n:Int = 2
    /* Number of hidden nodes for the Neural Network */
    var nodes:Int = 4
    /* Read training data : Breeze style*/
    var XY: BDM[Double] = breeze.linalg.csvread(new java.io.File("input.txt"))
    /* Number of training data*/
    var m:Int = XY.rows
    
    /*Create a neural network nolinear model with input dimension 2 and 4 hidden nodes*/
    var model: NonlinearModel = new NeuralNetworkModel(n, nodes)
    /* The dimensionality of the tunable parameters */
    var dim: Int = model.getDim()
    /* A random vector containing some initial parameters for the model 
     * We are not optimizing in this demo */
    var w = BDV.rand[Double](dim)
    
    /*
     * Define a SumOfSquaresFunction based on Breeze using the neural network model  
     */
    var modelfunBreeze :SumOfSquaresFunction = new  SumOfSquaresFunctionBreeze(model, XY)
    /*
     * Calculate function value and derivatives at the initial random point w 
     */
    var (f, g) = modelfunBreeze.calculate(w)
    var H = modelfunBreeze.hessian(w)
    System.out.println("--- Neural Network Model (Breeze Implementation) ---")
    System.out.println("Weights w = " + w)
    System.out.println("Error function = " + f)
    System.out.println("Gradient (first 5 elements)  :")
    System.out.println( g )
    System.out.println("Hessian : ")
    System.out.println(H(0, ::))
    
    
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("Sample Application").setMaster("local[2]")
    val sc = new SparkContext(conf)
    /* Read training data : Spark style*/
    val dataTxt = sc.textFile("input.txt", 2)
    /* Split the input data into target, input pairs (Double, Vector) */ 
    val data = dataTxt.map { 
        line =>
        val parts = line.split(',')
        (parts(parts.length-1).toDouble, Vectors.dense(parts.take(parts.length-1).map(_.toDouble)))
    }.cache() 
    /* Number of training data*/
    val numExamples = data.count()
    
     /*
     * Define a SumOfSquaresFunction based on RDDs using the neural network model  
     */   
    var modelfunRDD :SumOfSquaresFunction = new  SumOfSquaresFunctionRDD(model, data)
    /*
     * Calculate function value and derivatives at the initial random point w 
     */    
    var (f1, g1) = modelfunRDD.calculate(w)
    var H1 = modelfunRDD.hessian(w)
    System.out.println("--- Neural Network Model (RDD Implementation) ---")
    System.out.println("Weights w = " + w)
    System.out.println("Error function = " + f1)
    System.out.println("Gradient (first 5 elements)  :")
    System.out.println( g1)
    System.out.println("Hessian : ")
    System.out.println(H1(0, ::))
    
    /* 
     * Compare the two implementation results. They should be identical 
     * within the double precision accuracy
     */
    System.out.println("Diff f " + Math.abs(f-f1) )
    System.out.println("Diff g " + norm(g-g1) )
    System.out.println("Diff H row 0 " + norm(H(::, 0)-H1(::, 0)) )
    System.out.println("Diff H row 1 " + norm(H(::, 1)-H1(::, 1)) )
    System.out.println("Diff H row 2 " + norm(H(::, 2)-H1(::, 2)) )



	}

}