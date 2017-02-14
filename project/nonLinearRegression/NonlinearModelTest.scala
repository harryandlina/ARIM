package AccurateML.nonLinearRegression


import breeze.linalg.{DenseVector => BDV}

/**
 * @author Nodalpoint
 */
object NonlinearModelTest {
  def main(args: Array[String]) = {
    /**
     * Input dimensionality n = 2
     */
    var n:Integer = 2
    /**
     * Let x be a random data point 
     */
    var x: BDV[Double] = BDV.rand[Double](n)
    /**
     * Define one neural network with 2 hidden layers and one
     * Gaussian mixture with two gaussians.
     */
    var nnModel: NonlinearModel = new NeuralNetworkModel(n, 2)
    var gmmModel: NonlinearModel = new GaussianMixtureModel(n, 2)
    /**
     * Get the dimensionality of tunable parameters for each model
     */
    var nnDim: Int = nnModel.getDim()
    var gmmDim: Int = gmmModel.getDim()
    
    /**
     * Define a random initial set of parameters 
     */
    var nnW:  BDV[Double] = BDV.rand[Double](nnDim)
    var gmmW:  BDV[Double] = BDV.rand[Double](gmmDim)
    
    /**
     * Evaluate the model for input x on parameters nnW
     * nnW[0] weight 1st input to 1st hidden
     * nnW[1] weight 2nd input to 1st hidden
     * nnW[2] weight      bias to 1st hidden
     * nnW[3] weight 1st hidden to output 
     */
    System.out.println("Using weights " + nnW)
    System.out.println("On input x " + x)
    System.out.println("NN eval " + nnModel.eval(nnW, x))
    System.out.println("NN grad analytic " + nnModel.grad(nnW, x))
    System.out.println("NN grad numeric " + nnModel.gradnumer(nnW, x))
    
     /**
     * Evaluate the model for input x on parameters gmmW
     * gmmW[0] 1st gaussian 1st component of mean vector
     * gmmW[1] 1st gaussian 2nd component of mean vector
     * gmmW[2] 1st gaussian 1st component of diagonal covariance
     * gmmW[3] 1st gaussian 2nd component of diagonal covariance
     * gmmW[4] 1st gaussian scale factor alpha
     */
    System.out.println("Using weights " + gmmW)
    System.out.println("On input x " + x)    
    System.out.println("GMM eval " + gmmModel.eval(gmmW, x))
    System.out.println("GMM grad analytic " + gmmModel.grad(gmmW, x))
    System.out.println("GMM grad numeric " + gmmModel.gradnumer(gmmW, x))
  }
}