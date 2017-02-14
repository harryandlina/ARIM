package AccurateML.nonLinearRegression

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}


/**
 * @author Nodalpoint
 * Implementation of the sum-of-squares objective function for small-sized problems where training  data can fit in a
 * Breeze matrix. 
 * 
 *  Properties
 *    model: NonlinearModel -> The nonlinear model that defines the function
 *    data: DenseMatrix -> The training data used to calculate the loss in a form of a Breeze DenseMatrix
 *                         The first n columns are the input data and the n+1 column the target value
 */
class SumOfSquaresFunctionBreeze(fitmodel: NonlinearModel, xydata:BDM[Double]) extends  SumOfSquaresFunction{
  var model:NonlinearModel = fitmodel
  var dim = fitmodel.getDim()
  var data:BDM[Double] = xydata
  
  var m:Int = data.rows
  var n:Int = data.cols-1
  var Y:BDV[Double] = data(::, n)
  var X:BDM[Double] = data(::, 0 to n-1)

/**
 * Return the objective function dimensionality which is essentially the model's dimensionality  
 */
def getDim():Int = {
    return this.dim
}

/**
 * This method is inherited by Breeze DiffFunction. Given an input vector of weights it returns the 
 * objective function and the first order derivative
 */
def calculate(weights: BDV[Double]): (Double, BDV[Double]) = {
  assert(dim == weights.length)
  var f:Double = 0
  var grad:BDV[Double] = BDV.zeros(dim)
  
  var r:BDV[Double] = subsum(weights)
  f = 0.5 * (r.t * r)
  grad = granal(weights)

  return (f, grad)
}
  //zfchange
  def calculate(weights: BDV[Double],itN:Int): (Double, BDV[Double],Array[Double],Int) = {
//    assert(dim == weights.length)
//    var f:Double = 0
//    var grad:BDV[Double] = BDV.zeros(dim)
//
//    var r:BDV[Double] = subsum(weights)
//    f = 0.5 * (r.t * r)
//    grad = granal(weights)

//    return (f, grad,Array(1.0))
    return (0.0,BDV.zeros(1),Array(0.0),0)
  }

/**
 * This method returns a positive definite approximation of the Hessian matrix at point w 
 */
def hessian(weights: BDV[Double]): BDM[Double] = {
   var H = hanal(weights)
   
   var Hpos = posDef(H)
   return Hpos
}

/**
 * Helper method that returns a vector of errors per training input.
 * It uses model.eval() for each input 
 */
def subsum(weights: BDV[Double]):BDV[Double] = {
    var f:BDV[Double] = BDV.zeros(m)
    var per:Double = 0

    for (i <- 0 to m-1){
      var x: BDV[Double] = X(i, ::).t
      per = model.eval(weights, x)  
      f(i) = per - Y(i)
    }
    return f
}

/**
 * Helper method that returns the gradient of the objective function after 
 * iterating through whole the training set
 */
def granal(weights: BDV[Double]): BDV[Double] = {
   var g:BDV[Double] = BDV.zeros(dim)
   var gper:BDV[Double] = BDV.zeros(dim)
   var ff:BDV[Double] = subsum(weights)

   for (i<-0 to m-1){
     var x: BDV[Double] = X(i, ::).t
     gper = model.grad(weights, x)
     for (j<-0 to dim-1){
        g(j) = g(j) + 2.0*ff(i)*gper(j) 
     }
   }
   return g
}

/**
 * Helper method that returns the Jacobian matrix. Partial derivative of j-th pattern and i-th input 
 */
def janal(weights: BDV[Double]):BDM[Double] = {
  var J:BDM[Double] = BDM.zeros(m, dim)
  var gper:BDV[Double] = BDV.zeros(dim)

  for (i<-0 to m-1){
    var x: BDV[Double] = X(i, ::).t
    gper = model.grad(weights, x)
    for (j <- 0 to dim-1){
      J(i, j) = gper(j)             
    }
  }

  return J
}

/**
 * Helper method that returns the Hessian matrix approximation using the Jacobian
 */
def hanal(weights: BDV[Double]):BDM[Double] = {
  var J = janal(weights)

   return J.t * J
}

/**
 * Helper method that uses eigenvalue decomposition to enforce positive definiteness of the Hessian
 */
def posDef(H:BDM[Double]):BDM[Double] = {
    var n = H.rows
    var m = H.cols
    var dim = model.getDim()
    var Hpos :BDM[Double] = BDM.zeros(n, m)

    var eigens = eigSym(H)
    var oni:BDV[Double] = BDV.ones[Double](dim)
    var diag = eigens.eigenvalues
    var vectors = eigens.eigenvectors


   // diag = diag :+ (1.0e-4)
    for (i<-0 until dim){
        if (diag(i) < 1.0e-4){
          diag(i) = 1.0e-4
        }
    }

    var I = BDM.eye[Double](dim)
    for (i<-0 until dim){
      I(i, i) = diag(i)
    }

    Hpos = vectors*I*(vectors.t)
    return Hpos
  }    
}