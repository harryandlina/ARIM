package AccurateML.nonLinearRegression


import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

/**
 * @author osboxes
 */

object Gaussian{
  
  
  def exponent(x: BDV[Double], mu: BDV[Double], sigma: BDV[Double]) : Double = {
    var dim:Int = x.length

     var I = BDM.eye[Double](dim)
    for (i<-0 until dim){
      I(i, i) = 1.0 / (sigma(i)*sigma(i))
    }
    var det:Double = -0.5* ((x-mu).t * I * (x-mu))
    var expdet = Math.exp(det)
    return expdet
  }
  
  def f(x: BDV[Double], alpha: Double, mu: BDV[Double], sigma: BDV[Double]) : Double = {
    var dim:Int = x.length

    var res: Double = 0.0
    var I = BDM.eye[Double](dim)
    for (i<-0 until dim){
      I(i, i) = 1.0 / (sigma(i)*sigma(i))
    }
    var det:Double = -0.5* ((x-mu).t * I * (x-mu))
    var expdet = Math.exp(det)
    res = alpha * expdet
 
    return res
  }
  
   def df_alpha(x: BDV[Double], alpha: Double, mu: BDV[Double], sigma: BDV[Double]) : Double = {
     return f(x, 1.0, mu, sigma)
   }
  
   def df_mu(x: BDV[Double], alpha: Double, mu: BDV[Double], sigma: BDV[Double]) : BDV[Double] = {
      var dim:Int = x.length
      var g:BDV[Double] = BDV.zeros[Double](dim)
      val ex:Double = exponent(x, mu, sigma)
      var sc:Double =  alpha
      for (i<-0 until mu.length){
        g(i) = sc *(1.0 / (sigma(i)*sigma(i)) )*(x(i)-mu(i))*ex
      }
      
     return g
   }
   
  def df_sigma(x: BDV[Double], alpha: Double, mu: BDV[Double], sigma: BDV[Double]) : BDV[Double] = {
      var dim:Int = x.length
      var g:BDV[Double] = BDV.zeros[Double](dim)
      val ex:Double = exponent(x, mu, sigma)
      var sc: Double = alpha
      for (i<-0 until mu.length){
        g(i) = (1.0/(sigma(i)*sigma(i)*sigma(i)))* sc *(x(i)-mu(i))*(x(i)-mu(i))*ex
      }
      
     return g
  }
}

class GaussianMixtureModel(inputDim: Int, numGauss: Int) extends Serializable  with NonlinearModel{
  var n:Int = inputDim
  var m:Int = numGauss
  var dim:Int = m*(n*2+1)

  
  def eval(w:BDV[Double], x: BDV[Double]): Double = {
    assert(x.size == n)
    assert(w.size == dim) 
   var res:Double = 0.0
    
    for (i<-0 until m){
      var ww:BDV[Double] = w((2*n+1)*i until (2*n+1)*(i+1))
      var alpha:Double = ww(n*2)
      var mu:BDV[Double] = ww(0 until n)
      var sigma:BDV[Double] = ww(n until 2*n)
      
  
      res = res + Gaussian.f(x, alpha, mu, sigma)
    }
   
    return res
  }
   
  def grad(w: BDV[Double], x: BDV[Double]): BDV[Double] = {
    assert(x.size == n)
    assert(w.size == dim)
     var gper : BDV[Double] = BDV.zeros(dim) // (2n+1)*m 
     for (i<-0 until m){
         // System.out.println("from " + (2*n+1)*i  + " to  " + ((2*n+1)*(i+1)-1))
          var ww:BDV[Double] = w((2*n+1)*i until (2*n+1)*(i+1))
          
          var alpha:Double = ww(n*2)
          var mu:BDV[Double] = ww(0 until n)
          var sigma:BDV[Double] = ww(n until 2*n)
          //System.out.println("alphaX " + (2*n) + " mu " + 0 + " to " + (n-1) + " sigma " + n + " to " + (2*n-1))
         // System.out.println("alpha0 " + ((2*n+1)*i+2*n) + " mu " + ((2*n+1)*i+0) + " to " + ((2*n+1)*i+n-1) + " sigma " + ((2*n+1)*i+n) + " to " + ((2*n+1)*i+2*n-1))
          
          var alpha_der = Gaussian.df_alpha(x, alpha, mu, sigma)
          var mu_der = Gaussian.df_mu(x, alpha, mu, sigma)
          var sigma_der = Gaussian.df_sigma(x, alpha, mu, sigma)
          gper((2*n+1)*(i+1)-1) += alpha_der
          gper((2*n+1)*i to (2*n+1)*i+ (n-1)) := gper((2*n+1)*i to (2*n+1)*i+(n-1)) +  mu_der
          gper((2*n+1)*i+n to (2*n+1)*i+n+ (n-1)) := gper((2*n+1)*i+n to (2*n+1)*i+n+ (n-1)) + sigma_der
          
          //System.out.println(mu_der.length + " "+  sigma_der.length + " " + x.length)
          
         // System.out.println("alpha1 " + ((2*n+1)*(i+1)-1) + " mu " + ((2*n+1)*i) + " to " + ((2*n+1)*i+ (n-1)) + " sigma " + ((2*n+1)*i+n) + " to " + ((2*n+1)*i+n+ (n-1)))
          
     }
    return gper
  }
  
  def gradnumer(w: BDV[Double], x: BDV[Double]) : BDV[Double] = {
    var h: Double = 0.000001
      var g:BDV[Double] = BDV.zeros(this.dim)
      var xtemp:BDV[Double] = BDV.zeros(this.dim)
      xtemp = w.copy
      var f0 = eval(xtemp, x)

      for (i<-0 until this.dim){
            xtemp = w.copy
            xtemp(i) += h
            var f1 = eval(xtemp, x)
            g(i) = (f1-f0)/h
      }
      return g
  }
  
  def getDim(): Int = {
    return dim
  }
  
   
}