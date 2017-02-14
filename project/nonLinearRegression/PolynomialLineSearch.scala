package AccurateML.nonLinearRegression



import breeze.linalg.{DenseVector => BDV}
import breeze.optimize._

/**
 * @author nodalpoint
 */
class PolynomialLineSearch(initfval: Double,
                             maxIterations: Int = 30,
                             shrinkStep: Double = 0.5,
                             growStep: Double = 2.1,
                             cArmijo: Double = 1E-4,
                             cWolfe: Double = 0.9,
                             minAlpha: Double = 1E-10,
                             maxAlpha: Double = 1E10,
                             enforceWolfeConditions: Boolean = true,
                             enforceStrongWolfeConditions:Boolean = false) extends ApproximateLineSearch {

  var fcalls:Int = 0
  
   def getFcalls():Int  = {
    return this.fcalls
  }
  
  def  qmin(x1: Double, f1: Double, g1: Double, x2: Double, f2: Double) : (Double, Int) = {
    var dx:Double = x2-x1
    var df:Double  = f2-f1
    var b:Double = (df/dx-g1)/dx
    var xmin: Double = 0.0d
    var nr: Int = 0
    if (b > 0.0d){
      xmin = x1 - g1 / (2.0*b)
      nr = 1
    }else
      nr = 0
     
    (xmin, nr)
}
  
def cmin(x1: Double, f1: Double, g1: Double, x2: Double, f2: Double, g2: Double) : (Double, Int) = {
  var  h = x2-x1
  var  a = (g2+g1)/(h*h) - 2.0*(f2-f1)/(h*h*h)
  var  b =(3.0*(f2-f1) - h*(g2+2.0*g1))/(h*h)
  var d = b*b - 3.0*a*g1
  var xmin: Double = 0
  var nr: Int =  0

  if (d >= 0.0 ){
    if (a != 0.0){
      xmin = x1 + (-b + Math.sqrt(d))/(3.0*a)
      nr = 1;
    }else{
      if (b > 0.0){
        xmin = x1 - g1/(2.0*b)
        nr = 1
      }else{
        nr=0
      }
    }
  }else{
    nr=0
  }
  
  (xmin, nr)
}  
  

def qmin2(x1: Double, f1: Double, x2: Double, f2: Double, x3: Double, f3: Double): (Double, Int) = {
  var  d1 = x2-x1
  var  d2 = x1-x3
  var  d3 = x3-x2

  var  df1 = f2-f1
  var  df2 = f3-f2

  var a = (df1/d1 - df2/d3)/d2
  var xmin:Double = 0.0d
  var nr:Int = 0

  if (a > 0.0){
    var anu = df1*d3*d3 + df2*d1*d1
    var den = df1*d3 - df2*d1
    xmin = x2 + anu/(2.0*den)
    nr = 1
  }else{
    nr = 0
  }
  
  (xmin, nr)
}

def cmin2 (x1: Double, f1: Double, g1: Double, x2: Double, f2: Double, x3: Double, f3: Double):(Double, Int) = {
  var w1 = (f2-f1-(x2-x1)*g1)/Math.pow(x2-x1, 2.0)
  var w2 = (f3-f1-(x3-x1)*g1)/Math.pow(x3-x1, 2.0)
  var a = (w2-w1)/(x3-x2)
  var b = (x2-x1)*w2/(x2-x3) - (x3-x1)*w1/(x2-x3)
  var d = Math.pow(b, 2.0) - 3*a*g1
  var xmin: Double = 0.0d
  var nr: Int = 0

  if (d >= 0.0 ){
    if (a != 0.0){
      xmin = x1 + (-b + Math.sqrt(d))/(3.0*a)
      nr = 1
    }else{
      if (b > 0.0){
        xmin = x1 - g1/(2.0*b)
        nr = 1
      }else{
        nr=0
      }
    }
  }else{
    nr=0
  }
  
  (xmin, nr)
}


def keept(a: Double, al: Double, t1: Double, t2: Double, xmin: Double): Double = {
  var xleft:Double = 0.0d
  var xright:Double = 0.0d
  var aalen = Math.abs(a-al)
  var xminOut: Double = 0.0d
  if (a < al){
    xleft = a + t1*aalen
    xright = al - t2*aalen
  }else{
    xleft = al + t2*aalen
    xright = a - t1*aalen
  }

  if (xmin < xleft)
    xminOut = xleft
  else if (xmin > xright)
    xminOut = xright
  else  
    xminOut = xmin
    
    xminOut
}


def keepe(a: Double, al: Double, xmin: Double, b: Double, nob: Boolean, t3: Double, t4: Double, t5: Double, t6: Double): Double = {
  var xleft: Double = 0.0d
  var xright: Double = 0.0d
  var alblen = Math.abs(al-b)
  var xminOut: Double = 0.0d

  if (a < al){
    if (nob == true){
      xleft = t3*al
      xright = t4*al
    }else{
      xleft = al + t5*alblen
      xright = b - t6*alblen
    }
  }else{
    xleft = b + t6*alblen
    xright = al - t5*alblen
  }

  if (xmin < xleft)
    xminOut = xleft
  else if (xmin > xright)
    xminOut = xright
  else 
    xminOut = xmin 
    
    xminOut
}

 override def minimize(f: DiffFunction[Double], init: Double = 1.0) : Double = {
    var al:Double = init
    var fal: Double = 0
   var  gal:Double = 0
   var  b: Double = 0
   var fb: Double = 0
   var xmin: Double = 0
   var n = 1
   var nr: Int = 0
  //
  // Some constants
  var one:Double = 1.0
  var ten:Double = 10.0

  //
  // Truncation constants
  var t1:Double = 0.1
  var t2:Double = 0.5
  var t3:Double = 1+t1
  var t4:Double = 10.0
  var t5:Double = 0.1
  var t6:Double = 0.5

  //
  // flow  is the lowest function value so far. It need not satisfy
  // the line-search conditions and is returned whenever a normal
  // return is not possible.
  var (f0, g0) = f.calculate(0.0)
  var flow:Double = f0
  this.fcalls = 1
  var allow:Double = 0.0
  
  if (g0 >= 0.0){
    System.out.println("Line search: Positive directional derivative ");
    al = allow
    fal = flow
    return al
  }
    
//
  // Obtain a lower bound on AL
  
  var alamin: Double = minAlpha

  //
  // Initial values for the left bracket
  var a: Double = 0.0
  var fa: Double = f0
  var ga: Double = g0

  //
  // NOB = .TRUE. indicates that a right bracket has not been found.
  var nob: Boolean = true;

  //
  //  IBK is the number of backtracks towards the left bracket.
  //  Note that the left bracket always satisfies the rho-condition.
  var ibk: Int = 0;

  //
  //  ITER is number of iterations of the method.
  var iter: Int  = 0;
  //
  var forceb: Boolean = false;
  var iside: Int = 0;    
  
  
    //
  // Begin iterations
  while (true){

     iter = iter + 1;
     //
    //  If AL is too small, terminate the line search, since convergence
    //  would be detected anyway by the calling routine. (X-convergence)
    if (al < alamin) {
      al = allow
      fal = flow
      System.out.println("al < alamin")
      return al
    }

    //
    //  Is AL getting too close on the A-bracket ?
    //  This is probably an indication of severe round off error near
    //  the solution.
    if (Math.abs(a-al)/Math.max(a, one) < alamin){
      al = allow
      fal = flow
      System.out.println("Is AL getting too close on the A-bracket")

      return al
    }

    //
    // Are we getting out of the right margin ?
    if (al > maxAlpha){
       al = allow
       fal = flow
      return al
    }
  

    //
    // Evaluate the objective
    fal = f.calculate(al)._1
    this.fcalls += 1

   

    if(fal < flow){
      flow = fal
      allow = al
    }

  /*  if (fal < this.ftarget){
      return al;
    }*/

    //
    // If we are above rho-line or greater than the A-bracket

     if (fal > f0 + al * cArmijo * g0 || fal >= fa){
       ibk = ibk + 1
      if (ibk == 1){

        if (nob == true){
          iside = 0
        }else{
          iside = 0
          if (b - a < 0.0) 
            iside = 1
        }
        //
        //This is the first backtrack. Use a parabola.
        //
        var q = qmin(a, fa, ga, al, fal)
        xmin = q._1
        nr = q._2
        if (nr == 0){
          // Interpolation has failed
          al = allow
          fal = flow
          System.out.println("interpolation failed 1")
         
          return al
        }
        var  aal = Math.abs(a - al);
        if (iside == 0){
          if (xmin < a + t1*aal)
            xmin = a + t1*aal
        }
        else{
          if (xmin > a - t1*aal)
            xmin = a - t1 * aal
        }
       }else{

        //
        // Second and subsequent backtracks, use a cubic polynomial
        var c = cmin2(a, fa, ga, al, fal, b, fb)
        xmin = c._1
        nr = c._2 
        if (nr == 0){
          var q = qmin(a, fa, ga, al, fal)
          xmin = q._1
          nr = q._2 
          if (nr == 0){
            al = allow
            System.out.println("interpolation failed 2")
            
            return al
          }
        }
        xmin = keept(a, al, t1, t2, xmin)
 
      }
      b = al
      fb = fal
      nob = false
  
      //
      // Below rho-line and  less than the A-bracket
    }else{
 
      ibk = 0;

      // Evaluate the directional derivative
      gal = f.calculate(al)._2
      this.fcalls += 1
     
      //
      // Algorithm weak Wolfe (1)
      if (enforceWolfeConditions == true){
        if (gal >= cWolfe*g0){
 
           return al
        }
      }
      //
      // Algorithm strong Wolfe (2)
      else{
        if (Math.abs(gal) <= -cWolfe * g0){  
 
          return al
        }
      }

      //
      // The gradient criterion is not satisfied
      if (nob == true){
        iside = 0;
        if (gal > 0.0) iside=1
      }else{
        iside = 1;
        if ( (b-a)*gal < 0.0 ) iside = 0
      }

      //
      // Iside should be 0 for weak Wolfe
      if (enforceWolfeConditions == true  && iside == 1){
        al = allow;
        fal = flow;
        System.out.println("never never  ")

        return al
      }

      //
      // iside = 0
      if (iside == 0){
        if (gal > ga){
          var c = cmin(a, fa, ga, al, fal, gal)
          xmin = c._1
          nr = c._2
          if (nr == 0){
            var aa = (ga-gal)/(2.0*(a-al))
            var bb = ga - 2.0*aa*a
            xmin  = -bb / (2.0*aa)
          }
          xmin = keepe(a, al, xmin, b, nob, t3, t4, t5, t6);
        }
        //
        // The function is concave.
        // Interpolating schemes are no good if the slopes indicate that this
        // is a concave part of the function. (A maximum would actually be
        //found.)
        else{
          if (nob == true){
            xmin = t4*al;
          }else{
            var c = cmin2(al, fal, gal, a, fa, b, fb)
            xmin = c._1
            nr = c._2
            if (nr == 0){
             var q = qmin(al, fal, gal, b, fb)
             xmin = q._1
             nr = q._2
              if (nr == 0){
                al = allow
                fal = flow
                System.out.println("interpolation failed 3")
                
                return al
              }
            }
            xmin = keepe(a, al, xmin, b, nob, t3, t4, t5, t6);
          }
        }
      }
      //
      // other side
      else{
        var c = cmin(a, fa, ga, al, fal, gal)
        xmin = c._1
        nr = c._2
        if (nr == 0){
          var q = qmin(al, fal, gal, a, fa)
          xmin = q._1
          nr = q._2
          if (nr == 0){
            al = allow
            fal = flow
             System.out.println("interpolation failed 4")
            
            return al
          }
        }
        xmin = keept(a, al, t1, t2, xmin)
        b = a
        fb = fa
        nob = false
      }

      a = al
      fa = fal
      ga = gal
    }
    al = xmin

    if (iter >= maxIterations){
      al = allow
      fal = flow
       System.out.println(" max iterations ")

      return al
    }


  }
  return al
    
  }
 
  def iterations(f: DiffFunction[Double], init: Double = 1.0): Iterator[State] = {
    System.out.println("Oxi edw")
    
    val iter = Iterator.empty
    
    return iter
  }

}