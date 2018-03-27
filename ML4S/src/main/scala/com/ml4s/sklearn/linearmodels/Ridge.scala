package com.ml4s.sklearn.linearmodels

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import com.ml4s.sklearn.preprocessing.ScalarTransform
import org.nd4j.linalg.api.ndarray.INDArray

import com.ml4s.np.{NP => np}
import com.ml4s.np.NP._

class Ridge(
  val alpha: Double = 0.0,
  val fitIntercept: Boolean = true,
  val normalize: Boolean = false,
  val maxIter: Int = 1000) {
  
  private val logger:Logger = LoggerFactory.getLogger(classOf[Ridge])
  private val transformer:ScalarTransform = ScalarTransform(withMean=true, withStd=true)
  private var model: RidgeModel = RidgeModel(weights = np.zeros(shape = Array(1, 1)), 
      loss = 0.0, fitIntercept = this.fitIntercept, normalize = this.normalize, alpha = this.alpha)
  
  def fit(X:INDArray, Y:INDArray):Unit={
    
  }
  
  
  override def toString():String={
    return s"Ridge(alpha = ${this.alpha}, fitIntercept = ${this.fitIntercept}, normalize = ${this.normalize}, maxIter = ${this.maxIter})"
  }
  
   /**
   * Class representing model params of Linear Regression.
   */
  case class RidgeModel(
    var weights: INDArray,
    var loss: Double,
    var fitIntercept: Boolean,
    var normalize:Boolean,
    var alpha:Double)
  
}

object Ridge{
  def apply(
    alpha: Double = 0.0,
    fitIntercept: Boolean = true,
    normalize: Boolean = false,
    maxIter: Int = 1000) = {
    new Ridge(alpha, fitIntercept, normalize, maxIter)
  }
}