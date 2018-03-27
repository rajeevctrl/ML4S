package com.ml4s.sklearn.linearmodels

import org.nd4j.linalg.api.ndarray.INDArray
import com.ml4s.np.{NP => np}
import com.ml4s.np.NP._
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import com.ml4s.sklearn.preprocessing.ScalarTransform

/**
 * This class performs Ordinary Least Squares to compute the parameters for given inputs.
 */
class LinearRegression(val fitIntercept:Boolean=true,normalize:Boolean=false) {
  private val logger:Logger = LoggerFactory.getLogger(classOf[LinearRegression])
  private val transformer:ScalarTransform = ScalarTransform(withMean=true, withStd=true)
  private var model: LinearRegressionModel = LinearRegressionModel(weights = np.zeros(shape = Array(1, 1)), 
      loss = 0.0, fitIntercept = this.fitIntercept, normalize = this.normalize)
  
//  private def initModel(weights:INDArray,loss:Double,fitIntercept:Boolean)={
//    this.model=new LinearRegressionModel(weights,loss,fitIntercept)
//  }
  /**
   * Returns weights of model (excluding intercept term)
   */
  def weights:INDArray = {
    val indicies = if(this.model.fitIntercept == true){
      Range(0,this.model.weights.shape()(0)-1)
    }else{
      Range(0,this.model.weights.shape()(0))
    }
    val rows = this.model.weights.getRows(indicies: _*)
    return rows
  }
  /**
   * Returns intercept term of model.
   */
  def bias:INDArray = {
    val intercept = if(this.model.fitIntercept == true){
      this.model.weights.getRow(this.model.weights.shape()(0)-1)
    }else{
      np.zeros(shape=Array(1,1))
    }
    return intercept
  }
  /**
   * Returns loss of model generated during training.
   */  
  def trainLoss:Double= return this.model.loss
  
  def fit(X:INDArray, Y:INDArray):Unit={
    var X_train = if(this.normalize==true) this.transformer.fitTransform(X) else X
    //If intercept is to be added than add extra column of 1's to X.
    X_train = if(fitIntercept==true){
      val biasCol = np.scalars(1, shape=Array(X_train.shape()(0),1))
      np.hstack(X_train,biasCol)
    }else{
      X_train
    }    
//    val X_train = X
//    val bias = if(fitIntercept==true) np.random.randn(shape=Array(1,1)).getDouble(0,0)
    //Convert Y to column vector.
    val Y_train=if(Y.shape()(0)>1 && Y.shape()(1)==1){
      Y
    }else if(Y.shape()(0)==1 && Y.shape()(1)>1){
      val y=np.reshape(Y, newshape=Array(Y.shape().reverse: _*))
      y
    }else{
      throw new Exception(s"Y should be a vector. However shape of Y is ${Y.shape().toList}.")
    }
    logger.info("Training OLS model.")
    val xTx = np.matmul(X_train.transpose(), X_train)
    val inv = np.pinv(xTx)
    val xTy = np.matmul(X_train.transpose(), Y_train)
    val weights = np.matmul(inv, xTy)
    //Update model parameters.
    //this.initModel(weights, 0.0, fitIntercept)
    this.model.weights = weights
    val loss=this.score(X, Y)  
    this.model.loss=loss
    logger.info("Training finished.")
  }
  
  def predict(X:INDArray):INDArray={
    var X_test = if(this.model.normalize==true) this.transformer.transform(X) else X
    X_test = if(this.model.fitIntercept==true){
      val xWithBias = if(this.model.weights.shape()(0) == X_test.shape()(1)){
        //If weights a equal to number of columns in X then no change is needed.
        X_test
      }else{
        val biasCol = np.scalars(1, shape=Array(X_test.shape()(0),1))
        np.hstack(X_test,biasCol)
      }
      xWithBias
    }else{
      X_test
    }
    val prediction = np.matmul(X_test,this.model.weights)
    return prediction
  }
  
  def score(X:INDArray,Y:INDArray):Double={
    //val pred = np.matmul(X, this.model.weights)
    val pred=this.predict(X)
    val residue = np.subtract(pred, Y)
    val squared_residue = np.multiply(residue, residue)
    val loss = squared_residue.sum(-1).getDouble(0,0)
    return loss
  }
  
  override def toString():String={
    return s"LinearRegression(fitIntercept = ${this.fitIntercept}, normalize = ${this.normalize})"
  }
  
  private def addBias(X:INDArray)={
//    val biasCol = np.scalars(1, shape=Array(X_train.shape()(0),1))
//    np.hstack(X_train,biasCol)
//    X.addColumnVector(columnVector)
  }

  /**
   * Class representing model params of Linear Regression.
   */
  case class LinearRegressionModel(
    var weights: INDArray,
    var loss: Double,
    var fitIntercept: Boolean,
    var normalize:Boolean)
}

object LinearRegression{
  
  def apply(fitIntercept:Boolean = true,normalize:Boolean = false)={
    new LinearRegression(fitIntercept,normalize)
  }
}


