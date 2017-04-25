package com.ml4s.regression

import org.nd4j.linalg.api.ndarray.INDArray
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import org.nd4j.linalg.inverse.InvertMatrix
import com.ml4s.hypothesis.Hypothesis
import org.nd4j.linalg.factory.Nd4j
import com.ml4s.costfunctions.CostFunction
import scala.collection.mutable.Queue
import com.ml4s.util.CommonUtil

class LinearRegression {
  private val logger:Logger=LoggerFactory.getLogger(classOf[LinearRegression])
  
  private var thetas:INDArray= _
  private var trainingError:Double = _
  private var learningDataSize:Int = _
  private var featureCount:Int = _
  
  /*
   * Getter/Setter methods.
   */
  def getTrainingError():Double=return this.trainingError
  /**
   * This function returns value of computed parameters/thetas/weights in form of INDArray.
   */
  def getThetaINDArray():INDArray = return thetas;  
  /**
   * This function returns value of computed parameters/thetas/weights in form of Array[Double].
   */
  def getThetaArray():Array[Double] ={
    val thetaArr=(for(i<-Range(0,this.featureCount,1)) yield {
      this.thetas.getDouble(i)
    }).toArray
    return thetaArr;
  }
  /**
   * This function sets value of theta from an array of doubles.
   */
  /*def setTheta(arr:Array[Double])={
    this.thetas=Nd4j.create(arr,Array(arr.length,1))
  }*/
  /**
   * This function sets value of thetas from an INDArray.
   */
  /*def setTheta(indArray:INDArray)={
    this.thetas=indArray
  }*/
  /**
   * This function returns count of total number of examples.
   */
  def getLearningDataSize():Int= return this.learningDataSize;
  /**
   * This function returns count of features/dimensions on which model is trained.
   */
  def getFeatureCount():Int= return this.featureCount;
  
  
  /**
   * This function takes feature matrix X and a column vector Y representing output.
   * It then computes linear regression model using Least squares method.
   * 
   * @param X: 2-D matrix consisting of datapoints.
   * @param Y: 1-D col vector representing output corresponding to each datapoint.
   */
  def fitLeastSquare(X:INDArray, Y:INDArray):Unit={
    val learningDataSize=X.size(0)
    val featureCount=X.size(1)
    val xTx=X.transpose().mmul(X)
    val xTx_inverse=InvertMatrix.invert(xTx, false)
    val xTy=X.transpose().mmul(Y)
    val thetas=xTx_inverse.mmul(xTy)    
    this.thetas=thetas;
    /*for(i <- Range(0,X.columns(),1)){
      this.weightCoefficients(i)=thetas.getDouble(i)
    }*/
  }

  /**
   * This function performs learning of parameters using Gradient Descent algorithm.
   * The algorithm will keep on learning till cost is reducing. As soon as as iteration is found
   * at which cost increases, learning will stop at that particular iteration.
   *
   * @param X: Input matrix of training data.
   * @param Y: Output column vector of training data.
   * @param alpha: Learning Rate.
   * @param iter: Number of iterations for which learning is to be performed.
   */
  def fitGD(X: INDArray, 
            Y: INDArray, 
            alpha: Double, 
            iter: Int, 
            hypothesis: Hypothesis, 
            errorFunc: CostFunction) = {
    val learningDataSize=X.size(0)
    val featureCount=X.size(1)
    var tempAlpha=alpha
    //Compute initial cost and initial values of thetas.
    var theta=Nd4j.create(CommonUtil.randomNumbersInRange(featureCount, 0.0, 0.000001),Array(featureCount,1))
    //var theta=Nd4j.randn(Array(featureCount,1))
    var initialCost=errorFunc.costFunction(X, Y, theta, learningDataSize, featureCount, hypothesis)
    this.logger.info(s"Iter:${0}  Training Error:${initialCost}")
    //Perform Gradient Descent for given number of iterations.
    var bestTheta:INDArray=null;
    var bestCost:Double=Double.MaxValue;
    val queue:Queue[Double]=new Queue[Double]()
    val queueSize:Int=3;
    val loop=new scala.util.control.Breaks
    loop.breakable(
      for (i <- Range(1, iter, 1)) {
        val firstDerivative=errorFunc.firstDerivativeCostFunction(X, Y, theta, learningDataSize, featureCount, hypothesis);
        val newTheta=theta.sub(firstDerivative.mul(tempAlpha));
        val newCost=errorFunc.costFunction(X, Y, newTheta, learningDataSize, featureCount, hypothesis);
        /*
        this.logger.info(s"Iter:${i}  Training Error:${initialCost}  alpha:${tempAlpha}")
        initialCost=newCost;
        theta=newTheta;
        if(newCost< bestCost){
          bestCost=newCost;
          //Update thetas.
          bestTheta=newTheta;
          
        }
        queue.enqueue(newCost)
        //Update theta.
        if(queue.size >=queueSize){
          val distinct=queue.distinct
          //If all elements in queue are same then decrease alpha.
          if(distinct.size==1){
            tempAlpha=tempAlpha-(5.0*alpha/100.0)
          }
          queue.dequeue()
        }*/
        
        //If new cost is less than previous cost then keep on learning.
        if(newCost< initialCost){
          initialCost=newCost;
          //Update thetas.
          theta=newTheta;
          this.logger.info(s"Iter:${i}  Training Error:${initialCost}")
        }else{
          this.logger.info(s"Iter:${i}  Training Error stopped reducing thus exiting learning process.")
          loop.break();
        }
      });
    this.thetas=theta;
    this.trainingError=initialCost
    /*this.thetas=bestTheta;
    this.trainingError=bestCost*/
    
  }
  
  
  def fitSGD(X:INDArray, 
             Y:INDArray, 
             alpha:Double, 
             iter:Int, 
             batchSize:Int, 
             numEpochs:Int, 
             hypothesis:Hypothesis, 
             errorFunc:CostFunction)={
    
  }
  
  
  /**
   * This function predicts output values for given data points.
   * 
   * @param X: INDArray for which we need to predict output.
   * @param theta: INDArray of parameters.
   */
  def  predict(X:INDArray, theta:INDArray,hypothesis:Hypothesis):INDArray={
    val predictedValues=hypothesis.hypothesis(X, theta);
    return predictedValues;
  }
  
  
  
}