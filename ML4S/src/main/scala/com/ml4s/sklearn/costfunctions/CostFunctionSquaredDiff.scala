package com.ml4s.sklearn.costfunctions

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import com.ml4s.hypothesis.Hypothesis
import com.ml4s.hypothesis.HypothesisLinear

class CostFunctionSquaredDiff extends CostFunction {
  
  
  /**
   * This cost function depends on squared difference i.e. SUM(hypothesis-Y)
   * 
   * @param X: Feature matrix of training data. Dimensions of matrix are NUM_TRAINING_EXAMPLES x DIMENSIONS
   * @param Y: Output vector of training data. This is a column vector with dimensions NUM_TRAINING_EXAMPLES x 1
   * @param theta: Column vector of weights. Dimensions are DIMENSIONS x 1
   * @param numTrainingExamples: Total number of training examples.
   * @param featureCount: Total number of dimensions/features.
   * @param hypothesis: A function representing hypothesis function.
   * 
   * @return Double value representing cost. 
   */
  override def costFunction(X:INDArray, 
                             Y:INDArray, 
                             theta:INDArray,
                             numTrainingExamples:Int,
                             featureCount:Int,
                             hypothesis:Hypothesis):Double={
    val N=X.size(1)  //Count of dimension-1 i.e. rows.
    val P=X.size(0)  //Count of dimension-0 i.e. columns.
    //Compute hypothesis value of datapoints.
    val hypothesisMat:INDArray=hypothesis.hypothesis(X,theta);
    //Compute squared difference between hypothesis/predicted values and actual values Y.
    val residuals:INDArray=hypothesisMat.sub(Y);
    var cost:Double=(residuals.mul(residuals)).sumNumber().doubleValue();
    cost=cost/(2.0d * N.toDouble)
    return cost;    
  }
  
  /**
   * This function computes first derivative of cost function w.r.t weights.
   * 
   * @param X: Feature matrix of learning data.
   * @param Y: Output vector of learning data.
   * @param theta: Column vector of existing weights.
   * @param numTrainingExamples: Total number of training examples.
   * @param featureCount: Total number of features/dimensions.
   * @param hypothesis: An object of hypothesis to be used. This should be an object of class inheriting trait {@code Hypothesis}
   * 
   * @return A column vector representing partial derivative of error function with each weight. 
   */
  override def firstDerivativeCostFunction(X:INDArray, 
                             Y:INDArray, 
                             theta:INDArray,
                             numTrainingExamples:Int,
                             featureCount:Int,
                             hypothesis:Hypothesis):INDArray={
    val N=numTrainingExamples
    val P=featureCount
    //Compute hypothesis value of datapoints.
    val hypothesisMat:INDArray=hypothesis.hypothesis(X,theta);
    //Diff of hypothesis from actual output.
    val firstDerivativeHypo=hypothesis.hypothesisFirstDerivative(X, theta);
    var residuals=hypothesisMat.sub(Y)
    //Concatenate multiple columns of residuals to make a matrix equal in dimension to P.
    for(i <- Range(0,P-1)){
      residuals=Nd4j.hstack(residuals,residuals)
    }    
    val derivatives=residuals.mul(firstDerivativeHypo)
    val dimensionalDerivatives=(derivatives.sum(0)).div(N);
    //Convert row vector of derivative values to column vector.
    val derivativeCol=dimensionalDerivatives.transpose();
    return derivativeCol;
  }
  
  
  
}