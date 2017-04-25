package com.ml4s.costfunctions

import org.nd4j.linalg.api.ndarray.INDArray
import com.ml4s.hypothesis.Hypothesis

class CostFunction{
  def costFunction(X:INDArray, 
                             Y:INDArray, 
                             theta:INDArray,
                             numTrainingExamples:Int,
                             featureCount:Int,
                             hypothesis:Hypothesis):Double={ return 0.0}
                             
  def firstDerivativeCostFunction(X:INDArray, 
                             Y:INDArray, 
                             theta:INDArray,
                             numTrainingExamples:Int,
                             featureCount:Int,
                             hypothesis:Hypothesis):INDArray={ return null}
}

/*object CostFunctions extends CostFunctions {
  
  type costFunction = (INDArray, 
                    INDArray, 
                    INDArray,
                    Int,
                    Int,
                    Hypothesis) => Double
                    
  type firstDerivativeCostFunction = (INDArray, 
                             INDArray, 
                             INDArray,
                             Int,
                             Int,
                             Hypothesis) => INDArray
}*/

