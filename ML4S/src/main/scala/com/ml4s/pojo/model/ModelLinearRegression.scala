package com.ml4s.pojo.model


//case class ModelLinearRegression(val theta:Array[Double], val hypothesis:HypothesisLinear, val costFunction:CostFunctionSquaredDiff) extends ModelRegression

case class ModelLinearRegression(val theta:Array[Double],val trainingError:Double) extends ModelRegression

