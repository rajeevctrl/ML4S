package com.ml4s.pojo.model

import com.ml4s.hypothesis.Hypothesis
import com.ml4s.sklearn.costfunctions.CostFunction

trait ModelRegression {
  val theta:Array[Double]
  val trainingError:Double
  //val hypothesis:Hypothesis
  //val costFunction:CostFunction
}