package com.ml4s.sklearn.pojo

import org.nd4j.linalg.api.ndarray.INDArray

case class Model(var weights:INDArray,
    var lr:Double,
    var loss:Double,
    var valLoss:Double,
    var acc:Double,
    var valAcc:Double)

